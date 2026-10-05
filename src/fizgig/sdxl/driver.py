"""SDXL driver for Fizgig's standard layer (families/driver.py) - any SDXL checkpoint (Juggernaut XL, Illustrious,
Pony, RealVis, base SDXL ...), loaded from its single .safetensors file.

* model: diffusers' UNet2DConditionModel, AutoencoderKL and the two CLIP text encoders, read from the checkpoint
  (diffusers' single-file loader, with SDXL base 1.0's configs and tokenizers - the family's helper files)
* conditioning: both CLIPs' penultimate hidden states side by side -> {"hidden_states": (77, 2048)}, plus CLIP-G's
  pooled projection -> {"pooled": (1280,)}; SDXL's size conditioning (time_ids) is built from the image size
* latents: 4 channels at 1/8, scaled by the VAE's 0.13025
* training: the DDPM schedule SDXL was trained on (scaled-linear betas 0.00085-0.012, 1000 steps), uniform timesteps,
  epsilon prediction (or v-prediction for v-pred checkpoints, the "prediction" family option)
* sampling: DPM++ 2M SDE with Karras sigmas by default (the community's Juggernaut choice; its noise comes from the
  seed), or Euler with trailing spacing (option sampler=euler), both on SDXL's training schedule
* LoRA: the Linears of the 11 attention modules, named IN04 ... OUT05 as SDXL block-weight tools name them; kohya keys
  on diffusers module names (lora_unet_down_blocks_1_attentions_0_...), which ComfyUI maps
"""
import numpy as np
import torch
import torch.nn.functional as F

from fizgig.families.driver import Block, BlockGroup, FamilyDriver

CONFIG_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
DTYPE = torch.bfloat16

# the 11 attention modules, in run order, with SDXL block-weight names (input / middle / output)
ATTENTIONS = (
    ("IN04", "Input 4", "down_blocks.1.attentions.0"), ("IN05", "Input 5", "down_blocks.1.attentions.1"),
    ("IN07", "Input 7", "down_blocks.2.attentions.0"), ("IN08", "Input 8", "down_blocks.2.attentions.1"),
    ("MID", "Middle", "mid_block.attentions.0"),
    ("OUT00", "Output 0", "up_blocks.0.attentions.0"), ("OUT01", "Output 1", "up_blocks.0.attentions.1"),
    ("OUT02", "Output 2", "up_blocks.0.attentions.2"), ("OUT03", "Output 3", "up_blocks.1.attentions.0"),
    ("OUT04", "Output 4", "up_blocks.1.attentions.1"), ("OUT05", "Output 5", "up_blocks.1.attentions.2"),
)
# transformer blocks per attention module (SDXL's transformer_layers_per_block: 2 at 1/2, 10 at 1/4)
DEPTH = {"down_blocks.1": 2, "down_blocks.2": 10, "mid_block": 10, "up_blocks.0": 10, "up_blocks.1": 2}
TRANSFORMER_LINEARS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
                       "attn2.to_q", "attn2.to_k", "attn2.to_v", "attn2.to_out.0", "ff.net.0.proj", "ff.net.2")


# kohya / sd-scripts SDXL LoRAs (most community files) name the UNet in the original LDM layout:
# input_blocks_4_1_transformer_blocks_0_attn1_to_q -> down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_q
_LDM = {"input_blocks_4_1": "down_blocks_1_attentions_0", "input_blocks_5_1": "down_blocks_1_attentions_1",
        "input_blocks_7_1": "down_blocks_2_attentions_0", "input_blocks_8_1": "down_blocks_2_attentions_1",
        "middle_block_1": "mid_block_attentions_0",
        "output_blocks_0_1": "up_blocks_0_attentions_0", "output_blocks_1_1": "up_blocks_0_attentions_1",
        "output_blocks_2_1": "up_blocks_0_attentions_2", "output_blocks_3_1": "up_blocks_1_attentions_0",
        "output_blocks_4_1": "up_blocks_1_attentions_1", "output_blocks_5_1": "up_blocks_1_attentions_2"}


def _batched(key, v):
    """A conditioning tensor with its leading batch dim (hidden_states (77, 2048) -> (1, 77, 2048), pooled (1280,) ->
    (1, 1280))."""
    return v if v.dim() == (3 if key == "hidden_states" else 2) else v[None]


def _attention_modules(prefix):
    depth = DEPTH[prefix.rsplit(".attentions", 1)[0]]
    mods = [f"{prefix}.proj_in", f"{prefix}.proj_out"]
    for j in range(depth):
        mods += [f"{prefix}.transformer_blocks.{j}.{m}" for m in TRANSFORMER_LINEARS]
    return mods


class _TextEncoders:
    """SDXL's two CLIPs and their tokenizers, read from the checkpoint."""

    def __init__(self, path, device):
        from diffusers import StableDiffusionXLPipeline
        # the pipeline's single-file loader reads the CLIPs out of the checkpoint. It cannot be built without its UNet
        # and VAE, so those load on the CPU too and are dropped at once (about 10 s at the cache stage)
        pipe = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=DTYPE, config=CONFIG_REPO)
        pipe.unet = pipe.vae = None
        self.tok1, self.tok2 = pipe.tokenizer, pipe.tokenizer_2
        self.te1 = pipe.text_encoder.to(device).eval().requires_grad_(False)
        self.te2 = pipe.text_encoder_2.to(device).eval().requires_grad_(False)
        self.device = device
        del pipe

    @torch.no_grad()
    def encode(self, captions):
        out = []
        for cap in captions:
            if not cap.strip():
                # an empty prompt is all zeros (SDXL base's force_zeros_for_empty_prompt): an empty negative = none
                out.append({"hidden_states": torch.zeros(77, 2048, dtype=DTYPE),
                            "pooled": torch.zeros(1280, dtype=DTYPE)})
                continue
            ids1 = self.tok1([cap], padding="max_length", max_length=77, truncation=True,
                             return_tensors="pt").input_ids.to(self.device)
            ids2 = self.tok2([cap], padding="max_length", max_length=77, truncation=True,
                             return_tensors="pt").input_ids.to(self.device)
            o1 = self.te1(ids1, output_hidden_states=True)
            o2 = self.te2(ids2, output_hidden_states=True)
            h = torch.cat([o1.hidden_states[-2], o2.hidden_states[-2]], dim=-1)[0]       # (77, 2048)
            out.append({"hidden_states": h.to(DTYPE).cpu(), "pooled": o2.text_embeds[0].to(DTYPE).cpu()})
        return out

    def unload(self):
        self.te1.to("cpu")
        self.te2.to("cpu")
        del self.te1, self.te2
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SDXLDriver(FamilyDriver):

    # ---- models ---------------------------------------------------------------------------------
    def load_dit(self, path, device):
        from diffusers import UNet2DConditionModel
        unet = UNet2DConditionModel.from_single_file(path, config=CONFIG_REPO, subfolder="unet", torch_dtype=DTYPE)
        return unet.to(device).eval().requires_grad_(False)

    def load_vae(self, path, device):
        """The checkpoint's VAE, or a separate SDXL VAE file (the Preferences override). fp32: SDXL's VAE overflows in
        half precision."""
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_single_file(path, config=CONFIG_REPO, subfolder="vae", torch_dtype=torch.float32)
        return vae.to(device).eval().requires_grad_(False)

    def load_text_encoder(self, path, device):
        return _TextEncoders(path, device)

    def unload_text_encoder(self, te):
        te.unload()

    def enable_gradient_checkpointing(self, dit, on=True):
        if on:
            dit.enable_gradient_checkpointing()
        else:
            dit.disable_gradient_checkpointing()

    # ---- encoding -------------------------------------------------------------------------------
    @torch.no_grad()
    def encode_images(self, vae, images):
        x = torch.stack([torch.from_numpy(np.ascontiguousarray(a[..., :3])) for a in images])
        x = x.permute(0, 3, 1, 2).float().div(127.5).sub(1.0).to(next(vae.parameters()).device)
        z = vae.encode(x).latent_dist.mode() * vae.config.scaling_factor
        return [l.to(DTYPE).cpu() for l in z]

    @torch.no_grad()
    def encode_text(self, te, captions):
        return te.encode(captions)

    # ---- the schedule ---------------------------------------------------------------------------
    def _alphas(self, device):
        a = getattr(self, "_ac", None)
        if a is None or a.device != torch.device(device):
            betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000, dtype=torch.float64) ** 2
            a = self._ac = torch.cumprod(1.0 - betas, dim=0).float().to(device)
        return a

    def _v_pred(self):
        return str(self.options.get("prediction", "epsilon")) == "v"

    @staticmethod
    def _time_ids(height, width, device, n=1):
        return torch.tensor([[height, width, 0, 0, height, width]] * n, device=device, dtype=DTYPE)

    def _unet(self, dit, x, t, cond, height, width):
        n = x.shape[0]
        hs = _batched("hidden_states", cond["hidden_states"]).to(x.device, DTYPE)
        pooled = _batched("pooled", cond["pooled"]).to(x.device, DTYPE)
        if hs.shape[0] != n:
            hs, pooled = hs.expand(n, -1, -1), pooled.expand(n, -1)
        return dit(x.to(DTYPE), t, encoder_hidden_states=hs,
                   added_cond_kwargs={"text_embeds": pooled, "time_ids": self._time_ids(height, width, x.device, n)},
                   return_dict=False)[0]

    # ---- training -------------------------------------------------------------------------------
    def training_loss(self, dit, latents, cond, generator, *, min_t=0.0, max_t=1.0, refs=None, diff_ref=None,
                      diff_weight=0.0):
        device = latents.device
        x0 = latents.float()
        h, w = x0.shape[-2] * 8, x0.shape[-1] * 8
        lo, hi = int(round(min_t * 999)), max(int(round(min_t * 999)) + 1, int(round(max_t * 999)) + 1)
        t = int(torch.randint(lo, min(hi, 1000), (1,), generator=generator).item())
        noise = torch.randn(x0.shape, generator=generator)
        off = float(self.options.get("noise_offset") or 0.0)
        if off:
            # SDXL base was trained with a 0.0357 offset (sd-scripts docs/train_network_advanced.md)
            noise = noise + off * torch.randn((x0.shape[0], x0.shape[1], 1, 1), generator=generator)
        noise = noise.to(device)
        a = self._alphas(device)[t]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise
        pred = self._unet(dit, xt, torch.tensor([t], device=device), cond, h, w).float()
        target = (a.sqrt() * noise - (1 - a).sqrt() * x0) if self._v_pred() else noise
        loss = F.mse_loss(pred, target)
        gamma = float(self.options.get("min_snr") or 0.0)
        if gamma:
            # Min-SNR weighting (Hang et al. 2023), the SDXL trainers' default gamma 5
            snr = float(a / (1 - a))
            loss = loss * (min(snr, gamma) / ((snr + 1) if self._v_pred() else snr))
        return loss, {"t": t / 999.0}

    # ---- sampling -------------------------------------------------------------------------------
    @torch.no_grad()
    def initial_noise(self, seed, width, height):
        g = torch.Generator("cpu").manual_seed(int(seed))
        return torch.randn((1, 4, height // 8, width // 8), generator=g, dtype=torch.float32)

    def _scheduler(self, sampler="dpmpp_2m_sde_karras"):
        from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler
        common = dict(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000,
                      prediction_type="v_prediction" if self._v_pred() else "epsilon")
        if sampler == "euler":
            return EulerDiscreteScheduler(timestep_spacing="trailing", **common)
        return DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++", solver_order=2, use_karras_sigmas=True,
                                           **common)

    @torch.no_grad()
    def generate(self, dit, cond, width, height, *, steps, seed, cfg=1.0, neg_cond=None, sigmas=None, options=(),
                 noise=None, on_step=None, refs=None, **_ignored):
        device = next(dit.parameters()).device
        sch = self._scheduler(str(dict(options or ()).get("sampler", "dpmpp_2m_sde_karras")))
        sch.set_timesteps(int(steps), device=device)
        g = torch.Generator("cpu").manual_seed(int(seed) + 1)      # the SDE sampler's noise, from the seed
        x = (self.initial_noise(seed, width, height) if noise is None else noise).to(device) * sch.init_noise_sigma
        use_cfg = cfg > 1.0
        if use_cfg:
            # SDXL's empty-prompt negative is zeros (the base pipeline's force_zeros_for_empty_prompt)
            neg = neg_cond if neg_cond is not None else {k: torch.zeros_like(v) for k, v in cond.items()}
            both = {k: torch.cat([_batched(k, neg[k]), _batched(k, cond[k])]).to(device)
                    for k in ("hidden_states", "pooled")}
        for i, t in enumerate(sch.timesteps):
            if on_step is not None:
                on_step(i, len(sch.timesteps))
            xin = sch.scale_model_input(torch.cat([x, x]) if use_cfg else x, t)
            pred = self._unet(dit, xin, t, both if use_cfg else cond, height, width).float()
            if use_cfg:
                pu, pc = pred.chunk(2)
                pred = pu + cfg * (pc - pu)
            x = sch.step(pred, t, x, return_dict=False, **({"generator": g} if "generator" in self._step_args(sch)
                                                              else {}))[0]
        return x

    @staticmethod
    def _step_args(sch):
        import inspect
        return inspect.signature(sch.step).parameters

    def pad_conditioning(self, conds):
        """Both CLIPs always give 77 tokens, so prompts blend as they are (prompt travel)."""
        return list(conds)

    @torch.no_grad()
    def decode(self, vae, latents, width, height):
        from PIL import Image
        p = next(vae.parameters())
        img = vae.decode(latents.to(p.device, p.dtype) / vae.config.scaling_factor).sample[0].float().clamp(-1, 1)
        return Image.fromarray(((img.permute(1, 2, 0).cpu().numpy() + 1) * 127.5).round().astype(np.uint8))

    def compile_blocks(self, dit, boundary="outside", blocks_to_swap=0):
        """torch.compile every transformer block (70, spread over the attention modules' own lists), in place, after
        the LoRA has wrapped its Linears. Diffusers' gradient checkpoint stays around each compiled block, so memory
        is unchanged. SDXL's training step is launch-bound - thousands of small kernels a step - and compiling fuses
        them: a real 1 MP run on a 5090 went 1.03 -> 0.75 s/step with fused AdamW, memory unchanged."""
        import logging
        from fizgig.families.compile import ready_to_compile
        if not ready_to_compile(blocks_to_swap):
            return
        lists = [m.transformer_blocks for m in dit.modules()
                 if isinstance(getattr(m, "transformer_blocks", None), torch.nn.ModuleList)]
        n = 0
        for blocks in lists:
            for i, blk in enumerate(blocks):
                blocks[i] = torch.compile(blk, fullgraph=False)
                n += 1
        logging.getLogger(__name__).info("[compile] %d SDXL transformer blocks compiled (the checkpoint stays "
                                         "outside) - the first step of each new shape pauses to compile", n)

    def alias_flat(self, flat):
        for ldm, diffusers in _LDM.items():
            if flat.startswith(ldm + "_"):
                return diffusers + flat[len(ldm):]
        return None

    # ---- LoRA and the block map -----------------------------------------------------------------
    def block_map(self, dit=None):
        names = {n for n, _ in dit.named_modules()} if dit is not None else None

        def blk(bid, label, prefix):
            mods = _attention_modules(prefix)
            return Block(bid, label, [m for m in mods if names is None or m in names])
        groups = (("Input blocks", ATTENTIONS[:4]), ("Middle", ATTENTIONS[4:5]), ("Output blocks", ATTENTIONS[5:]))
        return [BlockGroup(label, [blk(*a) for a in items]) for label, items in groups]
