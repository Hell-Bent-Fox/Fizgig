# Fizgig v6.5.1

A maintenance release on top of 6.5.0. The full 6.5.0 notes follow below, since Qwen Image 2.1 support is still new.

## What's new in 6.5.1

- **Klein 9B trains on 10–12 GB cards.** The Training tab now shows **Base precision** for Klein: Auto, 4-bit NF4 or fp8. Auto picks 4-bit on cards under 16 GB, where a Klein LoRA trains in about 8.5 GB with no block swap, and keeps fp8 on 16 GB and up.
- **Qwen previews use Viggle's turbo at its own settings**, strength 1.0 for 6 steps. See the note on the Qwen turbo LoRA below: for now, Turbo strength 0 and 25 steps may give better previews.
- **A sample seed of 0 means a random seed** for Qwen previews, as on the other trainers.
- **Resuming checks the whole saved state up front**, and a preview turbo LoRA that doesn't match the model is dropped with a warning instead of silently doing nothing. Both reported by [@mabseyuk](https://github.com/mabseyuk).
- **The Problem Images window explains what makes an image a problem:** hard images are often the most valuable early on, and only one that has stopped teaching the model gets throttled, recaptioned or set aside.
- **A shorter README**, with per-model, install, training and fine-tuning guides in `docs/`.

---

# From 6.5.0

Qwen Image 2.1 arrives in Fizgig: LoRA and LoKR training, turbo previews and every workbench tool, with Fizgig's own training adapter to keep Qwen training stable.

## Qwen Image 2.1 (experimental)

Qwen Image 2.1 is now a Base Model option on the Training tab, with the same toolset the other models have:

- LoRA or LoKR training, Adaptive LR or Automagic, EMA, Context LoRA, pause and resume.
- Fast previews during training with Viggle's 6-step turbo LoRA.
- The per-image loss watch: problem-image detection, per-image LR, look-outlier warm-up, and auto-recaption of stuck images with the same Qwen3-VL captioner the Captions tab uses.
- The live sample override panel.
- All five workbench tools: Repair Studio, LoRA the Explorer, Profiler, Extract and LoRA Royale.

Saved LoRAs, LoKRs, Repair Studio saves and Extract outputs all load in ComfyUI's standard LoRA loader.

**What to do:** on the Preferences tab, press **Download models for me** in the Qwen Image 2.1 section (about 34 GB). Then pick Qwen Image 2.1 as the Base Model on the Training tab and load a preset.

**Coming soon:** edit training for Qwen Image 2.1.

## Built on Fizgig's new model driver system

Qwen Image 2.1 is the first model added through Fizgig's new driver system. A model is now described once, as its files, its LoRA format and its model code behind a standard interface, and training, previews, downloads, memory planning and all five workbench tools work from that. It doesn't need changes spread through the app.

In the coming week we'll publish a guide to the system and open Fizgig to pull requests adding models through it: new models, and older ones Fizgig doesn't support yet. This will free me up on core features and stop me dreading a weekend when two models arrive at once, lol!

## The Fizgig training adapter

Qwen 2.1 LoRAs have a habit of collapsing into texture or wobbling part-way through training, and a lower loss doesn't warn you it's happening. The fix is a training adapter: a small LoRA that stays frozen and active while you train, steadying the run. It's switched off for previews and never ends up in your saved LoRA, so your LoRA works on the plain model.

We trained our own. It's trained at a higher resolution than the existing Qwen 2.1 training assistant, and LoRAs trained with it come out much sharper as well as fixing the collapse.

It's on by default in every Qwen preset and is downloaded with the models. It's also on Hugging Face: [ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter](https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter).

## Presets: train fast to keep Qwen's sharpness

Qwen renders very sharp images out of the box. A LoRA pulls fine detail such as skin texture toward whatever your dataset has, so if your photos are softer than what Qwen produces natively, a long run gradually trades Qwen's sharpness for your dataset's. Training faster keeps more of it.

In our testing, 0.5 MP (fyi see table below with what res .5mp is - its not 512x512 as many assume) is the sweet spot for this model: quicker than 1 MP, and it keeps more of Qwen's sharpness than 0.25 MP. All three presets train at 0.5 MP with adamw8bit, EMA 0.98 and the training adapter, and save every epoch for 30 epochs:

- **Qwen 2.1 Fast (rank 8, adaptive LR):** the default. Adaptive LR between 2e-4 and 4e-4. The quickest to likeness in our tests, and the best at holding skin detail.
- **Qwen 2.1 Standard (rank 16, adaptive LR):** more capacity for bigger or mixed datasets. Adaptive LR between 1e-4 and 2e-4, because rank 16 at Fast's rates overtrains.
- **Qwen 2.1 Style (rank 16, 1.5e-4):** a flat learning rate. Adaptive LR tends to climb on style datasets, and that's where styles overbake.

Every epoch is saved, so if a later epoch starts to look softer than you'd like, an earlier one is often the better pick.

**0.5 MP doesn't mean 512 pixels a side.** It's half a million pixels in total, about 704×704 for a square image. Fizgig buckets each image by its shape at that pixel count, for example:

| Aspect | Training size |
|---|---|
| 1:1 | 704×704 |
| 4:5 | 624×784 |
| 2:3 | 576×848 |
| 9:16 | 528×928 |
| 3:2 | 848×576 |
| 16:9 | 928×528 |

512×512 is 0.25 MP, the lowest option in the Target MP box.

## Qwen trains on 10 GB cards

10 GB is the minimum for Qwen Image 2.1.

**Base precision: Auto** picks bf16, INT8 or 4-bit NF4 at launch, and sizes Blocks Swap to match, from your free VRAM and your training resolution. It picks the most precise option that fits, and quantises before it swaps because swapping costs far more speed. On cards that can't hold the full text encoder, Fizgig loads an 8-bit version automatically.

On an RTX 5090 limited to 12 GB, the Fast preset trained on INT8 with no block swap and a 9.9 GB peak, previews included. Limited to 10 GB, it trained on 4-bit NF4 with a 7.3 GB peak. Real 10 and 12 GB cards make the same choices but run slower than the 5090's 0.9 to 1 s per step.

## A note on the Qwen turbo LoRA

Fizgig uses Viggle's current turbo LoRA for Qwen samples, at 6 steps and strength 1.0 (I've just updated Fizgig to that setting). That turbo isn't mature yet: it sometimes produces body horror that the same seed and prompt don't show without it. For now I'd almost recommend setting **Turbo strength to 0** and **steps to 25** in the Samples tab for Qwen. I'm also training my own turbo, and I'm sure Viggle's will improve over time too.

## The download button fetches everything for Qwen

The Qwen Image 2.1 download fetches every file in its section, including the training adapter and the turbo LoRA. As with the other models, it also fetches the Qwen3-VL captioner the Captions tab uses, plus the small tokenizer files, so first use works offline.
