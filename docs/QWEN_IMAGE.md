# Qwen Image 2.1

[← Back to the README](../README.md)

Qwen Image 2.1 trains LoRAs and LoKRs in Fizgig, including edit LoRAs learned from before/after pairs, with turbo previews during training, the per-image loss watch, and all five workbench tools (Repair Studio, LoRA the Explorer, Profiler, Extract, LoRA Royale). Saved LoRAs, LoKRs, Repair Studio saves and Extract outputs load in ComfyUI's standard LoRA loader. It's the first model on Fizgig's driver system.

## Getting set up

1. On the **Preferences** tab, press **Download models for me** in the Qwen Image 2.1 section. It fetches the DiT, VAE, text encoder, the Fizgig training adapter and Viggle's turbo LoRA (about 34 GB), plus Krea 2's Qwen3-VL-4B captioner (about 5 GB) if you don't have it, and the tokenizer files so training works offline.
2. On the **Training** tab, pick **Qwen Image 2.1** as the Base Model. The Fast preset loads on your first visit.

## The Fizgig training adapter

Qwen 2.1 LoRAs tend to collapse into texture or wobble part-way through training, and a lower loss doesn't warn you. The training adapter fixes this: a small LoRA that stays frozen and active while you train, switched off for previews and never saved into your LoRA, so the LoRA works on the plain model.

Fizgig's adapter is trained at a higher resolution than the existing Qwen 2.1 training assistant, and LoRAs trained with it come out much sharper. It's on in every Qwen preset and downloads with the models. It's also on Hugging Face for any trainer: [ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter](https://huggingface.co/ShootTheSound/Fizgig-Qwen-Image-2.1-Training-Adapter).

## Presets

Qwen renders very sharp images. A LoRA pulls fine detail such as skin texture toward your dataset, so a long run on softer photos trades Qwen's sharpness for theirs. Training faster keeps more of it, and in our testing **0.5 MP is the sweet spot**: quicker than 1 MP, and it keeps more sharpness than 0.25 MP.

All three presets train at 0.5 MP with adamw8bit, EMA 0.98 and the training adapter, for 30 epochs with every epoch saved:

| Preset | Rank | Learning rate | For |
|---|---|---|---|
| **Qwen 2.1 Fast** (default) | 8 | Adaptive LR 2e-4 to 4e-4 | Most subjects. The quickest to likeness in our tests, and the best at holding skin detail. |
| **Qwen 2.1 Standard** | 16 | Adaptive LR 1e-4 to 2e-4 | Bigger or mixed datasets. Rank 16 at Fast's rates overtrains. |
| **Qwen 2.1 Style** | 16 | Flat 1.5e-4 | Styles. Adaptive LR climbs on style datasets, which is where styles overbake. |

If a later epoch looks softer than you'd like, an earlier one is often the better pick; scrub them in LoRA Royale.

### 0.5 MP is not 512×512

0.5 MP is half a million pixels, about 704×704 for a square image. Fizgig buckets each image by its shape at that pixel count:

| Aspect | Training size |
|---|---|
| 1:1 | 704×704 |
| 4:5 | 624×784 |
| 2:3 | 576×848 |
| 9:16 | 528×928 |
| 3:2 | 848×576 |
| 16:9 | 928×528 |

512×512 is 0.25 MP, the lowest option in the Target MP box.

## Memory: 10 GB and up

**Base precision: Auto** picks bf16, INT8 or 4-bit NF4 at launch and sizes Blocks Swap to match, from your free VRAM and training resolution. It picks the most precise option that fits and quantises before it swaps, because swapping costs far more speed. At the presets' 0.5 MP:

| Card | Auto picks |
|---|---|
| 24 GB and up | bf16 |
| 12–16 GB | INT8, no block swap |
| 10 GB | 4-bit NF4 |

The text encoder only encodes, and loads in 8-bit below about 20 GB free (about 8 GB in all), which is what sets the 10 GB floor. On an RTX 5090 limited to 12 GB, the Fast preset trained on INT8 with a 9.9 GB peak, previews included; limited to 10 GB, on NF4 with a 7.3 GB peak.

## Previews and the turbo LoRA

With Viggle's turbo LoRA set in Preferences, training previews render at its own settings: **strength 1.0 for 6 steps**. The Samples tab's **Turbo strength** box and step count change either. Without the turbo, previews render at 25 steps. The training adapter is off for previews; a Context LoRA stays on.

## Edit LoRAs

Qwen Image 2.1 is one model for text-to-image and editing, so a LoRA can learn a change: a grade, a style, a relight, a retouch. You give it pairs of the same picture before and after the change.

1. Put the **after** images in the Start tab's folder and the **before** images in a second folder with the same file names. `photo.png` pairs with `photo.png`, one before-image per after-image.
2. Caption each after-image with the instruction, e.g. "Make it a pencil sketch." The same instruction on every pair is fine for a single edit.
3. On the Training tab, tick **Edit LoRA** under Training Parameters and set the **Before-images folder**. Start refuses to run if any after-image is missing its before-image.
4. Previews edit the **Preview photo**, or the first before-image if you leave it empty. A photo from outside the dataset shows whether the edit carries over.

In ComfyUI, load the LoRA as usual and use **Text Encode Qwen Image 2.1**: connect the VAE, plug the photo to edit into its first image input, write the instruction as the prompt, and sample from the node's **latent** output so the result keeps the photo's shape. The node's **resolution** works best near the size the LoRA trained at: in our tests an edit LoRA trained at 0.5 MP matched its Fizgig previews at resolution 768 and came out slightly less exact at the default 1024.

In our tests, 40 pairs of a colour grade at 0.59 MP with rank 16 learned the grade on held-out photos within 6 epochs, about 8 minutes on an RTX 5090. Faces, poses and framing came through unchanged. Turbo previews score the same as 25-step ones, so they are a fair guide to an edit LoRA too.

## Licence

Qwen Image 2.1 is released under the Qwen Research License: non-commercial use only unless you get a commercial licence from the Qwen team, and a LoRA or fine-tune you share must say "Built with Qwen" or "Improved using Qwen". Read the [licence](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) before publishing or selling anything made with it.

## The per-image loss watch

The four Training-tab toggles work on Qwen as on Krea 2 (see [Krea 2](KREA2.md)): problem-image detection, per-image LR, look-outlier warm-up, and auto-recaption. Auto-recaption uses the same Qwen3-VL-4B captioner as the Captions tab. Picking Automagic as the optimizer turns Adaptive LR, per-image LR and the look warm-up off, since Automagic sets its own rate; detection keeps running.

## Command line

Qwen trains headless through `src/fizgig/families/cache.py` and `src/fizgig/families/train.py` with `--family qwen_image21`. See [the CLI guide](CLI.md#qwen-image-21-training) for a full example and the flags.
