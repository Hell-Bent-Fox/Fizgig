# Fizgig v7.0.1: Anima and SDXL arrive, fine-tuning for every model, and the driver system is complete - meaning the community can now add models to Fizgig (Docs included).

Anima and SDXL join with the full workbench. Full fine-tuning now works on all six models, trains the whole model at once when your card has room, and reaches cards down to 8 GB. The driver system is complete: every model now runs on it, and the guide to adding your own is ready.

## Headlines

- **Two new models, Anima and SDXL,** with LoRA and LoKR training, sliders, fine-tuning and every workbench tab.
- **Fine-tune every model.** Klein 9B, Krea 2, Qwen Image 2.1, MiniMax H3, SDXL and Anima all fine-tune from the Training tab: pick Fine-tune as the Kind of training. It's new for Klein, SDXL and Anima. Each checkpoint is a complete model file you load in place of the base.
- **Fine-tune on the card you have.** Every model fine-tunes on a 16 GB card, and Anima and SDXL from 8 GB. On a smaller card the model trains a part at a time, streaming the blocks it isn't training from system memory where it needs to.
- **Bigger cards train more at once, up to the whole model.** Fine-tuning packs as much of the model into each pass as your card holds, which is faster. The whole model trains in one go on SDXL and Anima from 10 GB, Qwen Image 2.1 from 24 GB, Klein on 32 GB and Krea 2 on 48 GB, and on a 32 GB card Krea 2 needs two passes instead of four and MiniMax H3 three. Training everything together got through the whole model about 2.6× faster than four separate passes (measured on Anima).
- **Window size:** a new Training-tab setting caps how many parts train together, for more headroom on a card that's close to its limit. The Training tab shows the plan your card gets before you start.
- **The driver system is complete, guide included.** Klein and MiniMax H3 now run on the driver system that Krea 2 and Qwen Image 2.1 already use, so every model shares training, previews, Repair Studio, LoRA the Explorer, Profiler, Extract and LoRA Royale, and a feature built for one model is available to the others. The guide to adding your own model is below.

Fine-tuning by card size, for photos at 1 MP:

| Model | Fine-tunes from | Whole model at once from | On a 24 GB card | On a 32 GB card |
|---|---|---|---|---|
| Anima | 8 GB | 10 GB | whole model at once | whole model at once |
| SDXL | 8 GB | 10 GB | whole model at once | whole model at once |
| Qwen Image 2.1 | 12 GB | 24 GB | whole model at once | whole model at once |
| Klein 9B | 16 GB | 32 GB | two passes | whole model at once |
| Krea 2 | 16 GB | 48 GB | five passes | two passes |
| MiniMax H3 | 16 GB | 48 GB | five passes | three passes |

**The community can now add models to Fizgig.** The guide to the driver system is in [`docs/drivers`](https://github.com/shootthesound/Fizgig/tree/master/docs/drivers): where your code is called from, walkthroughs for a stills model and a video model, every optional ability, fine-tuning (optional, and can be added after a model first ships), and a checklist of what a finished model includes. SDXL, Anima & Qwen were built by following it.

## New: Anima (experimental)

- **CircleStone Labs' anime and illustration model,** with the full workbench. Note its non-commercial licence.
- **LoRA presets:** Character (rank 16, 1e-4, 50 epochs), Style (rank 16, 5e-5) and Official (rank 32, 2e-5, the model card's recipe).
- **Fine-tuning** trains every block's attention and feed-forward layers, the text adapter left untouched. Two presets: Fine-tune (1e-5) and Fine-tune Official (1e-6). The checkpoint loads in ComfyUI in place of the base.
- **Previews:** the official negative prompt by default. The official Turbo LoRA is available for fast previews.
- **Slider LoRAs,** from prompt pairs or photo pairs, with a Slider preset (rank 4, 2e-4).

## New: SDXL (experimental)

- **Any SDXL checkpoint, one file:** its VAE and text encoders are read from the checkpoint. Juggernaut XL v9 is the default download, with Illustrious XL offered beside it.
- **LoRA presets:** Strong (rank 32, alpha 16, 5e-5) is the default, then Standard (rank 16, alpha 8, 5e-5) and Slider (rank 32, alpha 16, 5e-5). All run with EMA on, a flat learning rate, and per-image learning rates and auto-recaption on (not on Slider).
- **Fine-tuning** trains the UNet's attention and feed-forward layers with the text encoders frozen, and saves a complete checkpoint, text encoders and VAE included, that loads like any other. Two presets: Fine-tune (1e-5) and Fine-tune Official (3e-6).
- **Previews:** DPM++ 2M SDE Karras, 30 steps, CFG 3, with a full default negative prompt. The Samples tab shows the sampler with its ComfyUI names.
- **Fast:** about 0.5–0.75 s a step at 1 MP on a 5090 for LoRAs.
- **Long prompts and captions aren't cut off.** Anything past CLIP's 77 tokens is encoded in chunks, as ComfyUI does.
- **Community SDXL LoRAs load fully in the workbench,** including kohya files, LoCon, and speed LoRAs such as Lightning and LCM.
- **Slider LoRAs,** from prompt pairs or photo pairs.

## Klein and MiniMax H3

- **Klein:** trains on the new engine with your existing model files and presets. The first run on each dataset caches it again, which happens automatically. In-training previews still use Klein Distilled at 4 steps.
- **MiniMax H3:** photos, clips and voice, sliders, RefMod and fine-tuning all work as before, with fewer fine-tune passes on bigger cards. Your saved H3 settings carry over.
- **The new Profiler** now covers Klein and MiniMax H3 too.
- **Repair Studio:** moving a slider now re-renders only the blocks after the change, on every model, with the same picture as a full render. It's on by default and can be switched off with the tick on the Setup card.

## For every model

- **Negative prompt:** a proper multi-line box. Each model has its own default and remembers your edits.
- **Repair Studio:** its own negative prompt box for models that preview with CFG (SDXL, Anima), saved separately from the Samples tab's. A 1024 preview size, which SDXL and Anima start at.
- **Slider practice images** now use the Samples tab's CFG and negative prompt, matching your previews.
- **Problem images excluded in an earlier run train again.** They contribute as normal until they get stuck, and if they do, they're excluded straight away without new recaptions. Exclusions are now kept per model, so a Krea 2 exclusion no longer affects SDXL on the same photos.
- **Checkpoint to LoRA works with every fine-tune,** SDXL and Anima included: it turns a fine-tuned checkpoint into a LoRA at the ranks you pick, which loads in ComfyUI in full, and a Qwen Image 2.1 one now loads in full in Fizgig's workbench too.
- **LoRAs are matched to the right model more reliably** when you load one into a workbench tab.
- **LoHa LoRAs in the Profiler and Extract,** on every model. Extract turns a LoHa into a standard LoRA. Repair Studio, LoRA the Explorer and LoRA Royale already loaded them.
- **More LoRAs from other trainers load:** SDXL LoRAs saved in diffusers naming, as diffusers' own training scripts write them, now work everywhere, alongside kohya, LyCORIS, PEFT and ComfyUI-style files.

LoRAs keep the same format and load in ComfyUI as before.

## Fixes in 7.0.1

- **Fine-tuning an FP32 checkpoint: memory and disk are now sized correctly.** FP32 weights were counted at half their size, so an FP32 SDXL checkpoint could be kept in system memory when it didn't fit, or start writing its on-disk copy to a drive without enough room. Thanks to @mabseyuk.
- **Adding a model now starts with a Discussion.** Proposals for new models and fine-tuning go in the repo's Discussions (New models & fine-tuning category); bugs and everything else stay in Issues. The driver guide explains the steps.

To update, run `update_fizgig.bat` (or `update_fizgig_rocm.bat` on AMD).
