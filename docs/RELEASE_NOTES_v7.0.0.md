# Fizgig v7.0.0: the driver system. Klein and MiniMax H3 move over, SDXL and Anima arrive

Klein 9B and MiniMax H3 now run on Fizgig's new driver system, the same engine as Krea 2 and Qwen Image 2.1. Every family shares one path for training, previews and every workbench tab: Repair Studio, LoRA the Explorer, Profiler, Extract and LoRA Royale. Each model is described once, and everything else comes from that description, so new models, and older ones, are far quicker to add. SDXL and Anima, new in this release, are the first two added this way. A feature built for one family also becomes available to the others.

**Add your own model.** The guide to the driver system is in [`docs/drivers`](https://github.com/shootthesound/Fizgig/tree/master/docs/drivers): where your code is called from, step-by-step walkthroughs for a stills model and a video model, every optional ability a model can switch on, and a checklist of what a finished model includes. SDXL and Anima were built by following it.

## Klein and MiniMax H3

- **Klein:** trains on the new engine with your existing model files and presets. The first run on each dataset caches it again, which happens automatically. In-training previews still use Klein Distilled at 4 steps.
- **MiniMax H3:** photos, clips and voice, sliders, RefMod and fine-tuning all work as before. Your saved H3 settings carry over.
- **Fine-tuning on every model:** pick Fine-tune on the Training tab for Klein, MiniMax H3, Krea 2, Qwen Image 2.1, SDXL or Anima. It's new for Klein, SDXL and Anima. An SDXL fine-tune trains the UNet's attention and feed-forward layers with the text encoders frozen, and saves a complete checkpoint you load like any other.
- **Fine-tuning packs as much of the model into each window as your card holds.** The whole model trains at once when it fits (Anima from 10 GB, Qwen Image 2.1 from 24 GB, Klein on 32 GB), and Krea 2 and MiniMax H3 go from four windows to two on a 32 GB card. The Training tab shows what your card gets, and a new Window size setting gives you more headroom if you want it.
- **Klein fine-tunes on 16 GB cards,** streaming the blocks it isn't training from system memory.
- **The new Profiler** now covers Klein and MiniMax H3 too.
- **Repair Studio:** moving a slider now re-renders only the blocks after the change, on every family, with the same picture as a full render. It's on by default and can be switched off with the tick on the Setup card.

## New: SDXL (experimental)

- **Any SDXL checkpoint, one file:** its VAE and text encoders are read from the checkpoint. Juggernaut XL v9 is the default download, with Illustrious XL offered beside it.
- **Presets:** Strong (rank 32, alpha 16, 5e-5) is the default, then Standard (rank 16, alpha 8, 5e-5) and Slider (rank 32, alpha 16, 5e-5). All run with EMA on, a flat learning rate, and per-image learning rates and auto-recaption on (not on Slider).
- **Previews:** DPM++ 2M SDE Karras, 30 steps, CFG 3, with a full default negative prompt. The Samples tab shows the sampler with its ComfyUI names.
- **Fast:** about 0.5–0.75 s a step at 1 MP on a 5090.
- **Long prompts and captions aren't cut off.** Anything past CLIP's 77 tokens is encoded in chunks, as ComfyUI does.
- **Community SDXL LoRAs load fully in the workbench,** including kohya files, LoCon, and speed LoRAs such as Lightning and LCM.
- **Slider LoRAs,** from prompt pairs or photo pairs.

## New: Anima (experimental)

- **CircleStone Labs' anime and illustration model,** with the full workbench. Note its non-commercial licence.
- **Presets:** Character (rank 16, 1e-4, 50 epochs), Style (rank 16, 5e-5) and Official (rank 32, 2e-5, the model card's recipe).
- **Previews:** the official negative prompt by default. The official Turbo LoRA is available for fast previews.

## For every model

- **Negative prompt:** a proper multi-line box. Each model has its own default and remembers your edits.
- **Repair Studio:** its own negative prompt box for models that preview with CFG (SDXL, Anima), saved separately from the Samples tab's. A 1024 preview size, which SDXL and Anima start at.
- **Slider practice images** now use the Samples tab's CFG and negative prompt, matching your previews.
- **Problem images excluded in an earlier run train again.** They contribute as normal until they get stuck, and if they do, they're excluded straight away without new recaptions. Exclusions are now kept per model, so a Krea 2 exclusion no longer affects SDXL on the same photos.

LoRAs keep the same format and load in ComfyUI as before.

To update, run `update_fizgig.bat` (or `update_fizgig_rocm.bat` on AMD).
