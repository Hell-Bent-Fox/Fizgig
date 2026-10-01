# Fizgig v6.8.1

Krea 2 moves onto Fizgig's driver system and gains slider LoRAs, including a new Ultra mode for much higher strengths. Qwen Image 2.1 gains full fine-tuning, joining Krea 2 and MiniMax H3. Repair Studio gets bigger sliders with nudge buttons, unlimited strengths, and donor saves that look exactly as previewed.

## Krea 2 on the driver system

Krea 2 now trains through the same standard layer as Qwen Image 2.1, so every tool that works for one works for the other. It's still **Krea 2** in the Base Model list, and your settings and presets carry over. Before switching, it was measured against the previous trainer: the same caches, the same loss, the same previews, and training results within the old trainer's own run-to-run spread.

What changes for you:

- **Your dataset is re-cached once** on the first run. This happens automatically.
- **A run paused in an older Fizgig resumes** from its saved LoRA, epoch and adaptive LR, with a fresh optimizer. Fine-tune checkpoints continue as before.
- **Base precision: fp8 is gone.** Auto picks INT8, then NF4; INT8 is faster. If you had fp8 selected, choose Auto.
- **Training previews use the Turbo LoRA** on the training model. The Turbo checkpoint is no longer a training-preview option; Repair Studio, LoRA the Explorer and LoRA Royale now use it by default.
- **Gradient Accumulation works** on Krea 2 and Qwen Image 2.1. On Qwen it was accepted but ignored before.
- **Command line:** `krea2_train.py` and the two `krea2_cache_*` scripts are replaced by `python -m fizgig.families.cache` and `python -m fizgig.families.train` with `--family krea2`. See [CLI.md](https://github.com/shootthesound/Fizgig/blob/v6.8.0/docs/CLI.md). The command-line-only paired / motion training mode has been removed.

## Slider LoRAs for Krea 2

The slider LoRAs from 6.7.0 now train on Krea 2: a LoRA whose strength is a dial between two looks, from photo pairs or from a few words. Pick **Kind of LoRA: Slider**, or load the new **✨ Krea 2 Slider (rank 4, 2e-4)** preset. Push strength starts at 3 on Krea 2 (2 on Qwen Image 2.1); values from 2 to 9 have been tested on both.

### Ultra mode

A new tick box at the bottom of the Krea 2 slider section. Ultra mode trains the slider on the composition blocks (0 to 7) and the text-fusion blocks only, leaving the fine-detail blocks alone. The result holds up at much higher strengths: one test slider ran cleanly at 20 in ComfyUI. How far yours goes depends on what you train and for how long. The file is also much smaller. Ultra mode works best for sliders trained from prompts; it can work well with photo pairs too, especially when the change is compositional (pose, framing, layout) rather than fine detail.

## Fine-tuning for Qwen Image 2.1

Qwen Image 2.1 can now fine-tune the whole base model, so Fizgig fine-tunes three families: Krea 2, Qwen Image 2.1 and MiniMax H3. On Krea 2 and Qwen, pick **Fine-tune the whole model** under **Kind of training**. MiniMax H3 keeps its own fine-tune checkbox for now. When H3 moves onto the driver system shortly, its fine-tuning gets the same card as Krea 2 and Qwen. Training counts in rotations, and each checkpoint is a complete model file you load in place of the base. An optional folder of class photos (regularisation) trains alongside at a lower learning rate to keep the model's general knowledge intact.

A fine-tune takes longer than a LoRA, but far less than people expect. It trains one part of the model at a time, so it fits a single consumer GPU, and a full rotation through every part takes several epochs: four on a 24 or 32 GB card. With the defaults (10 rotations, 1 epoch per part) that's 40 epochs, at about 0.85 seconds a step on Krea 2 at 0.25 MP against about 0.6 for a LoRA. In total, expect a default fine-tune to take 3 to 4 times as long as the Ultra Fast Krea 2 LoRA preset: its lower learning rate needs more steps to get there. A 16 GB card can fine-tune too, but it splits the model into more parts and streams from system RAM (about 5 seconds a step), so expect a much longer run there.

In practice: a single-character Qwen Image 2.1 fine-tune at the default settings (including the 1e-5 learning rate that choosing Fine-tune sets) can be done in as little as 30 minutes on an RTX 5090. Put several characters in the same dataset folder, each with its own unique name or trigger word, and a multi-character fine-tune takes a couple of hours.

When it's done, use the result as a new checkpoint in place of the base, or press **Checkpoint to LoRA** on the fine-tune card to turn the difference between your fine-tune and the base into a LoRA. Concepts come out far better separated than in a LoRA trained directly. See [FINETUNE.md](https://github.com/shootthesound/Fizgig/blob/v6.8.0/docs/FINETUNE.md).

## Repair Studio

- **Strength boxes take any number.** The primary's and donor's Strength (and LoRA the Explorer's) had a hidden limit of 2. It's gone, so a slider LoRA can be previewed at 20.
- **Saving with a donor looks exactly as previewed.** Each LoRA is baked at its own Strength, and the saved file is used at 1.0. Before, a primary at 20 and a donor at 1 couldn't be reproduced at any single strength. A save without a donor works as before: use it at the primary's strength. The save message tells you which strength to use.
- **Bigger sliders** (about 25% taller, with larger text) and **− / + buttons** that nudge each slider by 0.1; hold one to keep stepping.
- **Krea 2: text-fusion and input/output sliders.** Four text-fusion sliders and one for the input, timestep and output layers sit below the 28 blocks, with the Text fusion ×2 / ×3 presets.
- **Reference picture** for Krea 2 and Qwen Image 2.1 in Repair Studio, LoRA the Explorer and LoRA Royale.
- Donor slider readouts start at +0.00, matching where the sliders sit.

## Also in 6.8.0

- **Klein: Pause / Resume works again.** A paused Klein run went back to idle with no Resume button.
- **Qwen Image 2.1 trains offline** once its files are cached. Reported by [@MatthewsC76](https://github.com/MatthewsC76) (#174).
- **MiniMax H3 on a machine with nearly full RAM** no longer crashes at a preview. Fizgig stages model blocks from normal memory when there isn't room to lock it. Reported by [@sgtsixpack](https://github.com/sgtsixpack) (#175).
- **Each epoch checkpoint's thumbnail shows its own epoch's preview**, not the next one's. Reported by [@DigitalBeer](https://github.com/DigitalBeer) (#122).
- **Samples tab:** Turbo strength is remembered per model family, and the preview length is no longer saved into presets.

## Licence

Qwen Image 2.1 is released under the Qwen Research License: non-commercial use only unless you get a commercial licence from the Qwen team, and a LoRA or fine-tune you share must say "Built with Qwen" or "Improved using Qwen". Read the [licence](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) before publishing or selling anything made with it.

## New in 6.8.1

- **Qwen Image 2.1 previews: Steps and CFG can be edited again.** A "Use Distilled model for samples" tick left over from Klein locked the Samples tab's Steps, CFG and Negative boxes on other models. It now only affects Klein. For Qwen, Turbo strength 0, 20 steps and CFG 3 work very well; a CFG above 1 slows down training samples and Repair Studio, LoRA the Explorer and LoRA Royale previews somewhat.
- **CFG works with Turbo previews.** A CFG above 1 on the Samples tab now applies with the Turbo LoRA on too, along with the negative prompt.
- **Qwen Image 2.1 in Repair Studio, LoRA the Explorer and LoRA Royale** now previews with the Samples tab's steps, CFG, negative prompt and Turbo strength, so one place sets how every Qwen preview looks.
- **A fine-tune no longer picks up a Context LoRA.** On Krea 2 and Qwen, the Context LoRA row hides while Fine-tune is chosen, and a path left in the box is no longer sent to the trainer.
