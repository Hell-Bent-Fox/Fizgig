# Fizgig v6.8.3

A new Profiler for Krea 2 and Qwen Image 2.1 that shows what each part of a LoRA actually does to the picture. Built on what it found, Qwen Image 2.1 gains a measured identity block map, identity presets in Repair Studio, and Fast Identity Mode: character LoRAs trained about 1.5x faster with very close to full-model likeness. Qwen Image 2.1 training also gains torch.compile, about 1.4x faster on the INT8 base.

## The new Profiler (Krea 2 and Qwen Image 2.1)

The Profiler now renders the LoRA instead of only reading its weights. It switches groups of blocks on and off, on your prompt and seed, and measures what changes: how much of the picture, how much of the likeness, and how much bleed (your subject turning up in prompts that don't use the trigger). Pick what to measure on the Profiler tab:

- **Weights only:** instant. How much rank the LoRA really uses, so you know how small Extract can make it.
- **Quick:** each group of blocks on its own and left out, on one seed. A few minutes.
- **Thorough:** two seeds, plus each block left out one at a time.

Add three photos of your subject to score likeness and bleed. They fill in automatically from a run's likeness baselines. The report puts every result in a plain sentence ("Gives 43% of the likeness", "Bleed drops by 24%") and ends with what to do about it. **Open in Repair Studio** loads the LoRA with those suggestions already on the sliders, side by side with the original, ready to save as a new file.

Klein and MiniMax H3 keep the existing Profiler for now; the new one comes to them when they move onto the driver system.

## Qwen Image 2.1: where the identity lives

Profiled across several character LoRAs, Qwen Image 2.1 keeps a person's identity in a small, consistent set of blocks: **blocks 10 to 14**. On their own they give most of a character LoRA's likeness, and switching them off removes almost all of it. The other blocks shape the picture: style, light and composition.

- **Repair Studio** colours Qwen's sliders by what they carry: **ID** (green) for blocks 10 to 14, **Look** (blue) for the rest. The Profiler report shows the same map.
- **New Repair Studio presets for Qwen:**
  - **✨Identity only** keeps the person and drops the LoRA's style.
  - **✨Look only (no identity)** keeps the style and drops the face, useful for a style LoRA that picked up a person.
  - **✨Identity ×0.5** softens a face that pushes too hard.

## Fast Identity Mode (Qwen Image 2.1)

A new **Fast Identity Mode** tick box under Kind of training (Standard LoRA), and a new **✨ Qwen 2.1 Fast Identity Mode (rank 8)** preset. It trains only the identity blocks. Everything before them skips the backward pass, so each step is about 1.5x faster than training every block.

In our tests the likeness came very close to a LoRA trained on every block. The base model's composition and styling also stay more intact, because only the identity blocks learn. The preset trains at 0.25 MP, the resolution it was tested at.

## torch.compile for Qwen Image 2.1

Qwen Image 2.1 now has **Compile Blocks** on the Training tab, as Krea 2 does. On **Auto** (the default), training compiles the model's blocks when the run is long enough to repay the compile time: about 300 steps on the INT8 base and 800 on bf16. Measured on a 5090: about **1.4x faster per step on INT8** and about 1.15x on bf16, with no extra memory. The first epoch runs slower while the blocks compile; the console says so, and full speed arrives from epoch 3. NF4 isn't compiled by Auto, and block swap turns compile off.

Krea 2 and Qwen Image 2.1 now share the same compile code, and Krea 2 behaves exactly as before. MiniMax H3 and Klein will get it when they move onto the driver system.

## Fixes

- **Extending a finished run with Resume no longer overwrites its last epoch.** The final save is now also kept under its epoch number, as every earlier epoch is (#176). This applies to all families.
- **Krea 2 and Qwen Image 2.1: a preview that fails no longer ends the run.** If a preview runs out of memory, previews switch off for the rest of the run and training carries on, with the model put back exactly as it was, however early in the preview it failed. Thanks to @mabseyuk.
- **Krea 2 and Qwen Image 2.1 fine-tuning: a new run never reuses a crashed run's partly trained weights.** Any leftover on-disk master from a run that stopped early is cleared at the start. Thanks to @mabseyuk.
- **Krea 2 and Qwen Image 2.1: the warm-up note is back.** During the first two epochs the console again says the slow start is normal and that full speed arrives from epoch 3. It went missing in 6.8.0.

To update, run `update_fizgig.bat` (or `update_fizgig_rocm.bat` on AMD).
