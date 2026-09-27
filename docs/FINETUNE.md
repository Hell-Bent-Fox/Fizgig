# Full fine-tuning (experimental)

[← Back to the README](../README.md)

Fine-tuning trains the **base model itself** — no adapter, no rank bottleneck — on a single consumer GPU, for **Krea 2** and **MiniMax H3**. Tick **⚗ Fine-tune the BASE MODEL instead of training a LoRA** on the Training tab and leave **Window** on **Auto (by VRAM)**.

For step-by-step answers, including five-minute recipes for both families, see **["How do I…?"](FINETUNE_HOWDOI.md)**.

> **Status.** It works and the numbers below are measured, but not every scenario is covered yet; field reports on GitHub shape what gets built next. The technique is model-agnostic, and extending it to other models depends on community support for those models — code, PRs and testing. — Peter

## Epochs and cycles

**An epoch trains one slice of the model.** The trainable window rotates each epoch, so a full cycle — typically **4 epochs** — trains every part of the model once. Rule of thumb: **4 fine-tune epochs ≈ 1 true epoch of the whole model.** This is why epoch defaults are high and why saves land on cycle boundaries: each saved checkpoint is a whole, evenly trained model. Some plans split windows and lengthen the cycle; the console prints your cycle length at launch. **Run at least one full cycle** or some weights never train; the console warns you.

## What card do I need?

The "trains on 8 GB" figures elsewhere apply to **LoRA** training. Fine-tuning tiers itself to your card: the planner measures free VRAM at launch, picks the largest plan that fits, and prints what it chose and why. At the default training resolution:

| Your card | Krea 2 — photos | MiniMax H3 — photos | H3 — voice | H3 — video, confirmed | H3 — video on likeness blocks, expected |
|---|---|---|---|---|---|
| **16 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** | up to **3.8 s** |
| **24 GB** | ✅ | ✅ | ✅ | ✅ up to **2.3 s** | up to **5.2 s** |
| **32 GB** | ✅ | ✅ | ✅ | ✅ up to **3.8 s** | up to **5.2 s** |

- Clip lengths follow Gizmo's grid: **2.3 s is the 56-frame slot**, confirmed by measured runs on every tier. **3.8 s is confirmed on 32 GB**, even with video training the whole model.
- The expected column assumes **Training mode: Default** (the default), which confines clips to the likeness blocks; in our tests that trains video as well as the whole model and makes clips far lighter. Those figures are conservative arithmetic from measured constants, not individually measured.
- Whole-model 5.2 s clips need more than 32 GB (measured).
- With Training mode set to Off, one clip anywhere in the folder trains the **whole** model, so a mixed photos + clips dataset uses the clip column.
- **16 GB is the fine-tune floor.** 12 GB cards train LoRAs only.
- Measured 16 GB peaks: **8.8–12.3 GB for H3**, **8.4–11.0 GB for Krea 2**. The console prints your run's peak every epoch.
- Too little VRAM refuses cleanly at launch instead of running out of memory mid-run.

## How it fits

A naive full fine-tune of Krea 2 (12.9B) needs roughly **78 GB** — bf16 weights, gradients and optimizer state at once. Fizgig closes the gap with:

- **Rotating windows.** Only one slice is trainable at a time, so gradients and optimizer state exist for that slice alone. Over a full cycle every weight trains.
- **A 4-bit NF4 frozen base** (the default). The frozen rest of the model is held at half size; on 16 GB it streams from system RAM.
- **A CPU-resident bf16 master copy** as the source of truth. Training never round-trips through a quantiser, so small updates aren't erased, and checkpoints are written in bf16 from the master.
- **Optimizer-in-backward**, which consumes and frees each gradient as it lands (saves 5.2 GB).
- **Adafactor**, whose factored state is ~10× smaller than AdamW's.

## Model files

Fine-tuning uses the same training bases as LoRA training; nothing new to download.

- **Krea 2** fine-tunes the **RAW bf16 model** (`krea2_raw_bf16.safetensors`, ~26 GB). The fp8 Turbo is the preview model and can't be fine-tuned.
- **MiniMax H3** fine-tunes the **pruned int8 checkpoint** (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, ~21 GB), the same file ComfyUI runs. The ~66 GB bf16 file works for LoRA training only; the trainer refuses it for fine-tuning with a clear message.

A finished fine-tune checkpoint is a valid base for its family:

- **Keep training it** — point the model path at the checkpoint; the console prints the exact continuation settings at every save.
- **Train LoRAs on top of it** — set it as the family's base in **Preferences**. Teach the base your world or cast once, then train quick LoRAs for individual subjects. Deploy those LoRAs with the same fine-tuned base in ComfyUI.

**Pause / Resume** works on a fine-tune. Pause saves a full checkpoint even between regular save epochs; Resume carries over the rotation window, checkpoint numbering and remaining epoch count.

## Why fine-tune

A LoRA constrains every update to a low-rank subspace, so concepts compete for the same few directions. That is why LoRAs tend to drag pose, framing and lighting toward the training set along with the likeness. A full-rank update can change how the model *represents* a concept, so it composes with what the model already knows. In our tests, multi-character and concept teaching landed at a much deeper level than LoRA training, with much better results. [Checkpoint to LoRA](#turn-it-back-into-a-lora) turns the result into a shareable file.

## Krea 2

Measured peaks (RTX 5090):

| Window mode | Peak VRAM | Speed | Fits |
|---|---|---|---|
| **component + 4-bit NF4 (the default)** — full-depth windows, resident | ~16 GB (24 GB budget) / ~21–23 GB (32 GB, more headroom held) | ~1.0 s/it | **24 GB and up** |
| component + **4-bit NF4** + streaming | 8.4–11.0 GB | ~2.8 s/it | **16 GB** |
| component on the **fp8 base** (explicit Base-precision pick) — depth-split + streamed | 15.6–17.6 GB | ~3.0 s/it | 24 GB |

**Base precision.** NF4 is the default and needs no setting. On 24 GB it keeps the full-depth component windows resident: **4 windows per cycle instead of 8**, at roughly **3× the step speed** of the fp8 base (~1.0 s/it vs ~3.0 s/it, same dataset, same 24 GB budget). On 16 GB it is the only base that fits. The trade is that the *frozen* part of the model is held more coarsely while the trainable window learns against it; the saved checkpoint is unaffected. For the more accurate frozen context, pick **fp8** under Base precision if you have the VRAM.

**Component mode** (used by Auto at every budget). Every window spans the model's full depth — attention across all 28 blocks, then each MLP matrix in turn — so a concept is learned by every layer at once. The text-fusion stack stays trainable throughout: rotation would never reach it, and it is where prompt-to-concept binding happens. When the budget is tight the planner **depth-splits** windows (a fat window trains in slices — more windows per cycle, still full speed); below that, frozen out-of-window blocks **stream from system RAM** — slower steps, same component-mode learning.

**Block mode** is an explicit Window-dropdown choice: contiguous depth slices with frozen blocks streamed, slower than component at every budget and **not quality-tested**. The good results have all come from component runs.

## MiniMax H3

The same checkbox under the MiniMax H3 family fine-tunes the 33B model. It uses **component windows only**: each window trains one attention or MLP matrix across all 50 blocks (4 windows per cycle), so every window spans the full depth from the first epoch.

- **Training mode: Default** is recommended on every tier: matched runs came out clearly better on look and prompt adherence than full-model fine-tuning, and it shrinks the windows. The VRAM figures below assume it.
  - **32 GB:** the 4-window cycle at full speed.
  - **24 GB:** the planner depth-splits `mlp.fc1` into two slices — a 5-window cycle, full speed, no offloading. Measured peaks 19.1–21.5 GB.
  - **16 GB:** frozen out-of-window blocks also stream from system RAM (~7 GB staged, a 9-window cycle). Measured peaks 8.8–12.3 GB at **~1.5× the step time**.
- **Full-model fine-tuning** (Training mode Off, or any dataset with video clips) plans itself too. On stills: measured peaks 17.3–18.7 GB on a 24 GB budget and 8.8–11.6 GB streamed on 16 GB. Clip datasets reserve activation memory before windows are sized (clips cost VRAM per frame), which is what the card table reflects. When clips are too long for your card, the trainer says so before training starts, with the fix: cut to the 2.3 s Gizmo slot or lower Target Megapixels.
- **Voice and mixed datasets** train too. Voice trains the same blocks as photos. The per-category **stop epoch** counts across Pause/Resume: pause a mixed run, set the stop to the current epoch, Resume, and it finishes voice-only.
- **Use unique trigger tokens.** An invented token gives the fine-tune somewhere clean to bind; a common word drags its existing meaning along.
- **Run length has no standard number.** It depends on learning rate, dataset size and what you're teaching. The 100-epoch default is a generous scrub range for a typical small dataset; **a large dataset probably needs far fewer epochs** (each epoch is more steps). Save once per cycle and compare checkpoints to find where yours peaks. Total *steps* still run well past LoRA habits, so budget wall-clock and disk accordingly.

Video, voice, Training mode and Gizmo are covered in [MINIMAX_H3.md](MINIMAX_H3.md).

## Saves, previews and numbering

These follow the rotation cycle, not the Samples tab, on both families.

- **Max epochs** and **Save every** snap to cycle boundaries — the Save-every box follows the fine-tune controls live in the GUI and the trainer snaps it again at launch — so every checkpoint has each window trained equally.
- **Previews ride the saves:** one render per saved checkpoint plus the final one, overriding the Samples tab's "every N epochs". Prompts, resolution, seed and the live sample override still come from the Samples tab and status bar. Every sample in the gallery maps to a deployable file. Krea 2 previews render on the training DiT with the Turbo LoRA.
- **Checkpoints are numbered by epoch** (`-000004`, `-000008`, …) and numbering continues across Pause/Resume, so a resumed run never overwrites an earlier save.
- **Adaptive LR is off** under fine-tune: rotation boundaries would read as instability to it. Judge the per-save previews, evaluate checkpoints in ComfyUI, or extract a LoRA and scrub the epochs in LoRA Royale.

An H3 fine-tune is a normal H3 checkpoint: load it in ComfyUI directly, or extract a LoRA from it (the extractor reads the int8 format natively).

## Learning rates

**Fine-tuning wants much lower learning rates than LoRAs.** A LoRA nudges a small adapter on a frozen model; a fine-tune moves the model's own weights, so a normal LoRA rate can wreck a fine-tune.

- **MiniMax H3: 3e-5**, the tested fast-and-reliable rate. **1e-4 destroys an H3 fine-tune** (measured).
- **Krea 2: 1e-5.** 1e-4 trains, but treat it as the top of the experiment range; the best results are found lower. When a run looks almost right but slightly overcooked, lower the rate rather than the epoch count.
- The regularisation **LR ×** multiplier (below) is part of the same tuning space.

## Regularisation images (optional)

Full fine-tuning moves every weight, so a long run on a few subjects drifts the model's whole notion of people; there is no low-rank bound to limit it. Point **Regularisation images** at a folder of ordinary photos of the broader class and they train at a reduced rate as an anchor. Leave the folder empty to train without one.

- **LR ×** (default 0.2): **0.1–0.3** tethers the model's prior while your subject trains. Toward **1.0** the reg set trains like a second subject set — class-balanced training rather than an anchor, which is a different, valid thing.
- If the fine-tune drifts the broader class, raise it a step; if the subject learns too slowly, lower it. Worth tuning per dataset.
- Use **real photos, not model output**: anchoring a fine-tune to its own samples distils its artifacts back in.
- Caption them normally. Anything left unsaid gets attributed to the class word.

## Turn it back into a LoRA

**Checkpoint to LoRA** opens from the **Checkpoint to LoRA…** button on the Training tab's fine-tune card, or runs outside the main window: `run_diff_to_lora.bat` in your Fizgig folder (or `python diff_to_lora_gui.py`; Linux/pods: `./run_diff_to_lora.sh`). Point it at the base model you started from and your fine-tuned checkpoint, and it extracts the difference as an ordinary kohya `.safetensors` at whichever ranks you tick — several at once, since one SVD per layer serves them all. It reads H3's int8 format natively.

- **Rank 64 was perceptually indistinguishable from the full ~26 GB checkpoint** at ~0.5 GB. Quality degrades smoothly at lower ranks.
- In our testing, a LoRA **extracted** from a fine-tune came out better than a LoRA **trained directly** at the same or higher rank on the same dataset. A low-rank file can hold a solution that low-rank training struggles to find; the full-rank phase does the work and extraction is nearly free.

## What it costs

- **VRAM:** on the default NF4 base, **24 GB** runs the full-depth component cycle at full speed and **16 GB** streams frozen blocks from RAM at ~1.5× the step time on H3 and ~2.8× on Krea 2 (see the tables above). The fp8 base costs a 24 GB card depth-split, streamed windows at ~3× the step time, and doesn't fit 16 GB.
- **System RAM** for the bf16 master copy, on top of VRAM: ~24 GB on Krea 2, ~23 GB (likeness) to ~38 GB (full model) on H3. H3's master spills to disk automatically, so full-model H3 fine-tuning runs on a 64 GB box. **Krea 2's does not spill, so Krea 2 fine-tuning realistically wants 48 GB+ of system RAM.** The trainer warns at launch when RAM looks tight.
- **Disk — set the save location before the run.** Every save is a full checkpoint: ~26 GB on Krea 2, ~21 GB on H3. Saving once per 4-epoch cycle over a 40-epoch run is ~260 GB, and a Pause writes an extra full checkpoint. The Training tab's **Output Directory** defaults to your LoRA folder; point it at a roomy drive before you press Start, because afterwards the only fix is moving huge files by hand.
- **NVIDIA only.** Fine-tuning is untested on AMD/ROCm: every measured tier is NVIDIA, and the NF4 default relies on bitsandbytes 4-bit, the least-travelled part of the ROCm stack. Reports welcome either way.
- **Low learning rates and long step counts** — see [Learning rates](#learning-rates).
