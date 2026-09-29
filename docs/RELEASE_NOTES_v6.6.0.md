# Fizgig v6.6.0

Edit LoRA training for Qwen Image 2.1: teach Qwen your own edit, a grade, a look or a relight, from pairs of your original and edited photos, and apply it to any photo afterwards.

## Edit LoRAs for Qwen Image 2.1

Qwen Image 2.1 is one model for both generating and editing images, so a LoRA can learn an edit as well as a subject or a style. You train it on pairs of the same photo: your original, and your edited version. The LoRA learns to make that edit to new photos, changing the colour and tone and leaving the content of the photo alone.

My first real test was one of my own film grades, from a recent photo trip to Sedona. Settings: 40 before/after pairs, the **Qwen 2.1 Fast** preset (rank 8, alpha 8, Adaptive LR 2e-4 to 4e-4, 0.5 MP) with Edit LoRA ticked. By epoch 9 it matched my edit, and it carried the grade over to photos it had never seen. Below: the original, my edit, and Fizgig's epoch 9 training preview.

![Before edit](https://raw.githubusercontent.com/shootthesound/Fizgig/v6.6.0/assets/qwen_edit/1_before_edit.jpg)

![After edit](https://raw.githubusercontent.com/shootthesound/Fizgig/v6.6.0/assets/qwen_edit/2_after_edit.jpg)

![Fizgig's epoch 9 preview](https://raw.githubusercontent.com/shootthesound/Fizgig/v6.6.0/assets/qwen_edit/3_fizgig_epoch9_preview.png)

**What you need:** about 40 pairs (20 at least; more if your photos vary a lot). Each photo 1 MP or larger, e.g. 1200×800; bigger is fine, Fizgig resizes them. An original and its edited version must have the same crop and shape.

**What to do:**

1. On the Training tab, pick Qwen Image 2.1 and load one of the two new Edit presets (see below). Each ticks **Edit LoRA** under Training Parameters.
2. **Originals folder (before editing):** your original, unedited photos. They need no captions.
3. **Edited folder (after editing):** the same photos after your edit, with the same file names as the originals (`IMG_0001.jpg` in both). This is the same folder as on the Start tab.
4. **Captions for the edited photos:** type what the edit is, e.g. "Apply my concert grade.", and press **Write captions**. It saves that text as the caption of every edited photo.
5. **Test photo for previews (optional):** an original photo that's in neither folder. The previews during training show your edit applied to it, so you can see it working on a new photo. Any size: it's fitted to the preview size automatically. Left empty, previews use the first original.

Start checks your pairs before training: it stops and names the files if an edited photo has no original with the same file name, has no caption, or has a different crop or shape from its original.

While Edit LoRA is on, the previews use your edit instruction as their prompt, and the Samples tab says so in place of its prompt box.

**Two Edit presets**, both 12 epochs at 0.5 MP, saving every epoch, with EMA 0.98 and the training adapter:

- **✨ Qwen 2.1 Edit (rank 8, adaptive LR):** the settings from my Sedona grade above, Adaptive LR 2e-4 to 4e-4. For most edits.
- **✨ Qwen 2.1 Edit Strong (rank 16, adaptive LR):** more capacity for trickier edits, Adaptive LR 1e-4 to 2e-4.

Pick your epoch in LoRA Royale; my grade was best at epoch 9.

**Speed and memory:** edit training is slower per step than normal training, because the model reads the original photo as well as the edited one on every step: about 2 s per step on an RTX 5090 at 0.5 MP. A 40-pair, 10-epoch run takes around 15 minutes there, previews included. It trains on 16 GB cards too; Auto picks INT8 with no block swap.

## Licence

Qwen Image 2.1 is released under the Qwen Research License: non-commercial use only unless you get a commercial licence from the Qwen team, and a LoRA or fine-tune you share must say "Built with Qwen" or "Improved using Qwen". Read the [licence](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) before publishing or selling anything made with it.

See the [Qwen Image 2.1 guide](QWEN_IMAGE.md#edit-loras) for the full walkthrough and the command-line flags.
