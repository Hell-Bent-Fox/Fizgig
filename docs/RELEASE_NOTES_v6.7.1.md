# Fizgig v6.7.1

A maintenance release with one important fix for Qwen Edit and Slider LoRAs. The 6.7.0 notes follow below, since slider training is still new.

## What's new in 6.7.1

- **Qwen Edit and Slider runs always train with their pairs.** If you chose the training folder first and then set Edit's Originals folder (or a slider's -1 end folder), the run could start without that second folder. An Edit LoRA then trained as a plain LoRA while its previews still looked like edits, and a photo-pair slider stopped at its first step. Start now saves the dataset settings just before every Qwen run, so the pairs are always included. If you trained an Edit LoRA that doesn't apply your edit to new photos, train it again on 6.7.1.
- **The training speed on the progress bar no longer counts preview time.** The s/it figure used to include the time spent rendering previews, so runs with previews looked slower than they were. It now shows training speed only, for every model family. The time-remaining estimate leaves previews out too, so a run with previews finishes a little later than it says.

---

# From 6.7.0

Slider LoRA training for Qwen Image 2.1: a LoRA whose strength is a dial between two looks, such as sad to happy or cool to warm, trained from photo pairs or from a few words.

## Slider LoRAs for Qwen Image 2.1

A slider LoRA moves a picture toward one look at strength +1 and toward the opposite look at -1, with everything in between along the way. Set it to 0.5 for a hint of a smile or -1 for a frown. Sliders are trained at +1 and -1, but most will also go further, so try 1.5 or 2 (or -1.5, -2) for a stronger effect; how far a slider goes before the picture breaks down varies from one to the next. It changes the one thing you trained it on and leaves the rest of the picture alone.

On the Training tab, pick Qwen Image 2.1, then **Kind of LoRA: Slider** under Training Parameters, or load the new **✨ Qwen 2.1 Slider (rank 4, 2e-4)** preset, which selects it for you. There are two ways to give it the two ends.

### From photo pairs

Two folders of the same shots, one for each end of the dial: the same person smiling and not smiling, the same scene lit warm and cool. 4 to 10 pairs is enough.

1. **+1 end folder:** the photos at the +1 end, e.g. smiling. This is the same folder as on the Start tab, and the captions go here.
2. **-1 end folder:** the same shots at the -1 end, each with the same file name as its +1 photo. No captions.
3. **Captions:** describe what the two photos of a pair have in common and leave out the difference, e.g. "a portrait photo of a woman", not "a smiling woman". Press **Write captions** and it's saved as the caption of every +1 photo.

Frame each pair the same way. Handheld shots from the same spot are fine; what matters is that the only change every pair has in common is the one you want the dial to learn. Start checks your pairs before training and names any photo without a partner, without a caption, or with a different shape from its partner.

### From prompts

No photos. You describe the picture and what each end adds, and Fizgig renders its own practice pictures to train on.

1. **What the picture is:** the start of the prompt. Each end's words are added after it with a space.
2. **The +1 end adds** and **The -1 end adds:** e.g. "happy" and "sad".
3. **Push strength:** how hard the two ends are pushed apart. 2 is a good start; higher gives a stronger dial but can change more than the one thing you asked for.

There are two ways to write the first line. Describe one person ("a close-up photo of a young woman with short dark hair,") and the dial changes her expression and keeps her the same. Keep it general ("a close-up photo of a person who is") and each practice picture shows a different person, so the dial learns only the change and works on anyone.

### Previews

Each preview is a strip of three pictures on the same seed: strength -1, 0 and +1, so you can watch the dial take shape during training. They use the slider's own prompt (the shared caption, or the "What the picture is" line), not the Samples tab prompts; the Samples tab still sets the seed, size and steps.

### Good to know

- Sliders are always a plain LoRA; if Network Type is set to LoKR, Fizgig switches it for the slider run.
- A Context LoRA works with sliders: the dial is learned on top of it, e.g. a smile dial for one character LoRA. Use the slider with the same context LoRA at the same strength.
- The saved file is an ordinary LoRA. In ComfyUI, load it as usual and set its strength anywhere from -1 to +1, or beyond as above.

Requested by [@jonwong666](https://github.com/jonwong666) and [@eccentricworx](https://github.com/eccentricworx) (#76).

## Also in 6.7.0

- **Edit and slider pairs ignore upper and lower case in file names**, so `IMG_0001.jpg` pairs with `img_0001.jpg`. Before, Start reported the pair as missing. Reported by [@mabseyuk](https://github.com/mabseyuk).
- **MiniMax H3: Clip Target Megapixels.** When your dataset has video clips, a second box under Target Megapixels sets the size clips are cached and trained at, so photos can train at 1.0 MP beside clips at 0.25. A clip costs far more memory and time than a photo of the same size. Suggested by [@CognitiveDiffusion](https://github.com/CognitiveDiffusion) (#170).
- **MiniMax H3: caching clips no longer stalls near the memory limit.** The size check for clips that won't fit in free VRAM allowed clips about 7% too big, so on a busy desktop Windows spilled into shared memory and caching slowed to a crawl. Clips now keep the intended margin. Reported by [@CognitiveDiffusion](https://github.com/CognitiveDiffusion) (#168).

## Licence

Qwen Image 2.1 is released under the Qwen Research License: non-commercial use only unless you get a commercial licence from the Qwen team, and a LoRA or fine-tune you share must say "Built with Qwen" or "Improved using Qwen". Read the [licence](https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/LICENSE) before publishing or selling anything made with it.

See the [Qwen Image 2.1 guide](https://github.com/shootthesound/Fizgig/blob/v6.7.0/docs/QWEN_IMAGE.md#slider-loras) for the full walkthrough and the command-line flags.
