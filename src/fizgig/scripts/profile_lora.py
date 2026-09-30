"""CLI entry point for LoRA profiling.

Usage:
    python src/fizgig/scripts/profile_lora.py \
      --lora path/to/lora.safetensors \
      --dit path/to/dit \
      --vae path/to/vae \
      --text_encoder path/to/qwen3 \
      --output profile.png
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fizgig.klein.inference import KleinInferencePipeline
from fizgig.profiler.profiler import LoRAProfiler
from fizgig.profiler.visualize import plot_profile_heatmap, print_profile_summary

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Profile a LoRA to map per-block per-timestep activity")
    parser.add_argument("--lora", type=str, required=True, help="Path to LoRA safetensors file")
    parser.add_argument("--krea2", action="store_true",
                        help="Krea 2 LoRA: weight-only per-block profile (HTML report + sidecar, no models loaded)")
    parser.add_argument("--dit", type=str, default=None, help="Path to Klein 9B DiT checkpoint (Klein mode only)")
    parser.add_argument("--vae", type=str, default=None, help="Path to VAE/AE checkpoint (Klein mode only)")
    parser.add_argument("--text_encoder", type=str, default=None, help="Path to Qwen3-8B checkpoint (Klein mode only)")
    parser.add_argument("--output", type=str, default="profile.png", help="Output heatmap path (default: profile.png)")
    parser.add_argument("--model_version", type=str, default="klein-base-9b", help="Model version")
    parser.add_argument("--num_samples", type=int, default=8, help="Samples per timestep bin (default: 8)")
    parser.add_argument("--num_bins", type=int, default=5, help="Number of timestep bins (default: 5)")
    parser.add_argument("--width", type=int, default=1024, help="Profile resolution width (default: 1024)")
    parser.add_argument("--height", type=int, default=1024, help="Profile resolution height (default: 1024)")
    parser.add_argument("--prompt", type=str, default=None, help="Conditioning prompt (should include trigger word). Will ask if not provided.")
    parser.add_argument("--fp8_scaled", action="store_true", help="Use FP8 scaled optimization")
    parser.add_argument("--fp8_text_encoder", action="store_true", help="Use FP8 for text encoder")
    parser.add_argument("--blocks_to_swap", type=int, default=12, help="Blocks to swap to CPU (default: 12)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    if args.krea2:
        from fizgig.families.registry import get
        from fizgig.families.weight_profile import profile_weight_only

        out_html = args.output
        if not out_html.lower().endswith(".html"):
            out_html = os.path.splitext(args.lora)[0] + "_krea2_profile.html"
        html, sidecar = profile_weight_only(get("krea2"), args.lora, out_html)
        print(f"\nKrea 2 weight-only profile:")
        print(f"  Report:  {html}")
        print(f"  Sidecar: {sidecar}")
        return

    if not (args.dit and args.vae and args.text_encoder):
        parser.error("--dit, --vae and --text_encoder are required for Klein activation profiling "
                     "(or pass --krea2 for a weight-only Krea 2 profile).")

    pipeline = KleinInferencePipeline()
    pipeline.load_models(
        dit_path=args.dit,
        vae_path=args.vae,
        text_encoder_path=args.text_encoder,
        model_version=args.model_version,
        device="cuda",
        fp8_scaled=args.fp8_scaled,
        fp8_text_encoder=args.fp8_text_encoder,
        blocks_to_swap=args.blocks_to_swap,
    )

    prompt = args.prompt
    if prompt is None:
        prompt = input("Enter a prompt (include the LoRA trigger word): ").strip()
        if not prompt:
            prompt = "a photo of a person"
            print(f"Using default prompt: {prompt}")

    profiler = LoRAProfiler(pipeline)
    result = profiler.profile(
        lora_path=args.lora,
        num_samples=args.num_samples,
        num_bins=args.num_bins,
        width=args.width,
        height=args.height,
        prompt=prompt,
        seed=args.seed,
    )

    print_profile_summary(result)
    plot_profile_heatmap(result, args.output)
    print(f"\nHeatmap saved: {args.output}")

    pipeline.unload_models()


if __name__ == "__main__":
    main()
