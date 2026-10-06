"""CLI: a LoRA's weights profile - how much rank it really uses and where its weights are - with no model loaded.
The rendered profile (each block group switched off, likeness and bleed) is on the Profiler tab.

Usage:
    python src/fizgig/scripts/profile_lora.py --family klein --lora path/to/lora.safetensors [--output report.html]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    from fizgig.families.registry import FAMILIES, get
    parser = argparse.ArgumentParser(description="A LoRA's weights profile (rank used, where its weights are)")
    parser.add_argument("--lora", type=str, required=True, help="Path to the LoRA .safetensors file")
    parser.add_argument("--family", type=str, required=True, choices=sorted(FAMILIES),
                        help="The model family the LoRA was trained for")
    parser.add_argument("--output", type=str, default="",
                        help="Report path (.html; default: beside the LoRA)")
    args = parser.parse_args()

    from fizgig.families.block_profile import weight_stats, write_report
    desc = get(args.family)
    out_html = args.output
    if not out_html.lower().endswith(".html"):
        out_html = os.path.splitext(args.lora)[0] + f"_{desc.lora_name_suffix}_profile.html"
    labels = {b.id: b.label for g in desc.load_driver().block_map() for b in g.blocks}
    stats = weight_stats(desc, args.lora)
    html, sidecar = write_report(desc, args.lora, stats, None, out_html, labels)
    rf = stats["rank_for"]
    print(f"\n{desc.display_name} weights profile: rank {stats['max_rank']} in the file; 95% of the change "
          f"fits in rank {rf[0.95]}, 99% in rank {rf[0.99]}.")
    print(f"  Report:  {html}")
    print(f"  Sidecar: {sidecar}")


if __name__ == "__main__":
    main()
