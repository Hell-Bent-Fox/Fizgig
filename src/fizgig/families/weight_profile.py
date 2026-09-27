"""Weight-only LoRA profile for any described family (the Profiler tab's standard-layer path).

No model is loaded: for each block of the driver's block map, sum ||up||_F * ||down||_F over its modules, the same
static signature the Krea 2 / H3 profilers report, laid out in the model's own groups and labels. Modules outside the
map (e.g. modulation layers) are reported as one row. Writes the HTML report plus the `<name>.json` sidecar in the
Klein schema, so Repair Studio's profile cross-link works for the family too.
"""
import datetime
import html as _html
import json
import os

OUTSIDE = "outside"


def block_norms(desc, lora_path):
    """-> ({block id or OUTSIDE: norm}, modules counted). Family / kohya / PEFT layouts; LoKR / LoHa are refused."""
    from safetensors import safe_open
    from fizgig.families.lorafile import block_of, lokr_factors, lokr_modules, lora_pairs
    blocks = block_of(desc)
    out, n = {}, 0
    with safe_open(lora_path, "pt") as f:
        for mod, down, up, _alpha in lora_pairs(desc, f.keys()):
            bid = blocks.get(mod, OUTSIDE)
            out[bid] = out.get(bid, 0.0) + float(f.get_tensor(up).float().norm()) * float(f.get_tensor(down).float().norm())
            n += 1
        for mod, stem in lokr_modules(desc, f.keys()):       # ||kron(w1, w2)|| = ||w1|| * ||w2||, exactly
            w1, w2, scale = lokr_factors(f, stem)
            bid = blocks.get(mod, OUTSIDE)
            out[bid] = out.get(bid, 0.0) + float(w1.norm()) * float(w2.norm()) * scale
            n += 1
    if not out:
        raise RuntimeError(f"No LoRA modules found — is this a {desc.display_name} LoRA?")
    return out, n


def profile_weight_only(desc, lora_path, output_html):
    """Write the report + sidecar. Returns (html path, sidecar path)."""
    from fizgig.profiler.visualize import compute_lora_hash
    norms, n_modules = block_norms(desc, lora_path)
    groups = desc.load_driver().block_map()
    labels = {b.id: b.label for g in groups for b in g.blocks}
    labels[OUTSIDE] = "Outside the blocks"
    total = sum(norms.values()) or 1.0
    ranked = sorted(norms.items(), key=lambda kv: kv[1], reverse=True)
    lora_hash = compute_lora_hash(lora_path)

    maxn = max(norms.values()) or 1.0
    sections = []
    for g in groups + ([type(groups[0])(labels[OUTSIDE], [])] if OUTSIDE in norms else []):
        ids = [b.id for b in g.blocks] if g.blocks else [OUTSIDE]
        rows = "".join(
            f'<div class="row"><div class="lbl">{_html.escape(labels[b])}</div><div class="barwrap">'
            f'<div class="bar" style="width:{max(1, round(100 * norms.get(b, 0) / maxn)) if b in norms else 0}%">'
            f'</div></div><div class="val">{100 * norms.get(b, 0) / total:.1f}%</div></div>' for b in ids)
        sections.append(f'<div class="grp">{_html.escape(g.label)}</div>{rows}')
    ranked_rows = "".join(f"<tr><td>{i + 1}</td><td>{_html.escape(labels.get(b, b))}</td>"
                          f"<td>{100 * v / total:.1f}%</td><td>{v:.4f}</td></tr>" for i, (b, v) in enumerate(ranked))
    name = os.path.basename(lora_path)
    title = f"{desc.display_name} weight profile"
    page = f"""<!doctype html><html><head><meta charset="utf-8">
{f'<meta name="fizgig-lora-hash" content="{lora_hash}">' if lora_hash else ''}
<title>{_html.escape(title)} — {_html.escape(name)}</title><style>
body{{font-family:Segoe UI,system-ui,sans-serif;background:#1b1d23;color:#e6e6e6;margin:0;padding:28px;}}
h1{{font-size:20px;margin:0 0 4px;}} .sub{{color:#9aa0aa;font-size:13px;margin-bottom:20px;}}
.card{{background:#23262e;border:1px solid #333;border-radius:10px;padding:18px 20px;margin-bottom:18px;}}
.grp{{font-weight:600;font-size:13px;margin:12px 0 6px;}}
.row{{display:flex;align-items:center;gap:10px;margin:3px 0;}}
.lbl{{width:150px;font-size:12px;color:#cfd3da;}} .val{{width:54px;text-align:right;font-size:12px;color:#9aa0aa;}}
.barwrap{{flex:1;background:#2c3038;border-radius:4px;height:14px;overflow:hidden;}}
.bar{{height:100%;background:linear-gradient(90deg,#4a90d9,#6db3f2);}}
table{{width:100%;border-collapse:collapse;font-size:13px;}} td,th{{padding:5px 8px;text-align:left;border-bottom:1px solid #2c3038;}}
th{{color:#9aa0aa;font-weight:600;}} .note{{color:#9aa0aa;font-size:12px;margin-top:10px;}}
</style></head><body>
<h1>{_html.escape(title)}</h1>
<div class="sub">{_html.escape(name)} &middot; {n_modules} modules &middot; weight-only</div>
<div class="card"><h3 style="margin:0 0 4px">Per-block weight signature</h3>{''.join(sections)}
<div class="note">||up||&middot;||down|| summed per block, as % of the LoRA's total. No block roles are mapped for
{_html.escape(desc.display_name)} yet, so blocks are shown in model order.</div></div>
<div class="card"><h3 style="margin:0 0 12px">Top blocks by weight</h3>
<table><tr><th>#</th><th>Block</th><th>Share</th><th>Norm</th></tr>{ranked_rows}</table></div>
</body></html>"""
    os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(page)
    sidecar = output_html.rsplit(".", 1)[0] + ".json"
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump({
            "version": 1, "hash": lora_hash, "lora_path": lora_path, "lora_name": name,
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "architecture": desc.key, "profile_kind": "weight-only", "html_report": os.path.basename(output_html),
            "top_active_blocks": [{"name": b, "category": "", "pct": round(100 * v / total, 1)} for b, v in ranked[:8]],
            "top_static_blocks": [{"name": b, "norm": round(v, 5)} for b, v in ranked[:8]],
        }, f, indent=2)
    return output_html, sidecar
