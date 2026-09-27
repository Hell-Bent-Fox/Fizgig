"""Weight-only rank reduction for any described family (the Extract tab's standard-layer path), in the family's own
key format (the old extractor writes kohya keys, which is the wrong format for families that approved another one).

For every LoRA pair the delta dW = (alpha/r) * B @ A has rank <= r, so its SVD is exact from two thin QR
factorisations and an r x r SVD - no dense out x in matrix, no model, fast on CPU. The top `rank` components are
kept and split evenly between the new up and down weights, with alpha = the new rank (scale 1).
"""
import time

import torch


@torch.no_grad()
def reduce_pair(A, B, scale, rank):
    """(A [r, in], B [out, r], scale) -> (down [k, in], up [out, k], kept energy fraction), k = min(rank, r)."""
    A, B = A.float(), B.float() * scale
    qb, rb = torch.linalg.qr(B)                   # B = qb rb
    qa, ra = torch.linalg.qr(A.T)                 # A^T = qa ra  ->  B A = qb (rb ra^T) qa^T
    u, s, vh = torch.linalg.svd(rb @ ra.T)
    k = min(rank, s.numel())
    root = s[:k].sqrt()
    up = (qb @ u[:, :k]) * root
    down = (root[:, None] * vh[:k]) @ qa.T
    energy = float((s[:k] ** 2).sum() / (s ** 2).sum().clamp_min(1e-30))
    return down, up, energy


def extract_weight_only(desc, source, output, rank, dtype=torch.bfloat16, progress=None):
    """Write `output` at `rank`. Returns a summary dict (layers, skipped, mean kept energy, seconds, params)."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from fizgig.families.lorafile import family_keys, lokr_factors, lokr_modules, lora_pairs
    t0 = time.time()
    sd, energies, skipped, params = {}, [], 0, 0
    with safe_open(source, "pt") as f:
        metadata = dict(f.metadata() or {})
        pairs = lora_pairs(desc, f.keys())
        for i, (mod, dk, uk, ak) in enumerate(pairs):
            if progress is not None:
                progress("SVD", i, len(pairs))
            if mod is None:
                skipped += 1
                continue
            A, B = f.get_tensor(dk), f.get_tensor(uk)
            alpha = float(f.get_tensor(ak).item()) if ak else float(A.shape[0])
            down, up, e = reduce_pair(A, B, alpha / A.shape[0], rank)
            kd, ku, ka = family_keys(desc, mod)
            sd[kd] = down.to(dtype).contiguous()
            sd[ku] = up.to(dtype).contiguous()
            sd[ka] = torch.tensor(float(down.shape[0]))
            energies.append(e)
            params += down.numel() + up.numel()
        lokrs = lokr_modules(desc, f.keys())
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for i, (mod, stem) in enumerate(lokrs):      # LoKR: the Kronecker delta is full rank - SVD it densely
            if progress is not None:
                progress("SVD (LoKR)", i, len(lokrs))
            if mod is None:
                skipped += 1
                continue
            w1, w2, scale = lokr_factors(f, stem)
            dW = torch.kron(w1.to(dev), w2.to(dev)) * scale
            U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
            k = min(rank, S.numel())
            root = S[:k].sqrt()
            down, up = (root[:, None] * Vh[:k]).cpu(), (U[:, :k] * root).cpu()
            kd, ku, ka = family_keys(desc, mod)
            sd[kd] = down.to(dtype).contiguous()
            sd[ku] = up.to(dtype).contiguous()
            sd[ka] = torch.tensor(float(k))
            energies.append(float((S[:k] ** 2).sum() / (S ** 2).sum().clamp_min(1e-30)))
            params += down.numel() + up.numel()
            del dW, U, S, Vh
    if not sd:
        raise RuntimeError(f"No {desc.display_name} LoRA modules found in {source}")
    for stale in ("sshs_model_hash", "sshs_legacy_hash", "modelspec.hash_sha256"):
        metadata.pop(stale, None)
    metadata.update({"ss_network_dim": str(rank), "ss_network_alpha": str(float(rank)),
                     "ss_fizgig_extract": f"weight-only SVD to rank {rank}"})
    save_file(sd, output, metadata=metadata)
    return {"layers": len(energies), "skipped": skipped, "energy": sum(energies) / len(energies),
            "seconds": time.time() - t0, "params": params, "output": output}
