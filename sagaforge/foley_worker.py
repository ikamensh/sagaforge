"""Generate clips with Stable Audio 3's MLX runtime; run by :class:`sagaforge.foley.StableAudioMLX` in the runtime's venv.

    <runtime>/.venv/bin/python foley_worker.py jobs.json

``jobs.json``: ``{"runtime": dir, "dit": "medium", "decoder": "same-l", "out": dir, "jobs": [{name, prompt, seconds, seed, steps}]}``.
Writes ``<out>/<name>.wav`` per job and one JSON line per job to stdout.  Only the standard library,
NumPy and the runtime's own modules are used, since this runs under the runtime's Python.
"""

import json
import math
import sys
import time
from itertools import groupby
from pathlib import Path


def main(jobs_path: str) -> None:
    spec = json.loads(Path(jobs_path).read_text())
    runtime = Path(spec["runtime"])
    sys.path.insert(0, str(runtime / "scripts"))
    sys.path.insert(0, str(runtime))
    import mlx.core as mx
    import numpy as np
    import sa3_mlx as S
    from models.defs.sa3_pipeline import apply_prompt_padding, build_pingpong_schedule, load_conditioner_from_npz, patched_decode, sample_flow_pingpong
    from models.defs.t5gemma_mlx import T5Gemma
    from weights import ensure_local

    out = Path(spec["out"])
    dtype = mx.float16
    encoder = T5Gemma.from_npz(str(ensure_local(S.T5GEMMA_NPZ_REL)))
    padding, seconds_embedder = load_conditioner_from_npz(str(ensure_local(S.DIT_CHOICES[spec["dit"]]["ckpt"])), prefix="cond.")
    decoder, chunked, (chunk, overlap) = S.load_decoder(spec["decoder"], mx.float32)

    def latents_for(seconds: float) -> int:
        return max(1, math.ceil(seconds * S.SAMPLE_RATE / S.SAMPLES_PER_LATENT))

    jobs = sorted(spec["jobs"], key=lambda j: (j["seconds"], j["steps"]))
    for (seconds, steps), group in groupby(jobs, key=lambda j: (j["seconds"], j["steps"])):
        t_lat = latents_for(seconds)
        dit, _ = S.load_dit(spec["dit"], T_lat=t_lat, dtype=dtype, num_steps=steps)
        sigmas = build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)
        seconds_embed = seconds_embedder(seconds).astype(dtype)
        for job in group:
            started = time.time()
            embeds, mask = encoder.encode([job["prompt"]], max_len=256)
            cross = mx.concatenate([apply_prompt_padding(embeds.astype(dtype), mask, padding.astype(dtype)), seconds_embed], axis=1)
            global_cond = seconds_embed[:, 0, :]
            noise = mx.random.normal((1, 256, t_lat), dtype=dtype, key=mx.random.key(job["seed"]))
            latents = sample_flow_pingpong(lambda x, t: dit(x, t, cross, global_cond, local_add_cond=None), noise, sigmas, seed=job["seed"] + 1)
            latents = latents.astype(mx.float32)
            if t_lat > chunk + 2 * overlap:
                patches = chunked(decoder, latents, chunk, overlap)
            elif t_lat % 2 == 0:
                patches = decoder(latents)
            else:
                patches = chunked(decoder, latents, 2, 2)
            audio = np.array(patched_decode(patches, patch_size=256, channels=2).astype(mx.float32))[0]
            audio = audio[..., :int(seconds * S.SAMPLE_RATE)]
            peak = float(np.abs(audio).max())
            if peak > 1:  # the decoder overshoots on hard transients; keep the shape, not the clipping
                audio = audio / peak
            S.save_wav(str(out / f"{job['name']}.wav"), audio)
            print(json.dumps({"name": job["name"], "seconds": round(time.time() - started, 2), "peak": round(peak, 3)}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
