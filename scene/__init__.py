import re

from collections import defaultdict
from pathlib import Path

from .runner import Config, Runner


def _index_checkpoints(ckpt_dir: Path):
    pattern = re.compile(r"step(\d+)_rank(\d+)\.pt")
    steps = defaultdict(dict)  # step -> {rank: path}

    for p in ckpt_dir.glob("step*_rank*.pt"):
        m = pattern.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        rank = int(m.group(2))
        steps[step][rank] = p

    assert len(steps) > 0, f"No checkpoint files found in {ckpt_dir}"
    return steps


def get_ckpt_files(ckpt_dir: Path, step: int = -1):
    steps = _index_checkpoints(ckpt_dir)
    if step < 0:
        step = max(steps.keys())
    assert step in steps, f"No checkpoints found for step {step} in {ckpt_dir}"

    rank_dict = steps[step]
    ranks = sorted(rank_dict.keys())
    assert len(ranks) > 0, f"No ranks found for step {step} in {ckpt_dir}"

    # Infer world size from global max
    max_world_size = max(len(r) for r in steps.values())

    if len(ranks) < max_world_size:
        print(f"[!] Incomplete checkpoint warning: found {ranks} ranks, expected {max_world_size}")

    ckpt_files = [rank_dict[r] for r in ranks]
    return ckpt_files, step
