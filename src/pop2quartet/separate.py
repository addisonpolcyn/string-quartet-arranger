"""Stage 1: split the mix into stems with Demucs."""
import subprocess
import sys
from pathlib import Path

STEMS = ("vocals", "bass", "other", "drums")


def separate(audio: Path, workdir: Path, model: str = "htdemucs") -> dict[str, Path]:
    out = workdir / model / audio.stem
    stems = {s: out / f"{s}.wav" for s in STEMS}
    if all(p.exists() for p in stems.values()):
        print(f"[separate] using cached stems in {out}")
        return stems
    from .export import one_at_a_time

    print(f"[separate] running Demucs ({model}) ...")
    with one_at_a_time("gpu"):
        subprocess.run(
            [sys.executable, "-m", "demucs", "-n", model, "-o", str(workdir), str(audio)],
            check=True,
        )
    return stems
