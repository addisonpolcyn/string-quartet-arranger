"""Ready previews rendered before --release existed for streaming, in place: the same step the
pipeline now runs on every preview (see export.master_for_release). Re-running is harmless.

    python tools/master_for_release.py [song ...]     # default: every song in out/
"""
import json
import sys
from pathlib import Path

from pop2quartet.export import loudness, master_for_release

OUT = Path("out")


def main() -> None:
    songs = sys.argv[1:] or sorted(p.parent.name for p in OUT.glob("*/preview.wav"))
    report = []
    for song in songs:
        src = OUT / song / "preview.wav"
        before = loudness(src)
        after = master_for_release(src, src)
        report.append({"song": song, "before_lufs": before[0], "before_peak": before[1],
                       "lufs": after[0], "true_peak": after[1], "ok": after[1] <= -1.0})
        print(f"{song:24s} {before[0]:6.1f} LUFS {before[1]:5.1f} dBTP -> "
              f"{after[0]:6.1f} LUFS {after[1]:5.1f} dBTP {'ok' if after[1] <= -1.0 else 'TOO HOT'}")
    (OUT / "release_report.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
