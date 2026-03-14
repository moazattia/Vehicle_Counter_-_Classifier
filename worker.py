"""Standalone worker script — launched via subprocess from the API."""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
from pathlib import Path

# add project dir to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pipeline


def main():
    if len(sys.argv) != 6:
        print("Usage: worker.py <input_video> <output_dir> <job_id> <line_position> <confidence>")
        sys.exit(1)

    input_video = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    job_id = sys.argv[3]
    line_position = float(sys.argv[4])
    confidence = float(sys.argv[5])

    output_dir.mkdir(parents=True, exist_ok=True)
    status_file = output_dir / "status.json"

    try:
        pipeline.run_pipeline(
            input_video=input_video,
            output_dir=output_dir,
            job_id=job_id,
            line_position=line_position,
            confidence=confidence,
        )
    except Exception as e:
        pipeline._write_status(status_file, {
            "status": "failed",
            "progress_percent": 0.0,
            "counts": {},
            "error": str(e),
        })
        print(f"FAILED: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
