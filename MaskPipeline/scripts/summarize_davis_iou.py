from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize DAVIS IoU outputs across part1/part2/part3.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/home/czy/cv_project/MaskPipeline/outputs/IoU/Davis"),
    )
    return parser


def load_rows(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = build_parser().parse_args()
    sources = {
        "part1": args.root / "part1" / "sambaseline_part1_metrics.csv",
        "part2": args.root / "part2" / "sambaseline_sam2_metrics.csv",
        "part3": args.root / "part3" / "part3_common_metrics.csv",
    }
    summary_rows: list[dict] = []
    for part, path in sources.items():
        for row in load_rows(path):
            if row.get("video_id") == "overall":
                continue
            summary_rows.append(
                {
                    "part": part,
                    "video_id": row.get("video_id", ""),
                    "target_text": row.get("target_text", ""),
                    "JM": row.get("JM", ""),
                    "JR": row.get("JR", ""),
                    "mean_iou": row.get("mean_iou", ""),
                    "min_iou": row.get("min_iou", ""),
                    "max_iou": row.get("max_iou", ""),
                    "warnings": row.get("warnings", ""),
                    "pred_video": row.get("pred_video", ""),
                }
            )

    csv_path = args.root / "davis_iou_summary.csv"
    fields = ["part", "video_id", "target_text", "JM", "JR", "mean_iou", "min_iou", "max_iou", "warnings", "pred_video"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    json_path = args.root / "davis_iou_summary.json"
    json_path.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[summarize_davis_iou] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
