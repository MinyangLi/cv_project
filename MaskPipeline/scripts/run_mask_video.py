from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cli import build_parser, resolve_cli_config
from src.pipeline.video_mask_pipeline import run_video_mask_pipeline


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = resolve_cli_config(args)
    result = run_video_mask_pipeline(args.input, args.output, config)
    logging.info("Masked video: %s", result.output_paths.output_mask_video)
    logging.info("Overlay video: %s", result.output_paths.output_overlay_video)
    logging.info("Masks dir: %s", result.output_paths.masks_dir)
    logging.info("meta.json: %s", result.output_paths.output_root / "meta.json")


if __name__ == "__main__":
    main()
