"""Command-line interface for signspell."""

import argparse

from . import __version__
from .app import run
from .config import DEFAULT_THRESHOLD


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="signspell",
        description="Live ASL fingerspelling alphabet recognition.",
    )
    parser.add_argument(
        "-c", "--camera", type=int, default=0,
        help="Camera index (default: 0).",
    )
    parser.add_argument(
        "-m", "--model", default=None,
        help="Path to a custom .h5 model (default: bundled model).",
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=DEFAULT_THRESHOLD,
        help=f"Confidence threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="Disable the mirrored webcam view.",
    )
    parser.add_argument(
        "-V", "--version", action="version",
        version=f"signspell {__version__}",
    )
    args = parser.parse_args(argv)

    run(
        model_path=args.model,
        camera=args.camera,
        threshold=args.threshold,
        mirror=not args.no_mirror,
    )


if __name__ == "__main__":
    main()
