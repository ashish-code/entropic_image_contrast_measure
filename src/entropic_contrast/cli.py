"""
cli.py — Command-line entry point for the Entropic Contrast scorer.

Registered as ``contrast-score`` via the ``[project.scripts]`` table in
``pyproject.toml``.  After installation (``uv sync`` or ``pip install -e .``),
run::

    # Score a single image
    contrast-score photo.jpg

    # Score all images in a directory, save CSV
    contrast-score --batch ./examples/ --output results/scores.csv

    # Run the visual demo (shows matplotlib plots)
    contrast-score --demo

Examples
--------
    $ contrast-score examples/correct1.jpg
    correct1.jpg: 4.8087

    $ contrast-score --batch examples/ --output results/scores.csv --bins 256
    Scored 9 images → results/scores.csv

    $ contrast-score --batch examples/ --recursive --verbose
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def _run_demo() -> None:
    """Display side-by-side contrast scores for three curated image sets."""
    import matplotlib.pyplot as plt
    from skimage import io

    from entropic_contrast.core import contrast_quality

    # Three sets illustrating different contrast conditions
    example_dir = Path(__file__).parent.parent.parent / "examples"
    image_sets = [
        [
            example_dir / "retinex1.jpg",
            example_dir / "retinex2.jpg",
            example_dir / "retinex3.jpg",
            example_dir / "retinex4.jpg",
        ],
        [
            example_dir / "under1.jpg",
            example_dir / "over1.jpg",
            example_dir / "correct1.jpg",
        ],
        [
            example_dir / "3894541598_bb37af2dcd_o.jpg",
            example_dir / "18056401685_c5b313e712_o.jpg",
        ],
    ]

    for image_set in image_sets:
        existing = [p for p in image_set if p.is_file()]
        if not existing:
            continue

        n = len(existing)
        fig, axes = plt.subplots(1, n, figsize=(n * 5, 5))
        if n == 1:
            axes = [axes]

        for ax, img_path in zip(axes, existing):
            score = contrast_quality(img_path)
            image = io.imread(str(img_path))
            ax.imshow(image)
            ax.set_title(f"{img_path.name}\nContrast Q: {score:.4f}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.tight_layout()
        plt.show()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="contrast-score",
        description=(
            "Entropic Contrast Measure — no-reference image contrast quality scorer.\n\n"
            "Scores each image on a scale from ~0 (very poor contrast) to 5+ "
            "(excellent contrast) using Mutual Information between the image "
            "intensity histogram and its contrast-equalised counterpart."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mutually exclusive primary modes
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "image",
        nargs="?",
        metavar="IMAGE",
        help="Path to a single image file to score.",
    )
    mode.add_argument(
        "--batch",
        metavar="DIR",
        type=Path,
        help="Score all images in this directory.",
    )
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Run the visual demo on the built-in example images.",
    )

    # Options
    parser.add_argument(
        "--output",
        "-o",
        metavar="FILE",
        type=Path,
        default=None,
        help="Output CSV path for --batch mode (default: <dir>/../contrast_quality.csv).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=128,
        metavar="N",
        help="Number of histogram bins (default: 128).",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recurse into subdirectories in --batch mode.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point called by the ``contrast-score`` console script.

    Parameters
    ----------
    argv : list[str] or None
        Argument vector; defaults to ``sys.argv[1:]`` when None.

    Returns
    -------
    int
        Exit code (0 on success, 1 on error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )

    # ------------------------------------------------------------------ Demo
    if args.demo:
        _run_demo()
        return 0

    # ----------------------------------------------------------- Batch mode
    if args.batch:
        from entropic_contrast.core import contrast_quality_batch

        output = contrast_quality_batch(
            args.batch,
            output_path=args.output,
            num_bins=args.bins,
            recursive=args.recursive,
        )
        if output is None:
            print("No images scored.  Check the directory path.", file=sys.stderr)
            return 1
        n = sum(1 for _ in output.open())
        print(f"Scored {n} images → {output}")
        return 0

    # -------------------------------------------------------- Single image
    if args.image:
        from entropic_contrast.core import contrast_quality

        score = contrast_quality(args.image, num_bins=args.bins)
        if score is None:
            print(f"Could not read '{args.image}'", file=sys.stderr)
            return 1
        print(f"{Path(args.image).name}: {score:.4f}")
        return 0

    # ------------------------------------------------- No argument given
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
