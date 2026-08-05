import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from wsi_preprocess.processor import WSIProcessor


WSI_EXTENSIONS = {".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".vms", ".vmu"}


def process_single_slide(
    slide_path: Path,
    out_dir: Path,
    patch_size: int = 256,
    tissue_threshold: float = 0.2,
    min_tissue_area_thumb: float = 500.0,
    remove_pen: bool = False,
    min_blur_score_val: float | None = None,
    min_blur_score_mean: float | None = None,
    color_norm: str | None = None,
    visualize: bool = False,
    save_images: bool = True,
) -> Path:
    """単一のWSIスライドに対する前処理とHDF5書き出しを行う関数。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = WSIProcessor(
        slide_path=slide_path,
        patch_size=patch_size,
        tissue_threshold=tissue_threshold,
        min_tissue_area_thumb=min_tissue_area_thumb,
        remove_pen=remove_pen,
        min_blur_score_val=min_blur_score_val,
        min_blur_score_mean=min_blur_score_mean,
        color_normalizer=color_norm,
    )

    h5_path = processor.run(out_dir=out_dir, progress=False, save_images=save_images)

    if visualize:
        vis_path = out_dir / f"{slide_path.stem}_vis.png"
        processor.visualize(save_path=vis_path)

    return h5_path


def get_slide_paths(input_path: Path, recursive: bool = True) -> list[Path]:
    """指定されたパス（ファイルまたはフォルダ）からWSIスライドのリストを取得する。"""
    if input_path.is_file():
        return [input_path]

    if recursive:
        all_files = input_path.rglob("*")
    else:
        all_files = input_path.glob("*")

    slide_paths = [
        f for f in all_files if f.is_file() and f.suffix.lower() in WSI_EXTENSIONS
    ]
    return sorted(slide_paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WSI Preprocess & Patch Extraction CLI Tool"
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to WSI file or directory containing WSI files",
    )
    parser.add_argument(
        "-o",
        "--out-dir",
        type=str,
        required=True,
        help="Output directory to save HDF5 patches and visual logs",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Size of square patch in pixels (default: 256)",
    )
    parser.add_argument(
        "--tissue-thresh",
        type=float,
        default=0.2,
        help="Tissue ratio threshold inside a patch (default: 0.2)",
    )
    parser.add_argument(
        "--min-tissue-area",
        type=float,
        default=500.0,
        help="Minimum connected tissue area threshold on thumbnail (default: 500.0)",
    )
    parser.add_argument(
        "--remove-pen",
        action="store_true",
        help="Remove marker pen artifacts from WSI tissue mask",
    )
    parser.add_argument(
        "--min-blur-val",
        type=float,
        default=None,
        help="Minimum Laplacian variance threshold to skip blurry patches",
    )
    parser.add_argument(
        "--min-blur-mean",
        type=float,
        default=None,
        help="Minimum mean edge response threshold to skip blurry patches",
    )
    parser.add_argument(
        "--color-norm",
        type=str,
        default=None,
        choices=["none", "reinhard", "macenko", "adjust"],
        help="Color/stain normalization method applied before saving patch (default: none)",
    )
    parser.add_argument(
        "-n",
        "--n-workers",
        type=int,
        default=None,
        help="Number of parallel worker processes (default: SLURM_CPUS_PER_TASK or os.cpu_count())",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save thumbnail visualization images with patch circles plotted",
    )
    parser.add_argument(
        "--coords-only",
        action="store_true",
        help="Save only patch coordinates and metadata (MahmoodLab CLAM/TRIDENT compatible format) without storing RGB images",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search recursively when input is a directory",
    )
    return parser


def main(args: argparse.Namespace | None = None):
    if args is None:
        parser = build_parser()
        args = parser.parse_args()

    input_path = Path(args.input)
    base_out_dir = Path(args.out_dir)

    slide_paths = get_slide_paths(input_path, recursive=not args.no_recursive)

    if not slide_paths:
        print(f"No valid WSI files found in {input_path}")
        return

    n_workers = (
        args.n_workers
        if args.n_workers
        else int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
    )

    print(
        f"Found {len(slide_paths)} slides. Processing with {n_workers} workers..."
    )

    start_time = time.time()

    if len(slide_paths) == 1 or n_workers == 1:
        for slide_path in tqdm(slide_paths, desc="Processing slides"):
            try:
                out_dir = (
                    base_out_dir
                    if input_path.is_file()
                    else base_out_dir / slide_path.parent.name
                )
                h5_path = process_single_slide(
                    slide_path=slide_path,
                    out_dir=out_dir,
                    patch_size=args.patch_size,
                    tissue_threshold=args.tissue_thresh,
                    min_tissue_area_thumb=args.min_tissue_area,
                    remove_pen=args.remove_pen,
                    min_blur_score_val=args.min_blur_val,
                    min_blur_score_mean=args.min_blur_mean,
                    color_norm=args.color_norm,
                    visualize=args.visualize,
                    save_images=not args.coords_only,
                )
                tqdm.write(f"Done: {slide_path.name} -> {h5_path}")
            except Exception as e:
                tqdm.write(f"Error processing {slide_path.name}: {e}")
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}
            for p in slide_paths:
                out_dir = (
                    base_out_dir
                    if input_path.is_file()
                    else base_out_dir / p.parent.name
                )
                future = executor.submit(
                    process_single_slide,
                    p,
                    out_dir,
                    args.patch_size,
                    args.tissue_thresh,
                    args.min_tissue_area,
                    args.remove_pen,
                    args.min_blur_val,
                    args.min_blur_mean,
                    args.color_norm,
                    args.visualize,
                    not args.coords_only,
                )
                futures[future] = p

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing slides"
            ):
                slide_path = futures[future]
                try:
                    result = future.result()
                    tqdm.write(f"Done: {slide_path.name} -> {result}")
                except Exception as e:
                    tqdm.write(f"Error processing {slide_path.name}: {e}")

    elapsed = time.time() - start_time
    print("-" * 40)
    print(
        f"SUMMARY: Processed {len(slide_paths)} slides in {elapsed:.2f}s "
        f"({len(slide_paths) / max(elapsed, 1e-3):.4f} slides/s)"
    )
    print("-" * 40)


if __name__ == "__main__":
    main()
