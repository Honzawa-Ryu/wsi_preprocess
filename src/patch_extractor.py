"""Step4: 学習用パッチの抽出。

条件(ぼやけスコアの閾値)を満たすパッチをスライドごとに最大N枚選び、
`out_dir/<slide_id>/` 以下に画像ファイルとして書き出す。選ばれたパッチの
一覧は `out_dir/patches.csv` にまとめる。

選択・符号化のロジックは `dataset_builder.py` (WebDataset/LMDB向け)と共通化する。
tarやLMDBにまとめたい場合は scripts/build_dataset.py を使うこと。
"""
import csv
from pathlib import Path

from tqdm import tqdm

from src.dataset_builder import collect_slide_samples, make_slide_ids
from src.patch_source import WSIIndex, iter_h5_paths, slide_stem


def extract_patches(h5_dir, out_dir, *, wsi_dir=None, n_per_slide=1000,
                     metric="blur_scores_val", threshold=None, threshold_percentile=None,
                     patch_size=None, patch_level=None, codec="png", seed=42):
    """h5_dir配下の各スライドから条件を満たすパッチを最大n_per_slide枚選び、
    out_dir/<slide_id>/ 以下に画像として書き出す。

    threshold・threshold_percentileのどちらも指定しなければ、閾値なし(全パッチ
    が候補)として扱う。両方同時の指定はできない。

    1スライドの失敗(対応するWSIが見つからない、metricがまだ付与されていない等)は
    他のスライドの処理を止めない(add_blur_scores.py/build_dataset.pyと同様、
    エラーを記録してスキップする)。

    戻り値: ({slide_id: 書き出した枚数} の辞書, 失敗したスライド数) のタプル。
    """
    h5_root = Path(h5_dir)
    h5_paths = iter_h5_paths(h5_root)
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found in {h5_root}")
    if threshold is not None and threshold_percentile is not None:
        raise ValueError("threshold と threshold_percentile は同時に指定できません")

    # どちらも未指定なら閾値なし(全パッチを候補にする)として扱う
    if threshold is None and threshold_percentile is None:
        threshold = float("-inf")

    rel_root = h5_root.parent if h5_root.is_file() else h5_root
    slide_ids = make_slide_ids(h5_paths, rel_root)
    index = WSIIndex(wsi_dir) if wsi_dir else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    counts = {}
    n_failed = 0
    for h5_path in tqdm(h5_paths, desc="Extracting patches"):
        slide_id = slide_ids[h5_path]
        try:
            wsi_path = (
                index.find(slide_stem(h5_path), h5_path.relative_to(rel_root).parent)
                if index else None
            )

            _, samples, _n_eligible = collect_slide_samples(
                h5_path, wsi_path, slide_id, patch_size, patch_level,
                metric, threshold, threshold_percentile, n_per_slide, seed, codec,
            )

            slide_dir = out_dir / slide_id
            if samples:
                slide_dir.mkdir(parents=True, exist_ok=True)
            for key, ext, payload, meta in samples:
                (slide_dir / f"{key}.{ext}").write_bytes(payload)
                manifest_rows.append(meta)
        except Exception as e:
            n_failed += 1
            tqdm.write(f"Error: {slide_id}: {e}")
            continue

        counts[slide_id] = len(samples)

    if manifest_rows:
        fieldnames = sorted({k for row in manifest_rows for k in row})
        with (out_dir / "patches.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

    return counts, n_failed
