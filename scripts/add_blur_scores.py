"""TRIDENTなどが出力したパッチh5に、ぼやけスコアを追記するスクリプト。

`coords` しか持たないh5に対しては元WSIから画素を読み直してスコアを計算し、
同じh5に `blur_scores_val` / `blur_scores_mean` を追加する。

使用例:
    python -m scripts.add_blur_scores <h5_dir> --wsi_dir <wsi_dir> --n_workers 8
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import blur
from src.patch_source import PatchSource, WSIIndex, iter_h5_paths


def process_one(h5_path, wsi_path, patch_size, patch_level, overwrite):
    """1つのh5にスコアを追記する。ワーカープロセスから呼ばれる。

    wsi_path の対応付けは親プロセス側で済ませておく（フォルダ分けされた
    データ置き場を、ワーカーごとに何度も走査し直さないため）。
    """
    h5_path = Path(h5_path)

    # すでにスコアがあるならスキップ（再実行時の無駄な再計算を避ける）
    if not overwrite:
        with h5py.File(h5_path, "r") as f:
            if all(name in f for name in blur.METRICS):
                return h5_path, len(f["coords"]), "skipped"

    with h5py.File(h5_path, "r") as f:
        needs_wsi = "images" not in f
    if needs_wsi and wsi_path is None:
        raise FileNotFoundError(
            f"{h5_path.name}: 画素が無いため元WSIが要りますが、対応するWSIが見つかりません"
        )

    # スコア計算（読み取り専用で開く）
    with PatchSource(h5_path, wsi_path, patch_size, patch_level) as src:
        n = len(src)
        scores = {name: np.empty(n, dtype="float32") for name in blur.METRICS}
        for i in range(n):
            patch = src.read_patch(i)
            for name, value in blur.compute_all(patch).items():
                scores[name][i] = value

    # 追記（別途開き直すことで、計算中の書き込みロックを避ける）
    with h5py.File(h5_path, "a") as f:
        for name, values in scores.items():
            if name in f:
                del f[name]
            f.create_dataset(name, data=values, dtype="float32")

    return h5_path, n, "done"


def main(args):
    h5_root = Path(args.h5_dir)
    # フォルダ分けされていても再帰的に拾う
    h5_paths = iter_h5_paths(h5_root)

    if not h5_paths:
        print(f"No .h5 files found in {h5_root}")
        return 1

    # h5からの相対フォルダを求めるための基準。単一ファイル指定なら親ディレクトリ
    rel_root = h5_root.parent if h5_root.is_file() else h5_root

    # WSIの対応付けは親プロセスでまとめて行う。索引は一度作れば使い回せる
    wsi_by_h5 = dict.fromkeys(h5_paths)
    if args.wsi_dir:
        index = WSIIndex(args.wsi_dir)
        for p in h5_paths:
            wsi_by_h5[p] = index.find(p.stem, p.relative_to(rel_root).parent)
        if index.ambiguous_stems:
            hit = sorted(s for s in index.ambiguous_stems if s in {q.stem.lower() for q in h5_paths})
            if hit:
                print(f"Warning: {args.wsi_dir} に同名のWSIが複数あります({len(hit)}件)。"
                      "h5と同じ相対フォルダにあるものを優先します")
                for stem in hit[:5]:
                    print(f"  - {stem}")

    print(f"Found {len(h5_paths)} h5 files. Processing with {args.n_workers} workers...")

    n_done = n_skipped = n_failed = 0
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(
                process_one, p, wsi_by_h5[p], args.patch_size, args.patch_level, args.overwrite
            ): p
            for p in h5_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Slides"):
            slide = futures[future]
            try:
                _, n, status = future.result()
                if status == "skipped":
                    n_skipped += 1
                else:
                    n_done += 1
                    tqdm.write(f"Done: {slide.name} ({n} patches)")
            except Exception as e:
                n_failed += 1
                tqdm.write(f"Error: {slide.name}: {e}")

    print("-" * 30)
    print(f"done={n_done}  skipped={n_skipped}  failed={n_failed}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Append blur scores to patch h5 files")
    parser.add_argument("h5_dir", type=str, help="パッチh5のディレクトリ（または単一ファイル）")
    parser.add_argument("--wsi_dir", type=str, default=None,
                        help="元WSIのディレクトリ。h5が画素を含まない場合は必須")
    parser.add_argument("--patch_size", type=int, default=None,
                        help="パッチサイズ。既定ではh5の属性から取得")
    parser.add_argument("--patch_level", type=int, default=None,
                        help="読み出しレベル。既定ではh5の属性から取得")
    parser.add_argument("--n_workers", type=int, default=4, help="並列ワーカー数")
    parser.add_argument("--overwrite", action="store_true",
                        help="既にスコアがあるh5も再計算する")
    args = parser.parse_args()

    raise SystemExit(main(args))
