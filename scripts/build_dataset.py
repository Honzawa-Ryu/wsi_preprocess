"""blurスコア付きパッチh5群から、学習用データセットを構築するスクリプト。

閾値を満たすパッチからスライドごとにN枚をランダム抽出し、
WebDataset(tar) または LMDB としてまとめて書き出す。

使用例:
    python -m scripts.build_dataset <h5_dir> <out_dir> \
        --wsi_dir <wsi_dir> --threshold 100 --n_per_slide 200 --format webdataset
"""
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import blur
from src.dataset_builder import (CODECS, LMDBWriter, ShardWriter, ShuffleBuffer,
                                 collect_slide_samples, make_slide_ids)
from src.patch_source import WSIIndex, iter_h5_paths


def main(args):
    h5_root = Path(args.h5_dir)
    # フォルダ分けされていても再帰的に拾う
    h5_paths = iter_h5_paths(h5_root)
    if not h5_paths:
        print(f"No .h5 files found in {h5_root}")
        return 1

    # h5からの相対フォルダを求めるための基準。単一ファイル指定なら親ディレクトリ
    rel_root = h5_root.parent if h5_root.is_file() else h5_root

    # slide_idはtar内のサンプルキーを兼ねるので、別フォルダの同名h5が衝突しないよう
    # ここで一意化する。衝突していないものはstemのまま。
    slide_ids = make_slide_ids(h5_paths, rel_root)
    stems = Counter(p.stem for p in h5_paths)
    duplicated = sorted(s for s, n in stems.items() if n > 1)
    if duplicated:
        print(f"Note: 同名のh5が複数フォルダにあります({len(duplicated)}件)。"
              "slide_idはフォルダ名を含む形に置き換えました")
        for stem in duplicated[:5]:
            renamed = [slide_ids[p] for p in h5_paths if p.stem == stem]
            print(f"  - {stem} -> {', '.join(renamed)}")

    if args.threshold is None and args.threshold_percentile is None:
        print("--threshold か --threshold_percentile のどちらかを指定してください")
        return 1
    if args.threshold is not None and args.threshold_percentile is not None:
        print("--threshold と --threshold_percentile は同時に指定できません")
        return 1

    # WSIの対応付けは親プロセスでまとめて行い、欠けているスライドを先に検出する。
    # 索引は一度だけ作る（h5ごとにツリーを歩き直すと探索だけで時間を食う）
    index = WSIIndex(args.wsi_dir) if args.wsi_dir else None
    jobs = []
    missing = []
    for p in h5_paths:
        # 同名WSIが複数ある場合に備え、h5側の相対フォルダを手がかりに渡す
        wsi_path = index.find(p.stem, p.relative_to(rel_root).parent) if index else None
        if wsi_path is None and index:
            missing.append(slide_ids[p])
        jobs.append((p, wsi_path))

    if missing:
        print(f"Warning: {len(missing)} 件のWSIが見つかりません（h5に画素があれば処理は継続）")
        for slide_id in missing[:5]:
            print(f"  - {slide_id}")

    if index and index.ambiguous_stems & {p.stem.lower() for p in h5_paths}:
        print(f"Note: {args.wsi_dir} に同名のWSIが複数あります。"
              "h5と同じ相対フォルダにあるものを優先して対応付けました")

    out_dir = Path(args.out_dir)
    if args.format == "webdataset":
        writer = ShardWriter(out_dir, prefix=args.shard_prefix,
                             max_count=args.shard_max_count,
                             max_size_mb=args.shard_max_size_mb)
    else:
        writer = LMDBWriter(out_dir, map_size_gb=args.lmdb_map_size_gb)

    sink = ShuffleBuffer(writer, size=args.shuffle_buffer, seed=args.seed)

    print(f"Found {len(h5_paths)} slides. Building with {args.n_workers} workers...")

    per_slide = {}
    n_failed = 0
    try:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(
                    collect_slide_samples, p, wsi, slide_ids[p],
                    args.patch_size, args.patch_level,
                    args.metric, args.threshold, args.threshold_percentile,
                    args.n_per_slide, args.seed, args.codec,
                ): p
                for p, wsi in jobs
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Slides"):
                slide = futures[future]
                try:
                    slide_id, samples, n_eligible = future.result()
                except Exception as e:
                    n_failed += 1
                    tqdm.write(f"Error: {slide.name}: {e}")
                    continue

                for sample in samples:
                    sink.write(sample)
                # slide_idが一意化で変わることがあるので、元のh5も辿れるようにしておく
                per_slide[slide_id] = {
                    "selected": len(samples),
                    "eligible": n_eligible,
                    "path": str(slide.relative_to(rel_root)),
                }

                if args.n_per_slide is not None and 0 < len(samples) < args.n_per_slide:
                    tqdm.write(
                        f"Warning: {slide_id} は閾値を満たすパッチが "
                        f"{n_eligible} 枚しかなく、{len(samples)} 枚のみ採用しました"
                    )
                elif not samples:
                    tqdm.write(f"Warning: {slide_id} は閾値を満たすパッチが0枚でした")
    finally:
        sink.close()

    total = writer.total
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "codec": args.codec,
        "metric": args.metric,
        "threshold": args.threshold,
        "threshold_percentile": args.threshold_percentile,
        "n_per_slide": args.n_per_slide,
        "seed": args.seed,
        "num_samples": total,
        "num_slides": len(per_slide),
        "num_failed_slides": n_failed,
        "shards": [p.name for p in getattr(writer, "shards", [])],
        "per_slide": per_slide,
    }
    # LMDBはout_dir自体がDBディレクトリなので、manifestは隣に置く
    if args.format == "lmdb":
        manifest_path = out_dir.parent / f"{out_dir.name}_manifest.json"
    else:
        manifest_path = out_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("-" * 30)
    print(f"Total samples : {total}")
    print(f"Slides        : {len(per_slide)} (failed: {n_failed})")
    print(f"Output        : {out_dir}")
    print(f"Manifest      : {manifest_path}")
    return 1 if n_failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a training dataset from blur-scored patch h5 files")
    parser.add_argument("h5_dir", type=str, help="パッチh5のディレクトリ（または単一ファイル）")
    parser.add_argument("out_dir", type=str, help="出力先（tarシャードのディレクトリ / LMDBのパス）")
    parser.add_argument("--wsi_dir", type=str, default=None,
                        help="元WSIのディレクトリ。h5が画素を含まない場合は必須")

    parser.add_argument("--metric", type=str, default="blur_scores_val",
                        choices=sorted(blur.METRICS), help="閾値判定に使うスコア")
    parser.add_argument("--threshold", type=float, default=None,
                        help="絶対閾値。これ以上のスコアのパッチを候補にする")
    parser.add_argument("--threshold_percentile", type=float, default=None,
                        help="スライドごとの分位点で閾値を決める(0-100)。スキャナ差がある場合に有効")
    parser.add_argument("--n_per_slide", type=int, default=None,
                        help="スライドあたりの抽出枚数。未指定なら閾値を満たす全パッチ")

    parser.add_argument("--format", type=str, default="webdataset",
                        choices=("webdataset", "lmdb"), help="出力形式")
    parser.add_argument("--codec", type=str, default="png", choices=CODECS,
                        help="画素の符号化方式（いずれも可逆）")
    parser.add_argument("--shard_prefix", type=str, default="patches", help="tarシャード名の接頭辞")
    parser.add_argument("--shard_max_count", type=int, default=10000, help="1シャードあたりの最大サンプル数")
    parser.add_argument("--shard_max_size_mb", type=int, default=1024, help="1シャードあたりの最大サイズ(MB)")
    parser.add_argument("--lmdb_map_size_gb", type=int, default=512, help="LMDBのmap_size(GB)")

    parser.add_argument("--patch_size", type=int, default=None, help="既定ではh5の属性から取得")
    parser.add_argument("--patch_level", type=int, default=None, help="既定ではh5の属性から取得")
    parser.add_argument("--shuffle_buffer", type=int, default=10000,
                        help="書き出し時のシャッフルバッファ長。スライド単位の偏りを崩す")
    parser.add_argument("--seed", type=int, default=42, help="サンプリングの乱数シード")
    parser.add_argument("--n_workers", type=int, default=4, help="並列ワーカー数")
    args = parser.parse_args()

    raise SystemExit(main(args))
