import os
import sys
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# このファイルを直接スクリプトとして実行した場合でも src パッケージを解決できるようにする
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.wsi_processer import WSIProcessor
from src.patch_source import WSI_EXTENSIONS, iter_wsi_paths
import psutil
import threading
import time

def monitor_resources(interval=10):
    """CPUとメモリの使用率をバックグラウンドで記録し続ける"""
    while True:
        cpu_usage = psutil.cpu_percent(interval=None)
        memory_info = psutil.virtual_memory()
        # ログに記録（tqdm.writeを使ってプログレスバーを壊さないように出力）
        print(f"[Resource Log] CPU: {cpu_usage}% | RAM: {memory_info.percent}% ({memory_info.used / (1024**3):.1f}GB used)")
        time.sleep(interval)


def process_single_slide(slide_path, slide_out_dir, visualize=False):
    """
    1枚のスライドを処理し、指定された findings 用のディレクトリに保存する
    """
    # 例外はあえて捕まえず呼び出し元に伝える。ProcessPoolExecutorがfutureに
    # 保持してくれるので、1枚失敗しても他のスライドの処理は続く。
    slide_out_dir.mkdir(parents=True, exist_ok=True)

    processor = WSIProcessor(str(slide_path))
    h5_path = processor.run(str(slide_out_dir))

    if visualize:
        vis_path = slide_out_dir / f"{slide_path.stem}_vis.png"
        processor.visualize(save_path=vis_path)

    return h5_path

def main(args):
    start = time.time()
    threading.Thread(target=monitor_resources, daemon=True).start()
    slide_dir = Path(args.slide_dir)
    base_out_dir = Path(args.out_dir)

    # SlurmのCPU数を自動取得、なければ4
    # argparseは未指定でも属性自体は生やすため、hasattrではなくNone判定で分岐する
    if args.n_workers is not None:
        n_workers = args.n_workers
    else:
        n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    # 対応拡張子のWSIを再帰的に探す。data/直下でも data/A/B/ のような
    # 多段のフォルダ分けでも、同じように拾える。
    slide_paths = iter_wsi_paths(slide_dir)

    if not slide_paths:
        exts = ", ".join(WSI_EXTENSIONS)
        print(f"No WSI files ({exts}) found under {slide_dir}")
        return 1

    print(f"Found {len(slide_paths)} slides. Processing with {n_workers} workers...")

    n_failed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # ここで一気に全スライドのタスクを登録します
            # 入力からの相対パスをそのまま出力側に再現し、フォルダ分けを保つ
            futures = {
                executor.submit(
                    process_single_slide,
                    p,
                    base_out_dir / p.relative_to(slide_dir).parent,
                    args.visualize
                ): p for p in slide_paths
            }

            # tqdm で進捗を表示しながら結果を回収
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slides"):
                slide_path = futures[future]
                try:
                    result = future.result()
                    tqdm.write(f"Done: {slide_path.name} -> {result}")
                except Exception as e:
                    n_failed += 1
                    tqdm.write(f"Error processing {slide_path.name}: {e}")
    end = time.time()
    print("-" * 30)
    print(f"SUMMARY: n_workers={n_workers}")
    print(f"Slides: {len(slide_paths)} (failed: {n_failed})")
    print(f"Total Time: {end - start:.2f}s")
    print(f"Throughput: {len(slide_paths)/(end - start):.4f} slides/s")
    print("-" * 30)
    # 失敗があればSlurm側にも異常として伝わるよう終了コードを立てる
    return 1 if n_failed else 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Patch Extraction")
    parser.add_argument("slide_dir", type=str, help="Directory containing the WSI files (e.g., data/)")
    parser.add_argument("out_dir", type=str, help="Directory to save the output H5 files")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of parallel workers (default: auto-detect from SLURM or use 4)")
    parser.add_argument("--visualize", action="store_true", help="Whether to save visualization images")
    args = parser.parse_args()

    raise SystemExit(main(args))