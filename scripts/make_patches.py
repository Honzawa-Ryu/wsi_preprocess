import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.wsi_processer import WSIProcessor
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
    try:
        # 出力先ディレクトリがなければ作成
        slide_out_dir.mkdir(parents=True, exist_ok=True)
        
        processor = WSIProcessor(str(slide_path))
        h5_path = processor.run(str(slide_out_dir))
        
        if visualize:
            vis_path = slide_out_dir / f"{slide_path.stem}_vis.png"
            processor.visualize(save_path=vis_path)
            
        return h5_path
    except Exception as e:
        return f"Error processing {slide_path.name}: {e}"

def main(args):
    start = time.time()
    threading.Thread(target=monitor_resources, daemon=True).start()
    slide_dir = Path(args.slide_dir)
    base_out_dir = Path(args.out_dir)

    # SlurmのCPU数を自動取得、なければ4
    n_workers = args.n_workers if hasattr(args, 'n_workers') else int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    # 再帰的に .svs を探す（data/findings/*.svs）
    # slide_path.parent.name が 'findings' の名前になる
    slide_paths = list(slide_dir.glob("*/*.svs"))
    
    if not slide_paths:
        print(f"No .svs files found in {slide_dir}")
        return

    print(f"Found {len(slide_paths)} slides. Processing with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
            # ここで一気に全スライドのタスクを登録します
            # p.parent.name を使うことで、 findings ごとのディレクトリ分けも同時に処理
            futures = {
                executor.submit(
                    process_single_slide, 
                    p, 
                    base_out_dir / p.parent.name, 
                    args.visualize
                ): p for p in slide_paths
            }
            
            # tqdm で進捗を表示しながら結果を回収（ここは完璧です！）
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slides"):
                slide_path = futures[future]
                try:
                    result = future.result()
                    tqdm.write(f"Done: {slide_path.name} -> {result}")
                except Exception as e:
                    tqdm.write(f"Error processing {slide_path.name}: {e}")
                

                result = future.result()
                print(f"Result: {result}")
    end = time.time()
    print("-" * 30)
    print(f"SUMMARY: n_workers={n_workers}")
    print(f"Total Time: {end - start:.2f}s")
    print(f"Throughput: {len(slide_paths)/(end - start):.4f} slides/s")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Patch Extraction")
    parser.add_argument("slide_dir", type=str, help="Directory containing the WSI files (e.g., data/)")
    parser.add_argument("out_dir", type=str, help="Directory to save the output H5 files")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of parallel workers (default: auto-detect from SLURM or use 4)")
    parser.add_argument("--visualize", action="store_true", help="Whether to save visualization images")
    args = parser.parse_args()

    main(args)