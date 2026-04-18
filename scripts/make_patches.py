import os
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

from src.wsi_processer import WSIProcessor

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
    slide_dir = Path(args.slide_dir)
    base_out_dir = Path(args.out_dir)

    # SlurmのCPU数を自動取得、なければ4
    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", 4))

    # 再帰的に .svs を探す（data/findings/*.svs）
    # slide_path.parent.name が 'findings' の名前になる
    slide_paths = list(slide_dir.glob("*/*.svs"))
    
    if not slide_paths:
        print(f"No .svs files found in {slide_dir}")
        return

    print(f"Found {len(slide_paths)} slides. Processing with {n_workers} workers...")

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for slide_path in slide_paths:
            # findings 名を取得 (例: images.svs の親ディレクトリ名)
            finding_name = slide_path.parent.name
            # 出力先を findings ごとに分ける
            slide_out_dir = base_out_dir / finding_name
            
            futures.append(
                executor.submit(process_single_slide, slide_path, slide_out_dir, args.visualize)
            )
        
        # tqdm で進捗を表示しながら結果を回収
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing slides"):
            slide_path = futures[future]
            try:
                result = future.result()
                print(f"Result: {result}")
                tqdm.write(f"Done: {slide_path.name}")
            except Exception as e:
                print(f"Error processing {slide_path.name}: {e}")
            

            result = future.result()
            print(f"Result: {result}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WSI Patch Extraction")
    parser.add_argument("slide_dir", type=str, help="Directory containing the WSI files (e.g., data/)")
    parser.add_argument("out_dir", type=str, help="Directory to save the output H5 files")
    parser.add_argument("--visualize", action="store_true", help="Whether to save visualization images")
    args = parser.parse_args()

    main(args)