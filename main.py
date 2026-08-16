"""WSI前処理パイプラインの統括モジュール(PLAN.md 参照)。

TRIDENTをコアエンジンとして使い、独自処理(ぼやけスコア付与・学習用パッチ抽出)を
その前後に挟み込む、ラッパー型のパイプライン。

    Step1: TRIDENTでセグメンテーション + パッチ座標抽出        (常に実行)
    Step2: ぼやけスコアの計算・付与                              (--calc_blur)
    Step3: TRIDENTで特徴量抽出                                   (--patch_encoder を指定した場合のみ)
    Step4: 学習用パッチの抽出                                    (--extract_patches N を指定した場合のみ)

Step2〜4を既定で無効にしているのは、無条件にパッチ画像やモデル依存の特徴量を
書き出すとディスク容量・実行時間を圧迫しやすいため(PLAN.md 6章)。

CLIとPython APIの両方から実行できる。

    python main.py --wsi_dir ./data/wsis --out_dir ./data/processed \\
        --calc_blur --extract_patches 1000

    from main import run_pipeline
    run_pipeline(
        wsi_dir="./data/wsis",
        out_dir="./data/processed",
        calc_blur=True,
        extract_patches=1000,
    )
"""
import argparse
import sys
from pathlib import Path

# このファイルを直接スクリプトとして実行した場合でも src/scripts パッケージを
# 解決できるようにする
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _auto_device():
    """GPUがあれば'cuda:0'、無ければ'cpu'を返す。"""
    import torch
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def run_pipeline(
    wsi_dir,
    out_dir,
    *,
    calc_blur=False,
    extract_patches=None,
    # Step1 (TRIDENT: segmentation + coords)
    mag=20.0,
    patch_size=256,
    overlap=0,
    segmenter=None,
    seg_conf_thresh=0.5,
    remove_holes=False,
    min_tissue_proportion=0.0,
    device=None,
    max_workers=None,
    skip_errors=True,
    wsi_ext=None,
    search_nested=True,
    # Step2 (blur scoring)
    blur_n_workers=4,
    blur_overwrite=False,
    # Step3 (TRIDENT: feature extraction)
    patch_encoder=None,
    patch_encoder_ckpt_path=None,
    stain_normalize_target=None,
    # Step4 (patch extraction)
    patch_metric="blur_scores_val",
    patch_threshold=None,
    patch_threshold_percentile=None,
    patch_codec="png",
    seed=42,
):
    """PLAN.mdの4ステップを順に実行する。

    Step1(TRIDENTによるセグメンテーション+座標抽出)は常に実行する。Step2〜4は
    明示的に有効化したときだけ実行する(calc_blurの既定はFalse、
    extract_patches/patch_encoderの既定はNone=スキップ)。

    戻り値: 各ステップの出力先と結果をまとめたdict
        (coords_dir, features_dir, patches_dir, patch_counts,
         blur_failed, patch_extract_failed)。
        blur_failed/patch_extract_failedはスライド単位の失敗があったかどうかで、
        CLI(main関数)はこれを見て終了コードを決める。
    """
    from src.trident_runner import TridentRunner

    # Step4はh5に既にmetric(既定はblur系)が入っている前提で、main.py経由でこれを
    # 満たせるのはcalc_blur=Trueの場合だけ(Step1のTRIDENT自体はblurスコアを書かない)。
    # ここで弾かないと、Step1(数時間かかりうる)が終わった後のStep4でKeyErrorになる。
    if extract_patches and not calc_blur:
        raise ValueError(
            "--extract_patches は既定で --calc_blur が付与するスコア"
            f"({patch_metric!r}) をパッチ選択に使うため、--calc_blur と併用してください。"
            "既に付与済みのh5を使う場合は src.patch_extractor.extract_patches() を"
            "直接呼び出してください。"
        )

    wsi_dir = Path(wsi_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = _auto_device()
    if segmenter is None:
        # GPUが無い環境ではhest(モデル推論)は動かせないため、CPUのみのotsuに倒す
        segmenter = "hest" if device.startswith("cuda") else "otsu"

    print(f"[Step1] TRIDENT: segmentation + coords (segmenter={segmenter}, device={device})")
    runner = TridentRunner(
        wsi_dir, out_dir,
        mag=mag, patch_size=patch_size, overlap=overlap,
        segmenter=segmenter, seg_conf_thresh=seg_conf_thresh,
        remove_holes=remove_holes, min_tissue_proportion=min_tissue_proportion,
        device=device, max_workers=max_workers, skip_errors=skip_errors,
        wsi_ext=wsi_ext, search_nested=search_nested,
    )
    runner.segment()
    coords_h5_dir = runner.extract_coords()

    blur_failed = False
    if calc_blur:
        print("[Step2] blur scoring")
        from scripts.add_blur_scores import main as add_blur_main
        blur_args = argparse.Namespace(
            h5_dir=str(coords_h5_dir), wsi_dir=str(wsi_dir),
            patch_size=None, patch_level=None,
            n_workers=blur_n_workers, overwrite=blur_overwrite,
        )
        # add_blur_scores.main()は1スライドの失敗では止まらず、最後に0/1を返す
        # (0以外 = 1枚以上失敗)。ここで握りつぶすと、後段(Step4)がそのまま
        # 「正常終了」してしまい、スコアが欠けたことに誰も気づけなくなる。
        blur_failed = bool(add_blur_main(blur_args))
        if blur_failed:
            print("[Step2] Warning: 一部のスライドでぼやけスコア付与に失敗しました(ログ参照)")

    features_dir = None
    if patch_encoder:
        print(f"[Step3] TRIDENT: feature extraction (encoder={patch_encoder})")
        features_dir = runner.extract_features(
            patch_encoder=patch_encoder,
            patch_encoder_ckpt_path=patch_encoder_ckpt_path,
            stain_normalize_target=stain_normalize_target,
        )

    patches_dir = None
    patch_counts = None
    patch_extract_failed = 0
    if extract_patches:
        print(f"[Step4] patch extraction (n_per_slide={extract_patches})")
        from src.patch_extractor import extract_patches as extract_patches_fn
        patches_dir = out_dir / "patches"
        patch_counts, patch_extract_failed = extract_patches_fn(
            coords_h5_dir, patches_dir, wsi_dir=wsi_dir,
            n_per_slide=extract_patches, metric=patch_metric,
            threshold=patch_threshold, threshold_percentile=patch_threshold_percentile,
            codec=patch_codec, seed=seed,
        )
        if patch_extract_failed:
            print(f"[Step4] Warning: {patch_extract_failed}件のスライドでパッチ抽出に失敗しました(ログ参照)")

    return {
        "coords_dir": coords_h5_dir,
        "features_dir": features_dir,
        "patches_dir": patches_dir,
        "patch_counts": patch_counts,
        "blur_failed": blur_failed,
        "patch_extract_failed": patch_extract_failed,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="WSI前処理パイプライン (TRIDENTラッパー)。詳細はPLAN.mdを参照。"
    )
    parser.add_argument("--wsi_dir", required=True, help="WSIが入ったディレクトリ")
    parser.add_argument("--out_dir", required=True,
                        help="出力先ディレクトリ (TRIDENTのjob_dirを兼ねる)")

    parser.add_argument("--calc_blur", action="store_true",
                        help="ぼやけスコアを計算し、座標h5に付与する")
    parser.add_argument("--extract_patches", type=int, default=None, metavar="N",
                        help="学習用に各スライドから抽出するパッチ枚数。未指定なら抽出しない")

    # Step1 (TRIDENT: segmentation + coords)
    seg = parser.add_argument_group("Step1: TRIDENT segmentation + coords")
    seg.add_argument("--mag", type=float, default=20.0, help="パッチ抽出の倍率")
    seg.add_argument("--patch_size", type=int, default=256, help="パッチサイズ(px)")
    seg.add_argument("--overlap", type=int, default=0, help="パッチ間のオーバーラップ(px)")
    seg.add_argument("--segmenter", type=str, default=None, choices=["hest", "grandqc", "otsu"],
                     help="組織セグメンタ。未指定ならGPUの有無で自動選択(hest/otsu)")
    seg.add_argument("--seg_conf_thresh", type=float, default=0.5)
    seg.add_argument("--remove_holes", action="store_true")
    seg.add_argument("--min_tissue_proportion", type=float, default=0.0)
    seg.add_argument("--device", type=str, default=None,
                     help="'cuda:0' や 'cpu'。未指定ならGPUの有無で自動選択")
    seg.add_argument("--max_workers", type=int, default=None)
    seg.add_argument("--wsi_ext", type=str, nargs="+", default=None)
    seg.add_argument("--no_search_nested", dest="search_nested", action="store_false",
                     help="サブディレクトリを再帰的に探索しない (既定: 探索する)")

    # Step2 (blur scoring)
    blur = parser.add_argument_group("Step2: blur scoring")
    blur.add_argument("--blur_n_workers", type=int, default=4)
    blur.add_argument("--blur_overwrite", action="store_true",
                      help="既にスコアがあるh5も再計算する")

    # Step3 (TRIDENT: feature extraction, optional)
    feat = parser.add_argument_group("Step3: TRIDENT feature extraction (optional)")
    feat.add_argument("--patch_encoder", type=str, default=None,
                      help="指定すると特徴量抽出(Step3)を実行する (例: uni_v1, conch_v15)")
    feat.add_argument("--patch_encoder_ckpt_path", type=str, default=None)
    feat.add_argument("--stain_normalize_target", type=str, default=None,
                      help="指定すると、この画像を目標染色としてMacenko染色正規化"
                           "(torchstain)をencoderのeval_transformsの先頭に差し込む")

    # Step4 (patch extraction, optional)
    ext = parser.add_argument_group("Step4: patch extraction (optional)")
    ext.add_argument("--patch_metric", type=str, default="blur_scores_val",
                     choices=("blur_scores_val", "blur_scores_mean"))
    ext.add_argument("--patch_threshold", type=float, default=None)
    ext.add_argument("--patch_threshold_percentile", type=float, default=None)
    ext.add_argument("--patch_codec", type=str, default="png", choices=("png", "npy"))
    ext.add_argument("--seed", type=int, default=42)

    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    result = run_pipeline(
        wsi_dir=args.wsi_dir,
        out_dir=args.out_dir,
        calc_blur=args.calc_blur,
        extract_patches=args.extract_patches,
        mag=args.mag,
        patch_size=args.patch_size,
        overlap=args.overlap,
        segmenter=args.segmenter,
        seg_conf_thresh=args.seg_conf_thresh,
        remove_holes=args.remove_holes,
        min_tissue_proportion=args.min_tissue_proportion,
        device=args.device,
        max_workers=args.max_workers,
        wsi_ext=args.wsi_ext,
        search_nested=args.search_nested,
        blur_n_workers=args.blur_n_workers,
        blur_overwrite=args.blur_overwrite,
        patch_encoder=args.patch_encoder,
        patch_encoder_ckpt_path=args.patch_encoder_ckpt_path,
        stain_normalize_target=args.stain_normalize_target,
        patch_metric=args.patch_metric,
        patch_threshold=args.patch_threshold,
        patch_threshold_percentile=args.patch_threshold_percentile,
        patch_codec=args.patch_codec,
        seed=args.seed,
    )

    print("-" * 30)
    print(f"coords  : {result['coords_dir']}")
    if result["features_dir"]:
        print(f"features: {result['features_dir']}")
    if result["patches_dir"]:
        print(f"patches : {result['patches_dir']}")

    # docs/howtouse.md の「失敗したスライドがあれば終了コードが1になる」という
    # 前提を、main.py経由(Step2/Step4)でも成り立たせる。
    if result["blur_failed"] or result["patch_extract_failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
