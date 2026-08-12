"""TRIDENT本体(`trident.Processor`)の呼び出しをラップするモジュール。

PLAN.md の Step1(セグメンテーション+パッチ座標抽出)と Step3(特徴量抽出)を
TRIDENTにそのまま委譲する。ぼやけスコア付与(Step2)や学習用パッチ抽出(Step4)
といった独自処理はここには含めず、呼び出し側の `main.py` が別途担当する。
"""
from pathlib import Path

from trident import Processor
from trident.patch_encoder_models import encoder_factory
from trident.segmentation_models import segmentation_model_factory


def format_mag(mag):
    """TRIDENT本体(`Processor.run_patching_job`)と同じ倍率の文字列化ルール。

    coords/featuresの出力先ディレクトリ名の一部として使われるため、ここがずれると
    `run_patch_feature_extraction_job` に渡す `coords_dir` が実際の座標の場所と
    食い違ってしまう。
    """
    m = float(mag)
    if abs(m - round(m)) < 1e-9:
        return str(int(round(m)))
    return f"{m:g}"


def coords_dirname(mag, patch_size, overlap=0):
    """座標・特徴量が入るサブディレクトリ名(job_dir相対)。"""
    return f"{format_mag(mag)}x_{patch_size}px_{overlap}px_overlap"


class TridentRunner:
    """`trident.Processor` をこのパッケージの既定値でラップしたもの。

    使い方::

        runner = TridentRunner("data/wsis", "results/trident")
        runner.segment()
        coords_h5_dir = runner.extract_coords()            # -> <job_dir>/<cdir>/patches
        features_dir = runner.extract_features("uni_v1")   # 任意
    """

    def __init__(self, wsi_dir, job_dir, *, mag=20.0, patch_size=256, overlap=0,
                 segmenter="hest", seg_conf_thresh=0.5, remove_holes=False,
                 min_tissue_proportion=0.0, device="cuda:0",
                 max_workers=None, skip_errors=True, wsi_ext=None,
                 search_nested=True):
        self.wsi_dir = Path(wsi_dir)
        self.job_dir = Path(job_dir)
        self.job_dir.mkdir(parents=True, exist_ok=True)

        self.mag = mag
        self.patch_size = patch_size
        self.overlap = overlap
        self.segmenter = segmenter
        self.seg_conf_thresh = seg_conf_thresh
        self.remove_holes = remove_holes
        self.min_tissue_proportion = min_tissue_proportion
        self.device = device

        # coords/featuresの置き場所はjob_dir相対の同じ文字列を使い回す必要がある。
        # run_patch_feature_extraction_job の coords_dir 引数は job_dir 相対の
        # ディレクトリ名として解釈される(絶対パスを渡すとjob_dirが二重結合される)。
        self.coords_dirname = coords_dirname(mag, patch_size, overlap)

        self.processor = Processor(
            job_dir=str(self.job_dir),
            wsi_source=str(self.wsi_dir),
            wsi_ext=wsi_ext,
            skip_errors=skip_errors,
            max_workers=max_workers,
            search_nested=search_nested,
        )

    @property
    def coords_dir(self):
        """座標(h5)が入るディレクトリの絶対パス。"""
        return self.job_dir / self.coords_dirname / "patches"

    def segment(self):
        """Step1前半: 組織セグメンテーション。"""
        seg_device = "cpu" if self.segmenter == "otsu" else self.device
        model = segmentation_model_factory(self.segmenter, confidence_thresh=self.seg_conf_thresh)
        self.processor.run_segmentation_job(
            model,
            seg_mag=model.target_mag,
            holes_are_tissue=not self.remove_holes,
            device=seg_device,
        )
        return self

    def extract_coords(self):
        """Step1後半: パッチ座標抽出。coords h5が入るディレクトリを返す。"""
        self.processor.run_patching_job(
            target_magnification=self.mag,
            patch_size=self.patch_size,
            overlap=self.overlap,
            saveto=self.coords_dirname,
            min_tissue_proportion=self.min_tissue_proportion,
        )
        return self.coords_dir

    def extract_features(self, patch_encoder="uni_v1", patch_encoder_ckpt_path=None,
                          stain_normalize_target=None):
        """Step3: 特徴量抽出。特徴量h5が入るディレクトリを返す。

        stain_normalize_target: 指定すると、そのパス(画像1枚)を目標染色として
        Macenko染色正規化(torchstain)をencoderの`eval_transforms`の先頭に差し込む。
        パッチ画像自体はディスクに保存されず、特徴量抽出時にオンザフライで適用される。
        """
        encoder = encoder_factory(patch_encoder, weights_path=patch_encoder_ckpt_path)
        if stain_normalize_target:
            from src.stain_norm import wrap_eval_transforms
            encoder.eval_transforms = wrap_eval_transforms(
                encoder.eval_transforms, stain_normalize_target
            )
        features_dir = self.processor.run_patch_feature_extraction_job(
            coords_dir=self.coords_dirname,
            patch_encoder=encoder,
            device=self.device,
            saveas="h5",
        )
        return Path(features_dir)
