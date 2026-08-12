"""Macenko染色正規化(torchstain)を特徴量抽出パイプラインに差し込むためのモジュール。

TRIDENTのpatch encoder(`trident.patch_encoder_models`)は`eval_transforms`という
PIL Image -> Tensorの変換をエンコーダごとに持っており、`WSIPatcherDataset`が
パッチを読み出すたびにこれを適用する(`trident/wsi_objects/WSIPatcherDataset.py`)。
`MacenkoStainNormalizer`はこのパイプラインの先頭に差し込むPIL Image -> PIL Imageの
変換で、パッチ画像自体はディスクに保存せずオンザフライで正規化する。
"""
import numpy as np
import torch
import torchstain
from PIL import Image


class MacenkoStainNormalizer:
    """torchstainのMacenko正規化をPIL Image -> PIL Imageの変換として使うラッパー。

    Macenko法は組織がほとんど写っていないパッチ(背景・脂肪組織など)で
    OD(optical density)の固有値分解が破綻し例外を投げることがある。
    そのパッチ1枚のために特徴量抽出ジョブ全体を落とすのは避けたいため、
    正規化に失敗した場合は元のパッチをそのまま返す。
    """

    def __init__(self, target_image_path):
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        with Image.open(target_image_path) as img:
            target = img.convert("RGB")
        target_t = torch.from_numpy(np.array(target)).permute(2, 0, 1).to(torch.float32)
        self.normalizer.fit(target_t)

    def __call__(self, patch: Image.Image) -> Image.Image:
        patch_t = torch.from_numpy(np.array(patch.convert("RGB"))).permute(2, 0, 1)
        try:
            normalized, _, _ = self.normalizer.normalize(I=patch_t, stains=False)
        except Exception:
            return patch
        return Image.fromarray(normalized.numpy().astype(np.uint8))


def wrap_eval_transforms(eval_transforms, target_image_path):
    """encoderの`eval_transforms`の先頭にMacenko正規化を差し込んだ変換を返す。

    `eval_transforms`はエンコーダごとに実体が異なる(torchvision Compose、
    timmのtransform等)ため、内部リストを直接いじらず外側でラップする。
    """
    stain_normalize = MacenkoStainNormalizer(target_image_path)

    def combined(patch):
        return eval_transforms(stain_normalize(patch))

    return combined
