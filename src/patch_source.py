"""パッチh5への統一アクセス層。

対象とするh5は2種類ある。

1. TRIDENT/CLAM形式 : `coords` のみを持ち、画素は元WSIから読み直す必要がある
2. `images` に画素そのものを持つ形式 : 過去に生成されたh5との互換用
   (現在このリポジトリで新規に画素埋め込みh5を生成する経路は無い)

どちらでも `PatchSource.read_patch(i)` でRGBのuint8配列が得られるようにする。
"""
import os
from pathlib import Path

import h5py
import numpy as np

# WSI本体として探索する拡張子（優先度順）
WSI_EXTENSIONS = (".svs", ".ndpi", ".tif", ".tiff", ".mrxs", ".scn", ".vms", ".bif")

# パッチh5として探索する拡張子
H5_EXTENSIONS = (".h5", ".hdf5")

# TRIDENTが座標h5を書き出す際のファイル名サフィックス ("<slide>_patches.h5")。
# 画素埋め込み形式("<slide>.h5")には無い。
_TRIDENT_COORDS_SUFFIX = "_patches"


def slide_stem(h5_path):
    """パッチh5のファイル名から、対応するWSIのstem(拡張子抜きファイル名)を求める。

    TRIDENTは `<slide>_patches.h5` という名前で座標h5を書き出すため、単純に
    `Path(h5_path).stem` を使うと "<slide>_patches" のままWSI検索やslide_id算出に
    渡ってしまい、元のWSI名と一致しない。末尾が `_patches` の場合だけ剥がす
    (画素埋め込み形式はサフィックスが無いため何もしない)。
    """
    stem = Path(h5_path).stem
    if stem.endswith(_TRIDENT_COORDS_SUFFIX):
        return stem[: -len(_TRIDENT_COORDS_SUFFIX)]
    return stem


def iter_files(root, extensions):
    """root以下のファイルを拡張子で絞って再帰的に列挙する。

    フォルダ分けの深さは問わない。data/直下でも data/caseA/2024/ のような
    多段構成でも同じように拾う。pathlibのrglobと違う点は2つ。

    - 拡張子の大文字小文字を無視する（`.SVS` のような表記が混ざることがある）
    - シンボリックリンクされたディレクトリも辿る（実体を別ボリュームに置いて
      リンクを張った構成が珍しくないため）

    rootがファイルなら、拡張子が一致する場合のみそれ1件を返す。
    """
    root = Path(root)
    exts = {e.lower() for e in extensions}

    if root.is_file():
        return [root] if root.suffix.lower() in exts else []

    found = []
    visited = set()
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        # リンクが循環していると無限に降りてしまうため、実体パスで訪問済みを管理する
        real = os.path.realpath(dirpath)
        if real in visited:
            dirnames[:] = []
            continue
        visited.add(real)

        for name in filenames:
            if os.path.splitext(name)[1].lower() in exts:
                found.append(Path(dirpath) / name)

    return sorted(found)


def iter_wsi_paths(root):
    """root以下のWSIを再帰的に列挙する。"""
    return iter_files(root, WSI_EXTENSIONS)


def iter_h5_paths(root):
    """root以下のパッチh5を再帰的に列挙する。"""
    return iter_files(root, H5_EXTENSIONS)


def _get_attr(h5file, keys, default=None):
    """coordsデータセットとファイルルートの両方から属性を探す。

    CLAM系は `coords` の属性に、実装によってはルート属性に書かれるため両対応する。
    """
    holders = []
    if "coords" in h5file:
        holders.append(h5file["coords"].attrs)
    holders.append(h5file.attrs)

    for holder in holders:
        for key in keys:
            if key in holder:
                value = holder[key]
                # h5py はスカラーを0次元配列で返すことがある
                if isinstance(value, np.ndarray) and value.ndim == 0:
                    value = value.item()
                if isinstance(value, bytes):
                    value = value.decode()
                return value
    return default


class WSIIndex:
    """wsi_dir を一度だけ走査して、stemからWSI本体を引けるようにした索引。

    h5ごとに毎回rglobし直すと、スライド数×拡張子数だけツリー全体を歩くことになり、
    フォルダ分けされた大きなデータ置き場では探索だけで時間を食う。ここで一度に
    集めておく。

    フォルダ分けされていると別フォルダに同名のWSIが存在しうる。その場合は
    h5側の相対フォルダと一致するものを優先して対応付ける。
    """

    def __init__(self, wsi_dir):
        self.root = Path(wsi_dir)
        self._by_stem = {}
        for path in iter_wsi_paths(self.root):
            self._by_stem.setdefault(path.stem.lower(), []).append(path)

        # 同名が複数あったstem。呼び出し側が警告を出せるように公開しておく
        self.ambiguous_stems = {
            stem for stem, paths in self._by_stem.items() if len(paths) > 1
        }

    def _rank(self, path):
        """曖昧なときの決定順。拡張子の優先度が同じならパス順で決める。"""
        ext = path.suffix.lower()
        order = WSI_EXTENSIONS.index(ext) if ext in WSI_EXTENSIONS else len(WSI_EXTENSIONS)
        return (order, str(path))

    def find(self, slide_stem, rel_dir=None):
        """stemに対応するWSIを返す。見つからなければ None。

        rel_dir にh5側の相対フォルダを渡すと、同じ相対位置にあるWSIを優先する。
        """
        candidates = self._by_stem.get(slide_stem.lower())
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        if rel_dir is not None:
            rel_dir = Path(rel_dir)
            # 1) 相対パスが完全に一致するもの（results/h5/caseA -> data/caseA）
            exact = [p for p in candidates if p.parent == self.root / rel_dir]
            if exact:
                return min(exact, key=self._rank)
            # 2) 途中の階層がずれていても、末尾のフォルダ名が一致すれば拾う
            if rel_dir.parts:
                tail = rel_dir.parts[-1]
                near = [p for p in candidates if p.parent.name == tail]
                if near:
                    return min(near, key=self._rank)

        return min(candidates, key=self._rank)


def find_wsi(slide_stem, wsi_dir, rel_dir=None):
    """stemに対応するWSI本体を再帰的に探す。見つからなければ None。

    単発の探索向け。多数のh5を捌くときは WSIIndex を使い回すこと。
    """
    return WSIIndex(wsi_dir).find(slide_stem, rel_dir)


class PatchSource:
    """1スライド分のパッチh5を読むためのコンテキストマネージャ。"""

    def __init__(self, h5_path, wsi_path=None, patch_size=None, patch_level=None):
        self.h5_path = Path(h5_path)
        self.wsi_path = Path(wsi_path) if wsi_path else None
        self._patch_size_override = patch_size
        self._patch_level_override = patch_level

        self._h5 = None
        self._slide = None
        self.coords = None
        self.has_images = False
        self.patch_size = None
        self.patch_level = None

    def __enter__(self):
        self._h5 = h5py.File(self.h5_path, "r")

        if "coords" not in self._h5:
            raise KeyError(f"{self.h5_path.name}: 'coords' データセットがありません")
        self.coords = self._h5["coords"][:]
        self.has_images = "images" in self._h5

        # パッチサイズ/レベルの解決（明示指定 > h5属性 > 既定値）
        # TRIDENT出力は"patch_level"属性を持たずlevel 0固定で読む前提のため、
        # level 0でのクロップ幅である patch_size_level0 を優先する。
        # (patch_size はターゲット倍率上の呼称サイズで、スライドのネイティブ倍率が
        #  ターゲット倍率と異なると level 0 上では別の画素数になり、そのまま使うと
        #  クロップ範囲がずれる。画素埋め込み形式のh5は patch_size_level0 を
        #  持たないため、この優先順でも従来通り patch_size がそのまま使われる)
        used_patch_size_level0 = self._patch_size_override is None and (
            _get_attr(self._h5, ("patch_size_level0",), None) is not None
        )
        self.patch_size = self._patch_size_override or int(
            _get_attr(self._h5, ("patch_size_level0", "patch_size"), 256)
        )
        if self._patch_level_override is not None:
            self.patch_level = self._patch_level_override
        else:
            self.patch_level = int(_get_attr(self._h5, ("patch_level",), 0))

        # patch_size_level0 は定義上level 0でのクロップ幅であり、level 0以外を
        # 明示指定された場合にそのまま使うとクロップ範囲がずれる(ダウンサンプル分の
        # 換算をしていないため)。この組み合わせは対応していないので黙って進めず、
        # 早期に検出する。
        if self.patch_level != 0 and used_patch_size_level0:
            raise ValueError(
                f"{self.h5_path.name}: patch_size_level0 (level 0基準のサイズ) と "
                f"patch_level={self.patch_level} (0以外) の組み合わせは未対応です。"
                "level 0以外で読み出す場合は patch_size を明示指定してください。"
            )

        if not self.has_images:
            if self.wsi_path is None:
                raise FileNotFoundError(
                    f"{self.h5_path.name} は画素を含まないため元WSIが必要ですが、指定されていません"
                )
            import tiffslide

            self._slide = tiffslide.TiffSlide(str(self.wsi_path))

        return self

    def __exit__(self, *exc):
        if self._slide is not None:
            self._slide.close()
        if self._h5 is not None:
            self._h5.close()
        return False

    def __len__(self):
        return len(self.coords)

    def read_patch(self, i):
        """i番目のパッチをRGBのuint8配列で返す。"""
        if self.has_images:
            return self._h5["images"][i]

        x, y = (int(v) for v in self.coords[i])
        # read_region の location は openslide 同様つねにレベル0基準
        region = self._slide.read_region(
            (x, y), self.patch_level, (self.patch_size, self.patch_size)
        )
        return np.asarray(region.convert("RGB"))
