"""blurスコア付きのパッチh5群から、学習用データセットを構築する。

流れは以下の通り。

1. スライドごとに blur スコアの閾値でパッチを絞り込む
2. 残ったパッチからランダムに N 枚サンプリングする
3. 全スライド分をまとめて WebDataset (tar) か LMDB に書き出す

WebDataset の書き出しは標準ライブラリの tarfile だけで行うため、
webdataset パッケージへの依存は無い（読み出し側で使う分には互換）。
"""
import io
import json
import pickle
import random
import re
import tarfile
import zlib
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np

from src.patch_source import PatchSource, slide_stem

# 画素をバイト列に符号化する方式。いずれも可逆（blur評価を壊さないため非可逆は避ける）
CODECS = ("png", "npy")


def make_slide_ids(h5_paths, root):
    """h5パス群から一意な slide_id を割り当てる。

    slide_id は tar内のサンプルキーと manifest の集計キーを兼ねるため、重複すると
    別スライドのパッチが同じサンプルに混ざる。フォルダ分けされていると別フォルダに
    同名のh5が置かれうるので、ここで衝突を解消しておく。

    衝突していないstemはそのまま使う（臨床メタデータとの突き合わせで素のスライド名が
    要ることが多いため）。衝突したものだけ相対パスを繋いだIDに置き換える。

    stemはTRIDENT形式("<slide>_patches.h5")の "_patches" サフィックスを剥がした
    `slide_stem()` を使う。生のファイル名(`path.stem`)のままだと、素のWSI名と
    一致しないslide_idになってしまう。
    """
    counts = Counter(slide_stem(p) for p in h5_paths)

    ids = {}
    taken = set()
    for path in h5_paths:
        stem = slide_stem(path)
        if counts[stem] == 1:
            base = stem
        else:
            rel = path.relative_to(root).with_suffix("").with_name(stem)
            base = re.sub(r"[^0-9A-Za-z._-]+", "_", "_".join(rel.parts))

        # 相対パス由来のIDが、たまたま別スライドの素のstemと当たることもある
        slide_id = base
        n = 1
        while slide_id in taken:
            slide_id = f"{base}_{n}"
            n += 1

        taken.add(slide_id)
        ids[path] = slide_id

    return ids


def encode_patch(patch_rgb, codec):
    """RGB配列を (拡張子, バイト列) に符号化する。"""
    if codec == "png":
        # cv2はBGR前提で書き出すため、ファイル上の色順が正しくなるよう変換する
        ok, buf = cv2.imencode(".png", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError("PNGへの符号化に失敗しました")
        return "png", buf.tobytes()
    if codec == "npy":
        bio = io.BytesIO()
        np.save(bio, patch_rgb, allow_pickle=False)
        return "npy", bio.getvalue()
    raise ValueError(f"未知のcodec: {codec}")


def select_indices(h5_path, metric, threshold, percentile, n_per_slide, rng):
    """閾値を満たすパッチから n_per_slide 枚をランダムに選ぶ。

    戻り値は (選択されたindexの配列, そのスコア配列, 閾値を満たした総数)。
    """
    with h5py.File(h5_path, "r") as f:
        if metric not in f:
            raise KeyError(
                f"{h5_path.name}: '{metric}' がありません。先に add_blur_scores.py を実行してください"
            )
        scores = f[metric][:]

    # 背景しか無いスライドなどでパッチが0枚のことがある。np.percentileは
    # 空配列で例外を投げるため、閾値の計算前に抜ける。
    if len(scores) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype="float32"), 0

    # 絶対閾値と分位点はどちらか一方。分位点はスライドごとに基準が変わるので
    # スキャナ差でスコアのスケールがずれる場合に有効。
    if percentile is not None:
        cut = float(np.percentile(scores, percentile))
    else:
        cut = threshold

    eligible = np.flatnonzero(scores >= cut)
    if len(eligible) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype="float32"), 0

    if n_per_slide is None or len(eligible) <= n_per_slide:
        chosen = eligible
    else:
        chosen = rng.choice(eligible, size=n_per_slide, replace=False)

    chosen = np.sort(chosen)
    return chosen, scores[chosen], len(eligible)


def collect_slide_samples(h5_path, wsi_path, slide_id, patch_size, patch_level,
                          metric, threshold, percentile, n_per_slide, seed, codec):
    """1スライド分の符号化済みサンプルを返す。ワーカープロセスから呼ばれる。"""
    h5_path = Path(h5_path)
    # スライドごとにseedを固定し、並列実行でも結果が再現するようにする。
    # 組み込みhash()はプロセス毎に撹拌されるため、安定なcrc32を使う。
    slide_seed = zlib.crc32(f"{seed}:{slide_id}".encode()) & 0xFFFFFFFF
    rng = np.random.default_rng(slide_seed)

    indices, scores, n_eligible = select_indices(
        h5_path, metric, threshold, percentile, n_per_slide, rng
    )
    if len(indices) == 0:
        return slide_id, [], 0

    samples = []
    with PatchSource(h5_path, wsi_path, patch_size, patch_level) as src:
        for idx, score in zip(indices, scores):
            patch = src.read_patch(int(idx))
            ext, payload = encode_patch(patch, codec)
            x, y = (int(v) for v in src.coords[idx])
            meta = {
                "slide_id": slide_id,
                "patch_index": int(idx),
                "x": x,
                "y": y,
                "patch_size": src.patch_size,
                "patch_level": src.patch_level,
                metric: float(score),
            }
            samples.append((f"{slide_id}_{int(idx):06d}", ext, payload, meta))

    return slide_id, samples, n_eligible


class ShardWriter:
    """WebDataset互換のtarシャードを順に書き出す。"""

    def __init__(self, out_dir, prefix="patches", max_count=10000, max_size_mb=1024):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_count = max_count
        self.max_size = max_size_mb * 1024 * 1024

        self.shard_index = -1
        self.tar = None
        self.count = self.size = 0
        self.total = 0
        self.shards = []

    def _open_next(self):
        self._close_current()
        self.shard_index += 1
        path = self.out_dir / f"{self.prefix}-{self.shard_index:06d}.tar"
        self.tar = tarfile.open(path, "w")
        self.count = self.size = 0
        self.shards.append(path)

    def _close_current(self):
        if self.tar is not None:
            self.tar.close()
            self.tar = None

    def _add(self, name, payload):
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        self.tar.addfile(info, io.BytesIO(payload))

    def write(self, key, ext, payload, meta):
        if self.tar is None or self.count >= self.max_count or self.size >= self.max_size:
            self._open_next()

        meta_payload = json.dumps(meta, ensure_ascii=False).encode()
        # 同一サンプルは必ず同じシャードの連続位置に置く（WebDatasetの結合規約）
        self._add(f"{key}.{ext}", payload)
        self._add(f"{key}.json", meta_payload)

        self.count += 1
        self.size += len(payload) + len(meta_payload)
        self.total += 1

    def close(self):
        self._close_current()


class LMDBWriter:
    """LMDBに連番キーで書き出す。"""

    def __init__(self, out_dir, map_size_gb=512):
        try:
            import lmdb
        except ImportError as e:
            raise ImportError(
                "LMDB出力には lmdb パッケージが必要です: uv add lmdb"
            ) from e

        self.out_dir = Path(out_dir)
        self.out_dir.parent.mkdir(parents=True, exist_ok=True)
        # map_sizeはスパースに確保されるため、大きめでも実容量は消費しない
        self.env = lmdb.open(str(self.out_dir), map_size=int(map_size_gb * 1024**3),
                             subdir=True, readonly=False, meminit=False, map_async=True)
        self.txn = self.env.begin(write=True)
        self.total = 0

    def write(self, key, ext, payload, meta):
        record = {"key": key, "ext": ext, "data": payload, "meta": meta}
        self.txn.put(f"{self.total:08d}".encode(), pickle.dumps(record))
        self.total += 1
        # 定期的にコミットしてトランザクションの肥大を防ぐ
        if self.total % 1000 == 0:
            self.txn.commit()
            self.txn = self.env.begin(write=True)

    def close(self):
        self.txn.put(b"__len__", pickle.dumps(self.total))
        self.txn.commit()
        self.env.sync()
        self.env.close()


class ShuffleBuffer:
    """有界のシャッフルバッファ。スライド単位の偏りを崩して書き出す。"""

    def __init__(self, writer, size, seed):
        self.writer = writer
        self.size = max(1, size)
        self.rng = random.Random(seed)
        self.buf = []

    def write(self, sample):
        self.buf.append(sample)
        if len(self.buf) >= self.size:
            self._flush_one()

    def _flush_one(self):
        i = self.rng.randrange(len(self.buf))
        self.buf[i], self.buf[-1] = self.buf[-1], self.buf[i]
        self.writer.write(*self.buf.pop())

    def close(self):
        while self.buf:
            self._flush_one()
        self.writer.close()
