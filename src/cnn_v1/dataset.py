# --- file: src/cnn_v1/dataset.py ---
from __future__ import annotations
import os, glob, json
from bisect import bisect_right
from typing import Tuple, Dict, Optional
import numpy as np
import torch
from torch.utils.data import Dataset

# 新增依赖
import warnings as _warnings
try:
    import pandas as _pd
except Exception:
    _pd = None

# 每个 worker 维护一个简易的“最近一次”分片缓存，减少重复 I/O
_WORKER_CACHE: Dict[int, Dict[str, object]] = {}

def _get_worker_cache() -> Dict[str, object]:
    pid = os.getpid()
    if pid not in _WORKER_CACHE:
        _WORKER_CACHE[pid] = {'last_part_id': None, 'last_data': None}
    return _WORKER_CACHE[pid]


class ShardedUTRDataset(Dataset):
    """
    统一读取两种分片形态（自动适配）：
    - **PT 模式**（你最新重分片）：`shards/data.part-*.pt`（推荐）
      每个 .pt 为一个 dict/tuple，至少包含（utr5, utr3, labels）。
    - 兼容旧 **NPZ 模式**：`features_utr5.part-*.npz` 等三件套。

    采用“分片级懒加载 + 每 worker 最近分片缓存”。长度与索引使用前缀和定位，避免在 __init__ 中加载全部数据。
    """

    def __init__(self, dataset_dir: str, split: str):
        self.dataset_dir = dataset_dir
        self.split = split
        shards_dir = os.path.join(dataset_dir, 'shards')

        # 优先识别 PT 分片
        self.pt_paths = sorted(glob.glob(os.path.join(shards_dir, 'data.part-*.pt')))
        self.mode = None
        if len(self.pt_paths) > 0:
            self.mode = 'pt'
            self.part_paths = self.pt_paths
            self.part_sizes = self._load_part_sizes_from_manifest() or [self._pt_part_length(i) for i in range(len(self.part_paths))]
        else:
            # 兼容老的 npz 三件套
            u5 = sorted(glob.glob(os.path.join(shards_dir, 'features_utr5.part-*.npz')))
            u3 = sorted(glob.glob(os.path.join(shards_dir, 'features_utr3.part-*.npz')))
            yy = sorted(glob.glob(os.path.join(shards_dir, 'labels.part-*.npz')))
            assert len(u5) == len(u3) == len(yy) and len(u5) > 0, "未发现 PT 分片，也没有匹配到 NPZ 三件套。"
            self.mode = 'npz'
            self.u5_paths, self.u3_paths, self.y_paths = u5, u3, yy
            self.part_sizes = [self._npz_part_length(p) for p in self.u5_paths]
        # 前缀和用于 O(1) len，O(logN) 定位 idx→(part_id, local_idx)
        self.cum_sizes = np.cumsum(self.part_sizes).astype(np.int64)

        # ---- 按 split 载入索引表（若存在）；否则给出强警告并回退到全集 ----
        self._has_split = False
        self._idx_part = None  # np.ndarray[int]
        self._idx_local = None # np.ndarray[int]
        self._init_split_indices()

    # ---- 基本协议 ----
    def __len__(self):
        return int(self._idx_part.size) if self._has_split else int(self.cum_sizes[-1])

    def _locate(self, global_idx: int) -> Tuple[int, int]:
        # 返回 (part_id, local_idx)
        pid = int(bisect_right(self.cum_sizes, global_idx))
        prev = int(self.cum_sizes[pid-1]) if pid > 0 else 0
        li = int(global_idx - prev)
        return pid, li

    def __getitem__(self, idx: int):
        if self._has_split:
            pid = int(self._idx_part[idx]); li = int(self._idx_local[idx])
        else:
            pid, li = self._locate(idx)

        if self.mode == 'pt':
            x5, x3, y = self._get_item_from_pt(pid, li)
        else:
            x5, x3, y = self._get_item_from_npz(pid, li)
        # 统一返回 torch 张量
        x5 = _ensure_ch_first(torch.as_tensor(x5).float())  # (C,L)
        x3 = _ensure_ch_first(torch.as_tensor(x3).float())
        y  = torch.as_tensor(y)
        if y.ndim==0 or (y.ndim==1 and y.numel()==1):
            y = y.long().view(() )
        else:
            y = y.float().view(-1)
        return {'utr5': x5, 'utr3': x3, 'label': y}

    # ---- PT 模式 ----
    def _load_part_sizes_from_manifest(self) -> Optional[list]:
        man_path = os.path.join(self.dataset_dir, 'manifest.json')
        if not os.path.exists(man_path):
            return None
        try:
            with open(man_path, 'r', encoding='utf-8') as f:
                man = json.load(f)
        except Exception:
            return None
        # 支持多种可能字段名：parts/shards + size/num_rows/length
        parts = man.get('parts') or man.get('shards') or man.get('files')
        if not isinstance(parts, list):
            return None
        # 以文件名为键构造映射
        size_keys = ('size', 'num_rows', 'length', 'n')
        fn2size = {}
        for p in parts:
            path = (p.get('path') or p.get('file') or p.get('name') or '')
            base = os.path.basename(path)
            size = None
            for k in size_keys:
                if k in p:
                    size = int(p[k])
                    break
            if base and size is not None:
                fn2size[base] = size
        sizes = []
        for p in self.pt_paths:
            base = os.path.basename(p)
            if base not in fn2size:
                return None
            sizes.append(int(fn2size[base]))
        return sizes if len(sizes) == len(self.pt_paths) else None

    def _pt_part_length(self, part_id: int) -> int:
        data = torch.load(self.pt_paths[part_id], map_location='cpu')
        x5, x3, y = _unpack_triplet(data)
        n = int(_first_dim(x5))
        # 显式释放临时对象
        del data, x5, x3, y
        return n

    def _get_item_from_pt(self, part_id: int, local_idx: int):
        cache = _get_worker_cache()
        if cache['last_part_id'] != part_id or cache['last_data'] is None:
            data = torch.load(self.pt_paths[part_id], map_location='cpu')
            cache['last_part_id'] = part_id
            cache['last_data'] = data
        x5, x3, y = _unpack_triplet(cache['last_data'])
        return x5[local_idx], x3[local_idx], y[local_idx]

    # ---- NPZ 模式（兼容） ----
    def _npz_part_length(self, path: str) -> int:
        z = np.load(path, mmap_mode='r')
        arr = z[list(z.keys())[0]]
        return int(arr.shape[0])

    def _get_item_from_npz(self, part_id: int, local_idx: int):
        cache = _get_worker_cache()
        tag = ('npz', part_id)
        if cache['last_part_id'] != tag or cache['last_data'] is None:
            a5 = np.load(self.u5_paths[part_id], mmap_mode='r')
            a3 = np.load(self.u3_paths[part_id], mmap_mode='r')
            ay = np.load(self.y_paths[part_id],  mmap_mode='r')
            data = (a5[list(a5.keys())[0]], a3[list(a3.keys())[0]], ay[list(ay.keys())[0]])
            cache['last_part_id'] = tag
            cache['last_data'] = data
        x5, x3, y = cache['last_data']
        return x5[local_idx], x3[local_idx], y[local_idx]

    # ---- split 索引加载 ----
    def _init_split_indices(self) -> None:
        """
        读取 <dataset_dir> 下的 split 索引表，支持以下路径优先级：
          1) index/{split}/index.parquet
          2) {split}.index.parquet
          3) index/{split}/index.csv
          4) {split}.index.csv
          5) index/{split}/index.json
          6) {split}.index.json
        支持字段方案：
          - 方案A: 'part_id', 'local_idx'
          - 方案B: 'global_idx'（将映射为 A）
        """
        dataset_dir = self.dataset_dir
        split = self.split
        cands = [
            os.path.join(dataset_dir, 'index', split, 'index.parquet'),
            os.path.join(dataset_dir, f'{split}.index.parquet'),
            os.path.join(dataset_dir, 'index', split, 'index.csv'),
            os.path.join(dataset_dir, f'{split}.index.csv'),
            os.path.join(dataset_dir, 'index', split, 'index.json'),
            os.path.join(dataset_dir, f'{split}.index.json'),
        ]
        path = next((p for p in cands if os.path.exists(p)), None)
        if path is None:
            _warnings.warn(f"[ShardedUTRDataset] Split index for '{split}' not found under {dataset_dir}. "
                           f"Falling back to FULL dataset (evaluation will be meaningless).", RuntimeWarning)
            self._has_split = False
            return

        df = self._load_index_df(path)
        cols = {c.lower(): c for c in df.columns}
        if 'part_id' in cols and 'local_idx' in cols:
            part = df[cols['part_id']].to_numpy(dtype=np.int64, copy=False)
            loc  = df[cols['local_idx']].to_numpy(dtype=np.int64, copy=False)
        elif 'global_idx' in cols:
            gid = df[cols['global_idx']].to_numpy(dtype=np.int64, copy=False)
            # 向量化 global_idx -> (part_id, local_idx)
            part = np.searchsorted(self.cum_sizes, gid, side='right').astype(np.int64)
            prev = np.concatenate(([0], self.cum_sizes[:-1]))
            loc  = (gid - prev[part]).astype(np.int64)
        # ---- 方案C：把业务清单视作索引（包含 sample_id / split 等列）----
        elif 'sample_id' in cols:
            # 若清单里还带 split 字段，则按当前 split 过滤一遍以避免误用
            if 'split' in cols:
                split_col = cols['split']
                # 仅当文件确实混有不同 split 时才筛选；否则不动
                unique_splits = set(str(s).lower() for s in df[split_col].unique())
                if len(unique_splits) > 1 and self.split.lower() in unique_splits:
                    df = df[df[split_col].astype(str).str.lower() == self.split.lower()]
            gid = df[cols['sample_id']].to_numpy(dtype=np.int64, copy=False)
            # 仍然将 sample_id 视为“全局行号”来映射到 (part_id, local_idx)
            part = np.searchsorted(self.cum_sizes, gid, side='right').astype(np.int64)
            prev = np.concatenate(([0], self.cum_sizes[:-1]))
            loc  = (gid - prev[part]).astype(np.int64)
            print(f"[Data] Using 'sample_id' as global_idx for split '{self.split}'.")
        else:
            raise ValueError(f"Unsupported index schema in {path}. Expect columns "
                             f"('part_id','local_idx') or ('global_idx'). Got: {list(df.columns)}")

        # 基本校验
        if part.size == 0:
            raise ValueError(f"Empty split index in {path}")
        if part.min() < 0 or part.max() >= len(self.part_sizes):
            raise IndexError(f"part_id out of range in {path}")
        if (loc < 0).any() or (loc >= np.asarray(self.part_sizes, dtype=np.int64)[part]).any():
            raise IndexError(f"local_idx out of range in {path}")

        self._idx_part  = part
        self._idx_local = loc
        self._has_split = True
        print(f"[Data] Loaded split '{split}' indices from {path}: {len(self._idx_part)} samples")

    def _load_index_df(self, path: str):
        suf = os.path.splitext(path)[-1].lower()
        if suf == '.parquet':
            if _pd is None:
                raise RuntimeError("pandas/pyarrow is required to read parquet indexes. "
                                   "Please `pip install 'pandas[parquet]'`.")
            return _pd.read_parquet(path)
        elif suf == '.csv':
            if _pd is None:
                raise RuntimeError("pandas is required to read csv indexes. Please `pip install pandas`.")
            return _pd.read_csv(path)
        elif suf == '.json':
            # 支持两种：纯列表或对象里某个键
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return _pd.DataFrame(obj) if _pd else _json_rows_to_df(obj)
            elif isinstance(obj, dict):
                # 选最可能的键
                for k in ('data','rows','indices','index','items'):
                    if k in obj and isinstance(obj[k], list):
                        return _pd.DataFrame(obj[k]) if _pd else _json_rows_to_df(obj[k])
                raise ValueError(f'Unsupported json index format in {path}')
        else:
            raise ValueError(f'Unsupported index file suffix: {suf}')


# ---- 工具函数 ----
def _unpack_triplet(obj):
    """从 checkpoint/分片载入对象中抽取 (utr5, utr3, labels)。
    允许以下形态：
      - dict: 可能的键名组合 ('utr5'|'x5'|'u5'), ('utr3'|'x3'|'u3'), ('labels'|'y'|'label')
      - tuple/list: (utr5, utr3, labels)
    返回 numpy 或 torch 张量，后续由上层转为 torch 并统一 (C,L) 排布。
    """
    if isinstance(obj, dict):
        def pick(d, keys):
            for k in keys:
                if k in d:
                    return d[k]
            raise KeyError(keys)
        x5 = pick(obj, ('utr5','x5','u5'))
        x3 = pick(obj, ('utr3','x3','u3'))
        y  = pick(obj, ('labels','y','label'))
        return x5, x3, y
    elif isinstance(obj, (list, tuple)) and len(obj) == 3:
        return obj[0], obj[1], obj[2]
    else:
        raise ValueError('Unsupported shard object type for pt mode')


def _first_dim(x) -> int:
    if isinstance(x, torch.Tensor):
        return int(x.shape[0])
    elif isinstance(x, np.ndarray):
        return int(x.shape[0])
    else:
        # 可能是 memmap 或其他 array-like
        try:
            return int(x.shape[0])
        except Exception as e:
            raise TypeError(f'Unsupported array type: {type(x)}')


def _ensure_ch_first(x: torch.Tensor) -> torch.Tensor:
    """把单样本张量调整到 (C, L)。容错支持 (L, C) 或 (C, L)。"""
    if x.dim() == 2:
        C, L = x.shape[0], x.shape[1]
        if C <= 8 and L > C:  # 常见 onehot 小通道推断
            return x  # (C,L)
        else:
            return x.transpose(0,1)  # 假定 (L,C)
    elif x.dim() == 3:
        # 可能是 (N,C,L) 的切片未 squeeze；尽量挤掉 batch 维
        return x.squeeze(0) if x.shape[0] == 1 else x[0]
    else:
        raise ValueError(f'Unexpected sample tensor shape: {tuple(x.shape)}')


def _json_rows_to_df(rows):
    # 极小兜底，不强依赖 pandas：只支持 part_id/local_idx 或 global_idx 三列之一
    keys = set().union(*(r.keys() for r in rows))
    sel = None
    for cand in (('part_id','local_idx'), ('global_idx',)):
        if set(cand).issubset(keys):
            sel = cand; break
    if sel is None:
        raise ValueError(f"Unsupported json rows schema for split index: keys={sorted(keys)}")
    # 构造“类 DataFrame”对象，只需 .columns/.to_numpy 即可
    class _DF:
        def __init__(self, rows, cols):
            self._rows = rows; self._cols = list(cols); self.columns = self._cols
        def __getitem__(self, key):
            arr = np.array([r[key] for r in self._rows])
            class _Col:
                def __init__(self, arr): self._a = arr
                def to_numpy(self, dtype=None, copy=False): 
                    a = self._a.astype(dtype) if dtype is not None else self._a
                    return a.copy() if copy else a
            return _Col(arr)
    return _DF(rows, sel)
