from typing import Optional, Tuple, Any
import glob
import numpy as np
from torch.utils.data import Dataset


def do_nothing(x: Any) -> Any:
    return x


def do_nothing_sampling(index: int, n: int, *args) -> Any:
    return args


class HSIData(Dataset):
    def __init__(
        self,
        data_dir: str,
        label_dir: Optional[str] = None,
        transform=do_nothing,
    ):
        self.data_paths = sorted(glob.glob(f"{data_dir}/*.npy"))
        if isinstance(label_dir, str):
            self.label_paths = sorted(glob.glob(f"{label_dir}/*.npy"))
        else:
            self.label_paths = None
        self.transform = transform

    def __getitem__(self, index) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        # ひとまずnumpyでのloadを前提に制作する
        data = np.load(self.data_paths[index], mmap_mode="r")
        data = self.transform(data)  # データ処理
        if self.label_paths is None:
            return data, None  # ひとまずNone
        label = np.load(self.label_paths[index], mmap_mode="r")
        return data.astype(np.float32), label.astype(
            np.int64
        )  # ここでタイプを変更するようにする

    def __len__(self):
        if self.label_paths is not None:
            assert len(self.data_paths) == len(self.label_paths)
        return len(self.data_paths)


class HSIDataWithSampling(Dataset):
    def __init__(
        self,
        data_path: str,
        label_path: Optional[str] = None,
        sample_square_size: int = 64,
        sampler=do_nothing_sampling,
        transform=do_nothing,
    ):
        self.data = np.load(data_path, mmap_mode="r")
        if label_path:
            self.label = np.load(label_path, mmap_mode="r")
        else:
            self.label = None
        self.sample_square_size = sample_square_size
        self.sampler = sampler
        self.transform = transform

    def __len__(self):
        H, W, C = self.data.shape
        length = (H - self.sample_square_size + 1) * (
            W - self.sample_square_size + 1
        )
        return length

    def __getitem__(self, index):
        data, label = self.sampler(
            index,
            self.sample_square_size,
            self.data,
            self.label,
        )
        data = self.transform(data)  # データ処理
        if label is None:
            return data.astype(np.float32), None  # ひとまずNone
        return data.astype(np.float32), label.astype(np.int64)
