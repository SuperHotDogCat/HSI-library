from typing import Optional
import numpy as np


def retrieve_square_images(
    n: int, hsi_image: np.array, ipget: np.array
) -> tuple[np.array, np.array]:
    """
    pineデータセットに合わせて(H, W, C)の入力を想定していますが、今後変化しうる恐れは大いにあります。
    pineデータセット(H, W, C)の入力から、n✕nの大きさの画像を何個も切り出すことで(B, n, n, C)のデータを得ます。
    paddingは今の所なし

    出力: 画像(B, n, n, C), ラベル(B, n, n)
    """
    H, W, C = hsi_image.shape
    data = [0 for _ in range((H - n + 1) * (W - n + 1))]
    labels = [0 for _ in range((H - n + 1) * (W - n + 1))]
    for i in range(n - 1, H):
        for j in range(n - 1, W):
            data_index = (i - n + 1) * (W - n + 1) + (
                j - n + 1
            )  # iとjをそれぞれn-1だけ平行移動して格納すべきデータのindexにしている
            data[data_index] = hsi_image[
                i - n + 1 : i + 1, j - n + 1 : j + 1, :
            ]
            labels[data_index] = ipget[i - n + 1 : i + 1, j - n + 1 : j + 1]
    return np.array(data), np.array(labels)


def sampling_square_image(
    index: int,
    n: int,
    hsi_image: np.array,
    label: Optional[np.array],
) -> tuple[np.array, Optional[np.array]]:
    # retrieve_square_imagesの二重ループでやっている処理をDataset Classの__getitem__で呼び出し, 1枚1枚切り出すように変更した関数
    H, W, C = hsi_image.shape
    i = index // (
        W - n + 1
    )  # indexは0<=index<=(W - n + 1)(H - n)の値を取るので0<=i<=(H - n)
    j = index % (W - n + 1)  # 0<=j<=(W-n)
    if label is None:
        hsi_image[i : i + n, j : j + n, :], None
    return (
        hsi_image[i : i + n, j : j + n, :],  # n×nの画像を切り出す
        label[i : i + n, j : j + n],
    )
