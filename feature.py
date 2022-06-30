from typing import Tuple
import pandas as pd
import numpy as np
from pywt import cwt
from tqdm import tqdm
import matplotlib.pyplot as plt


def __loadData(filename: str) -> pd.DataFrame:
    data: pd.DataFrame = pd.read_csv(filename, dtype={
        "src_ip_count": int,
        "dst_ip_count": float,
        "label": int
    })
    return data

def __cwtFeature(data: np.ndarray, wavelet_func: str) -> np.ndarray:
    coef, _ = cwt(data, np.arange(1, data.shape[0]+1), wavelet_func)
    return coef

def extractFeatureWithLabel(filename: str, wavelet_func: str, feature_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    data = __loadData(filename)
    src_ip_count: np.ndarray = data["src_ip_count"].to_numpy()
    dst_ip_count: np.ndarray = data["dst_ip_count"].to_numpy()
    labels: np.ndarray = data["label"].to_numpy()
    labels = labels[feature_size-1:,]
    assert src_ip_count.shape[0] == dst_ip_count.shape[0]
    size = src_ip_count.shape[0]
    features = []
    for i in tqdm(range(size - feature_size + 1), desc="Featuring"):
        data_slice = src_ip_count[i:i+feature_size,]
        feature_1 = __cwtFeature(data_slice, wavelet_func)
        data_slice = dst_ip_count[i:i+feature_size,]
        feature_2 = __cwtFeature(data_slice, wavelet_func)
        feature = np.array([feature_1, feature_2])
        features.append(feature)
    features = np.array(features)
    return (features, labels)

def gui():
    fig = plt.figure()
    features, labels = extractFeatureWithLabel("data/record_07_28.csv", "gaus1")
    size = features.shape[0]
    should0 = True
    should1 = True
    for i in range(size):
        if not (should0 or should1):
            break
        feature = features[i]
        mata = feature[0]
        matb = feature[1]
        if labels[i] == 0 and should0:
            # plt.clf()
            ax1 = fig.add_subplot(2,2,1)
            ax1.set_title("Normal src")
            ax1.matshow(mata)
            # plt.matshow(mata)
            # plt.savefig("img/normal_src.png")
            # plt.clf()
            ax2 = fig.add_subplot(2,2,2)
            ax2.set_title("Normal dst")
            ax2.matshow(matb)
            # plt.savefig("img/normal_dst.png")
            should0 = False
        elif labels[i] == 1 and should1:
            # plt.clf()
            ax1 = fig.add_subplot(2,2,3)
            ax1.set_title("Attack src")
            ax1.matshow(mata)
            # plt.savefig("img/attack_src.png")
            # plt.clf()
            ax2 = fig.add_subplot(2,2,4)
            ax2.set_title("Attack dst")
            ax2.matshow(matb)
            # plt.savefig("img/attack_dst.png")
            should1 = False
    plt.show()
    # plt.savefig("img/feature.png")

if __name__ == "__main__":
    gui()
