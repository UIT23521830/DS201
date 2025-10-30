import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np


def collate_fn(items: list[dict]) -> dict[torch.Tensor]:
    # Gom các item (ảnh, nhãn) thành batch
    images = np.stack([np.expand_dims(item["image"], axis=0) for item in items], axis=0)
    labels = np.stack([item["label"] for item in items], axis=0)

    # Chuyển sang tensor
    images = torch.tensor(images, dtype=torch.float32) / 255.0  # chuẩn hóa [0,1]
    labels = torch.tensor(labels, dtype=torch.long)

    return {"image": images, "label": labels}


class MnistDataset(Dataset):
    def __init__(self, image_path, label_path):
        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)

        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        return {"image": self.images[index], "label": int(self.labels[index])}
