import torch
from torch.utils.data import Dataset
import idx2numpy
import numpy as np

def collate_fn(items: list[dict]) -> dict[torch.Tensor]:
    items = [{
        "image": np.expand_dims(item["image"], axis=0),
        "label": np.array(item["label"])
    } for item in items]

    items = {
        "image": np.stack([item["image"] for item in items], axis=0),
        "label": np.stack([item["label"] for item in items], axis=0)
    }

    # Convert to torch tensors with correct dtypes:
    # - images should be float (normalize to [0,1]) so linear layers receive FloatTensor
    # - labels should be long (int64) for CrossEntropyLoss
    items = {
        "image": torch.tensor(items["image"], dtype=torch.float32) / 255.0,
        "label": torch.tensor(items["label"], dtype=torch.long)
    }

    return items

class Item:
    def __init__(self, image, label):
        self.image = image
        self.label = label

class MnistDataset(Dataset):
    def __init__(self, image_path, label_path):
        images = idx2numpy.convert_from_file(image_path)
        labels = idx2numpy.convert_from_file(label_path)

        self._data = [
            {
                "image": np.array(image),
                "label": label
            } 
            for image, label in zip(images.tolist(), labels.tolist())
        ]

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index: int) -> dict:
        return self._data[index]

    


