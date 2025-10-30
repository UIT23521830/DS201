import torch
from torch.utils.data import Dataset

import os
import cv2 as cv
from typing import List, Dict, Any, Tuple

def collate_fn(samples: list[dict]) -> torch.Tensor:
    images = [sample["image"] for sample in samples]
    labels = [sample["label"] for sample in samples]

    # images = torch.cat(images,dim=0)
    # labels = torch.tensor(labels)
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)

    return {
        "image": images,
        "label": labels
    }

class VinaFood(Dataset):
    def __init__(self, path: str, image_size: Tuple[int,int]=(224,224)):
        super().__init__()

        self.path = path
        self.image_size = image_size
        
        self.label2idx: Dict[str,int] = {}
        self.idx2label: Dict[str,int] = {}

        self.data: List[Dict[str,Any]] = self.load_data(path)

    def load_data(self, path):
        data = []
        label_id = 0
        for folder in os.listdir(path):
            label = folder
            if label not in self.label2idx:
                self.label2idx[label] = label_id
                label_id += 1
            for image_file in os.listdir(os.path.join(path, folder)):
                image_path = os.path.join(path, folder, image_file)
                image = cv.imread(image_path)          
                if image is None:
                    continue

                data.append({
                    "image": image,
                    "label": label
                })

        self.idx2label = {id: label for label, id in self.label2idx.items()}

        return data

    def __len__(self):
        return len(self.data) 

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        image = item["image"]
        label = item["label"]

        image = cv.resize(image, self.image_size)
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(-1,0,1)
        
        return {
            "image": image,
            "label": self.label2idx[label]
        }
