from mnist_dataset import MnistDataset, collate_fn

from torch.utils.data import DataLoader

train_dataset = MnistDataset(
    image_path="train-labels-idx1-ubyte",
    label_path="train-labels-idx1-ubyte"
)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=collate_fn
)

for item in train_dataloader:
    print(item["image"].shape)
    print(item["label"].shape)
    break   