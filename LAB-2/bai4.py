import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import ResNetForImageClassification

from data_utils.vinafood_dataset import VinaFood, collate_fn
from model.pretrained_resnet import PretrainedResnet  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for item in dataloader:
            image: torch.Tensor = item["image"].to(device)
            label: torch.Tensor = item["label"].to(device)

            output: torch.Tensor = model(image)
            output = output.argmax(dim=-1)

            predictions.extend(output.cpu().numpy().tolist())
            true_labels.extend(label.cpu().numpy().tolist())

    return {
        "f1": f1_score(true_labels, predictions, average="macro"),
        "precision": precision_score(true_labels, predictions, average="macro"),
        "recall": recall_score(true_labels, predictions, average="macro")
    }


if __name__ == "__main__":
    train_dataset = VinaFood(
        path="dataset/VinaFood21/VinaFood21/train"
    )

    test_dataset = VinaFood(
        path="dataset/VinaFood21/VinaFood21/test"
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = PretrainedResnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for item in train_dataloader:
            image: torch.Tensor = item["image"].to(device)
            label: torch.Tensor = item["label"].to(device)

            optimizer.zero_grad()
            output: torch.Tensor = model(image)
            loss: torch.Tensor = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        metrics = evaluate(test_dataloader, model)
        print(f"Epoch {epoch+1}/{num_epochs} - F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")    

