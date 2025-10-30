import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score

from model.ResNet import ResNet18       
from data_utils.vinafood_dataset import VinaFood, collate_fn

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
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )

    num_classes = 21
    model = ResNet18(num_classes=num_classes, input_channels=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
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

        eval_metrics = evaluate(test_dataloader, model)
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Loss: {running_loss/len(train_dataloader):.4f}, "
              f"F1: {eval_metrics['f1']:.4f}, "
              f"Precision: {eval_metrics['precision']:.4f}, "
              f"Recall: {eval_metrics['recall']:.4f}")
        
