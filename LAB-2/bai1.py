import torch
import numpy as np
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import f1_score, precision_score, recall_score

from model.lenet import LeNet
from data_utils.mnist_dataset import MnistDataset, collate_fn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(dataloader: DataLoader, model: nn.Module) -> dict:
    model.eval()
    predictions = []
    true_labels = []
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
    train_dataset = MnistDataset(
        image_path="dataset/mnist/train-images-idx3-ubyte",
        label_path="dataset/mnist/train-labels-idx1-ubyte"
    )

    test_dataset = MnistDataset(
        image_path="dataset/mnist/t10k-images-idx3-ubyte",
        label_path="dataset/mnist/t10k-labels-idx1-ubyte"
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = LeNet().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10
    best_score = 0.0
    best_score_name = "f1"
    for epoch in range(num_epochs):
        losses = []
        model.train()
        for batch in train_dataloader:
            images: torch.Tensor = batch["image"].to(device)
            labels: torch.Tensor = batch["label"].to(device)

            output = model(images)
            loss = criterion(output, labels)

            optimizer.zero_grad()
            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        metrics = evaluate(test_dataloader, model)
        print(f"Epoch {epoch+1}/{num_epochs}, F1: {metrics['f1']:.4f}, Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")    