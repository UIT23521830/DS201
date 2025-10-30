import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from collections import Counter
import numpy as np

from Perceptron_3_Layer import Perceptron_3_Layer
from mnist_dataset import MnistDataset, collate_fn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")


def evaluate(dataloader: DataLoader, model: nn.Module):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for item in dataloader:
            images = item["image"].to(device)
            labels = item["label"].to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)

    return f1, precision, recall


if __name__ == "__main__":
    train_dataset = MnistDataset(
        image_path="train-images-idx3-ubyte",
        label_path="train-labels-idx1-ubyte"
    )

    test_dataset = MnistDataset(
        image_path="t10k-images-idx3-ubyte",
        label_path="t10k-labels-idx1-ubyte"
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    model = Perceptron_3_Layer(image_size=(28, 28), hidden1=512, hidden2=256, num_labels=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for batch_idx, item in enumerate(train_loader):
            images = item["image"].to(device)
            labels = item["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # if (batch_idx + 1) % 100 == 0:
            #     print(f"Epoch {epoch+1}/{num_epochs} Batch {batch_idx+1} Loss: {running_loss/100:.4f}")
            #     running_loss = 0.0

        # print(f"--- Epoch {epoch+1} ---")
        # evaluate(test_loader, model)

        f1, precision, recall = evaluate(test_loader, model)
        print(f"Epoch {epoch+1}/{num_epochs} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")






