import torch
from torch.utils.data import DataLoader
from torch import nn, optim
import numpy
from sklearn.metrics import f1_score, precision_score, recall_score
from Perceptron_1_Layer import Perceptron_1_Layer
from mnist_dataset import MnistDataset, collate_fn

# Select device: prefer CUDA if available, otherwise CPU.
# On Windows most users won't have MPS (Apple Silicon) so avoid selecting it.
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

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

    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

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

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = Perceptron_1_Layer(
        image_size=(28, 28),
        num_labels=10
    ).to(device)

    loss_fn = nn.NLLLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        

        for item in train_dataloader:
            images = item["image"].to(device)
            labels = item["label"].to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        f1, precision, recall = evaluate(test_dataloader, model)
        print(f"Epoch {epoch+1}/{num_epochs} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
