import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torchvision.models import resnet18
from tqdm import tqdm
import os

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "dataset"
MODEL_SAVE_PATH = "ambulance_resnet.pth"

BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0001
VAL_SPLIT = 0.2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using Device:", DEVICE)

# -----------------------------
# TRANSFORMS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3),
    transforms.ToTensor(),
])

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset = ImageFolder(DATASET_PATH, transform=transform)

print("Classes Found:", dataset.classes)

# -----------------------------
# TRAIN / VAL SPLIT
# -----------------------------
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = resnet18(weights="IMAGENET1K_V1")

model.fc = torch.nn.Linear(model.fc.in_features, len(dataset.classes))
model = model.to(DEVICE)

# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0

# -----------------------------
# TRAIN LOOP
# -----------------------------
for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---------- TRAIN ----------
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader):

        images, labels = images.to(DEVICE), labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total * 100

    # ---------- VALIDATION ----------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total * 100

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_acc:.2f}%")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # ---------- SAVE BEST MODEL ----------
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print("✅ Best Model Saved")

print("\nTraining Finished!")
