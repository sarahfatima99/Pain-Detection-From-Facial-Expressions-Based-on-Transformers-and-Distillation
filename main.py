# main.py
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from config.config import *
from teacher import ResNet50Teacher
from student import StudentDeiT
from distillation_loss import DistillationLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])




# ======================
# DATASETS & LOADERS
# ======================
train_ds = ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds   = ImageFolder(VAL_DIR, transform=val_transform)



train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_ds = ImageFolder(TEST_DIR, transform=test_transform)

test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,      
    num_workers=4,
    pin_memory=True
)


print("Class mapping:", train_ds.class_to_idx)


# ======================
# LOAD TEACHER
# ======================
teacher = ResNet50Teacher(NUM_CLASSES)
# teacher.load_state_dict(torch.load(TEACHER_WEIGHTS, map_location=DEVICE))
teacher.to(DEVICE)
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False


# ======================
# LOAD STUDENT
# ======================
student = StudentDeiT().to(DEVICE)
student.load_state_dict(torch.load("model/spatial_student_pain.pth", map_location=DEVICE))

criterion = DistillationLoss()
optimizer = torch.optim.AdamW(student.parameters(), lr=LR)


# ======================
# TRAIN ONE EPOCH
# ======================
def train_epoch():
    student.train() # the function  inherits from torch.nn.Module.
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        with torch.no_grad():
            teacher_logits = teacher(imgs)

        cls_logits, dist_logits = student(imgs)
        loss = criterion(cls_logits, dist_logits, teacher_logits, labels)

        optimizer.zero_grad() # Clears gradients from previous batch.
        loss.backward() # Applies backpropagation. Compute gradiants of loss_cls and loss_dist
        optimizer.step() # Applies AdamW update to student parameters.

        total_loss += loss.item()

    return total_loss / len(train_loader)



def evaluate_metrics(model, dataloader, device):
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)

            # handle student model output
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # cls_logits

            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)

    return {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, average="binary"),
        "Recall": recall_score(all_labels, all_preds, average="binary"),
        "F1": f1_score(all_labels, all_preds, average="binary"),
    }



# ======================
# EVALUATION
# ======================
def evaluate():
    student.eval()
    correct, total = 0, 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            cls_logits, _ = student(imgs)
            preds = cls_logits.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

      

    return correct / total



# ======================
# MAIN TRAIN LOOP
# ======================
def main():
    print(f"Training on device: {DEVICE}")

    for epoch in range(EPOCHS):
        loss = train_epoch()
        acc = evaluate()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {loss:.4f} | Val Acc: {acc:.4f}"
        )

    test_metrics = evaluate_metrics(student, test_loader, DEVICE)

    print(" TEST RESULT")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")


    torch.save(student.state_dict(), STUDENT_WEIGHTS)
    print(f"Student model saved to {STUDENT_WEIGHTS}")


# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    main()
