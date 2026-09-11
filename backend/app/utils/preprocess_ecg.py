"""
Preprocessing pipeline for the ECG (image) branch of MedFusion AI.
Builds a labeled file list from the 4 class folders, splits into
train/test, and provides a PyTorch Dataset + DataLoaders.
"""

import os
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ECG_ROOT = "data/ecg"

# Maps the actual (messy) folder names on disk to clean class labels
CLASS_FOLDERS = {
    "ECG Images of Myocardial Infarction Patients": "MI",
    "ECG Images of Patient that have abnormal heartbeat": "Abnormal_Heartbeat",
    "ECG Images of Patient that have History of MI": "History_of_MI",
    "Normal Person ECG Images": "Normal",
}

CLASS_TO_IDX = {
    "Normal": 0,
    "Abnormal_Heartbeat": 1,
    "History_of_MI": 2,
    "MI": 3,
}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

IMG_SIZE = 224


def build_file_label_list():
    """Scans data/ecg/, matching actual folder names by prefix (since they
    have extra text like '(240x12=2880)' appended)."""
    file_paths = []
    labels = []

    actual_folders = os.listdir(ECG_ROOT)

    for expected_prefix, class_name in CLASS_FOLDERS.items():
        matched_folder = None
        for folder in actual_folders:
            if folder.startswith(expected_prefix):
                matched_folder = folder
                break

        if matched_folder is None:
            raise FileNotFoundError(
                f"Could not find a folder starting with '{expected_prefix}' "
                f"inside {ECG_ROOT}. Found folders: {actual_folders}"
            )

        folder_path = os.path.join(ECG_ROOT, matched_folder)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                file_paths.append(os.path.join(folder_path, fname))
                labels.append(CLASS_TO_IDX[class_name])

    return file_paths, labels


class ECGDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = Image.open(self.file_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_train_test_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, test_transform


def get_dataloaders(batch_size=16, test_size=0.2, random_state=42):
    file_paths, labels = build_file_label_list()

    train_files, test_files, train_labels, test_labels = train_test_split(
        file_paths, labels, test_size=test_size, random_state=random_state, stratify=labels
    )

    train_transform, test_transform = get_train_test_transforms()

    train_dataset = ECGDataset(train_files, train_labels, transform=train_transform)
    test_dataset = ECGDataset(test_files, test_labels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    file_paths, labels = build_file_label_list()
    print(f"Total images found: {len(file_paths)}")
    from collections import Counter
    counts = Counter(labels)
    for idx, count in sorted(counts.items()):
        print(f"  {IDX_TO_CLASS[idx]}: {count}")

