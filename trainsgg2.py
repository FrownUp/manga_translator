import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
from collections import defaultdict

# --- Step 1: Load JSONL data ---
def load_data(jsonl_path):
    data = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

# --- IoU function ---
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    if union == 0:
        return 0
    return intersection / union

# --- Feature extraction ---
def extract_geometric_features(text_bbox, char_bbox, img_width, img_height):
    # Normalize bounding boxes: each bbox is [x_min, y_min, x_max, y_max]
    def normalize_bbox(bbox):
        return [bbox[0]/img_width, bbox[1]/img_height, bbox[2]/img_width, bbox[3]/img_height]

    text_norm = normalize_bbox(text_bbox)
    char_norm = normalize_bbox(char_bbox)

    # Centers
    text_center = [(text_bbox[0]+text_bbox[2])/2/img_width, (text_bbox[1]+text_bbox[3])/2/img_height]
    char_center = [(char_bbox[0]+char_bbox[2])/2/img_width, (char_bbox[1]+char_bbox[3])/2/img_height]

    offset = [char_center[0] - text_center[0], char_center[1] - text_center[1]]

    features = text_norm + char_norm + offset
    return features

# --- Dataset class ---
class RelationDataset(Dataset):
    def __init__(self, relations):
        self.relations = relations

    def __len__(self):
        return len(self.relations)

    def __getitem__(self, idx):
        feature = torch.tensor(self.relations[idx]["features"], dtype=torch.float32)
        label = torch.tensor(self.relations[idx]["label"], dtype=torch.float32)
        return feature, label

# --- Model ---
class RelationClassifier(nn.Module):
    def __init__(self, input_dim=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

# --- Helper for accuracy ---
def calculate_accuracy(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='binary', zero_division=0)
    return correct / total, precision, recall, f1

# --- Main function to build relation dataset ---
def build_relation_dataset(data, num_negatives=5, iou_threshold=0.3):
    relations = []

    for item in data:
        sg = item["scenegraph"]
        texts = sg.get("texts", [])
        chars = sg.get("characters", [])
        tails = sg.get("tails", [])
        text_char_assocs = sg.get("text_character_associations", [])
        text_tail_assocs = sg.get("text_tail_associations", [])

        img_width = item.get("width", 1000)  # fallback width if no info
        img_height = item.get("height", 1000) # fallback height if no info

        positive_pairs = set()

        # Direct associations
        for text_idx, char_idx in text_char_assocs:
            text_bbox = texts[text_idx]
            char_bbox = chars[char_idx]
            features = extract_geometric_features(text_bbox, char_bbox, img_width, img_height)
            relations.append({"features": features, "label": 1})
            positive_pairs.add((text_idx, char_idx))

        # Tail-mediated associations
        for tail_idx, tail_bbox in enumerate(tails):
            for char_idx, char_bbox in enumerate(chars):
                if calculate_iou(tail_bbox, char_bbox) > iou_threshold:
                    # Find all text indices linked to this tail
                    for text_idx, tail_assoc in text_tail_assocs:
                        if tail_assoc == tail_idx:
                            features = extract_geometric_features(texts[text_idx], char_bbox, img_width, img_height)
                            relations.append({"features": features, "label": 1})
                            positive_pairs.add((text_idx, char_idx))

        # Negative samples: random pairs not in positive_pairs
        texts_len = len(texts)
        chars_len = len(chars)
        for _ in range(num_negatives * (len(positive_pairs) or 1)):
            text_idx = random.randint(0, texts_len - 1) if texts_len > 0 else 0
            char_idx = random.randint(0, chars_len - 1) if chars_len > 0 else 0
            if (text_idx, char_idx) not in positive_pairs and texts_len > 0 and chars_len > 0:
                features = extract_geometric_features(texts[text_idx], chars[char_idx], img_width, img_height)
                relations.append({"features": features, "label": 0})

    return relations

# --- Train function ---
def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze()
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        val_acc, val_prec, val_rec, val_f1 = calculate_accuracy(model, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f} - Val Acc: {val_acc:.4f} - Precision: {val_prec:.4f} - Recall: {val_rec:.4f} - F1: {val_f1:.4f}")

# --- Main execution ---
def main():
    random.seed(42)
    torch.manual_seed(42)

    data = load_data("magiv3_scenegraph_outputs.jsonl")
    print(f"Loaded {len(data)} entries")

    relations = build_relation_dataset(data, num_negatives=3, iou_threshold=0.3)
    print(f"Total relation samples (pos + neg): {len(relations)}")

    # Shuffle and split dataset 80/20
    random.shuffle(relations)
    split_idx = int(0.8 * len(relations))
    train_rel = relations[:split_idx]
    val_rel = relations[split_idx:]

    train_dataset = RelationDataset(train_rel)
    val_dataset = RelationDataset(val_rel)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RelationClassifier(input_dim=len(relations[0]["features"])).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train(model, train_loader, val_loader, criterion, optimizer, device, epochs=10)
    torch.save(model.state_dict(), "relation_classifier.pth")
    print("Model saved to relation_classifier.pth")

if __name__ == "__main__":
    main()
