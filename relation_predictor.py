# relation_predictor.py (unchanged)
import torch
import numpy as np
from trainsgg2 import (
    calculate_iou,
    extract_geometric_features,
    RelationClassifier,
)

import torch

def predict_relations(texts, characters, width, height, classifier_model, text_feats, char_feats):
    """
    Predict relations between text and characters.

    Args:
        texts: List of text detections (with bbox info).
        characters: List of character bboxes.
        width, height: Image dimensions.
        classifier_model: The RelationClassifier model.
        text_feats: handcrafted features for texts (list of lists, each with 4 dims).
        char_feats: handcrafted features for characters (list of lists, each with 4 dims).

    Returns:
        List of dicts with relation predictions.
    """

    # Build combined features for each text-character pair (example: 10-dim vector)
    features = []
    pairs = []
    for t_idx, t in enumerate(texts):
        for c_idx, c in enumerate(characters):
            # Example handcrafted features: concatenate text feats and char feats (4+4=8 dims),
            # plus 2 more features (e.g. distances or IoU) to get 10 dims total.

            # Here just an example of how you might build those:
            tf = text_feats[t_idx]    # e.g. [center_x, center_y, w, h]
            cf = char_feats[c_idx]    # e.g. [center_x, center_y, w, h]

            dist_x = abs(tf[0] - cf[0])
            dist_y = abs(tf[1] - cf[1])
            feature_vec = tf + cf + [dist_x, dist_y]  # length = 4 + 4 + 2 = 10

            features.append(feature_vec)
            pairs.append((t_idx, c_idx))

    feature_tensor = torch.tensor(features, dtype=torch.float32)

    with torch.no_grad():
        outputs = classifier_model(feature_tensor).squeeze(-1)  # shape: [num_pairs]

    predictions = []
    for i, score in enumerate(outputs):
        predictions.append({
            "text_index": pairs[i][0],
            "character_index": pairs[i][1],
            "score": score.item(),
            "label": 1 if score.item() > 0.5 else 0
        })

    return predictions



def extract_geometric_features(text_bbox, char_bbox, img_w, img_h):
    # Normalize coordinates
    def norm(val, dim):
        return val / dim

    # Text center
    tx1, ty1, tx2, ty2 = text_bbox
    t_cx = norm((tx1 + tx2) / 2, img_w)
    t_cy = norm((ty1 + ty2) / 2, img_h)

    # Character center
    cx1, cy1, cx2, cy2 = char_bbox
    c_cx = norm((cx1 + cx2) / 2, img_w)
    c_cy = norm((cy1 + cy2) / 2, img_h)

    # Distance features
    dx = abs(t_cx - c_cx)
    dy = abs(t_cy - c_cy)

    # Size features
    t_area = norm((tx2 - tx1) * (ty2 - ty1), img_w * img_h)
    c_area = norm((cx2 - cx1) * (cy2 - cy1), img_w * img_h)

    return [t_cx, t_cy, c_cx, c_cy, dx, dy, t_area, c_area]

def convert_box(box):
    """Convert [x1,y1,x2,y2] to [x_center, y_center, width, height]"""
    x1, y1, x2, y2 = box
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [x_center, y_center, w, h]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0