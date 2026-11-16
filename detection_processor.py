import numpy as np
from typing import List, Dict, Tuple, Any

# -----------------------------------------
# YOLO → MAGI conversion
# -----------------------------------------
def convert_yolo_to_magi_format(
    yolo_results: Dict[str, Any],
    image_size: Tuple[int, int]
) -> Dict[str, List[List[float]]]:
    """
    Convert YOLOv8 Results objects to MAGI format with pixel coordinates.
    Supports panels, texts (bubbles), and characters.
    """
    magi_detections = {
        "panels": [],
        "texts": [],
        "characters": []
    }

    height, width = image_size

    # Panels
    for box in yolo_results.get("panels", []):
        cls_id = int(box.cls)  # Ensure correct class access
        if cls_id == 0:
            bbox = box.xyxyn[0].tolist()
            magi_detections["panels"].append([
                bbox[0] * width,
                bbox[1] * height,
                bbox[2] * width,
                bbox[3] * height
            ])

    # Bubbles / characters
    for box in yolo_results.get("bubbles", []):
        cls_id = int(box.cls)
        bbox = box.xyxyn[0].tolist()
        pixel_bbox = [
            bbox[0] * width,
            bbox[1] * height,
            bbox[2] * width,
            bbox[3] * height
        ]

        if cls_id == 0:
            magi_detections["texts"].append(pixel_bbox)
        elif cls_id == 1:
            magi_detections["characters"].append(pixel_bbox)

    return magi_detections

# -----------------------------------------
# Panel Sorting
# -----------------------------------------
def sort_panels(
    panels: List[List[float]], 
    reading_order: str = "manga"
) -> List[int]:
    """
    Sort panels based on reading order.
    Manga: right-to-left, Western: left-to-right
    """
    centers = [((x1+x2)/2, (y1+y2)/2) for x1, y1, x2, y2 in panels]

    if reading_order == "manga":
        sorted_indices = sorted(
            range(len(centers)),
            key=lambda i: (centers[i][1], -centers[i][0])
        )
    else:
        sorted_indices = sorted(
            range(len(centers)),
            key=lambda i: (centers[i][1], centers[i][0])
        )
    return sorted_indices

# -----------------------------------------
# Text Sorting within a Panel
# -----------------------------------------
def sort_text_in_panel(
    text_bboxes: List[List[float]], 
    panel_bbox: List[float],
    reading_order: str = "manga"
) -> List[int]:
    """
    Sort text boxes inside a panel using column-based and vertical ordering.
    """
    x1_p, y1_p, x2_p, y2_p = panel_bbox
    panel_width = x2_p - x1_p

    # Filter boxes inside panel
    inside_panel = []
    for i, bbox in enumerate(text_bboxes):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if x1_p <= cx <= x2_p and y1_p <= cy <= y2_p:
            inside_panel.append((i, cx, cy))

    # Column-based sorting
    column_width = max(panel_width / 3, 1)  # Avoid division by zero

    def sort_key(item):
        idx, cx, cy = item
        col = int((cx - x1_p) / column_width)
        return (col, cy) if reading_order == "manga" else (col, cy)

    sorted_indices = sorted(inside_panel, key=sort_key)
    return [idx for idx, _, _ in sorted_indices]
