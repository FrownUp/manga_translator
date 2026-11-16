# manga_pipeline.py
import os
from model_paths import MODEL_DIR
import cv2
from detection_processor import *
from association_engine import *
from ocr_processor import *
from transcript_generator import *
from ocr_processor import DonutOCR
import torch
import joblib
from ultralytics import YOLO
from character_mapper import CharacterMapper
from relation_predictor import predict_relations
import torch.nn as nn
from sklearn.cluster import DBSCAN
from face_processor import FaceProcessor
from collections import defaultdict
import numpy as np
from PIL import Image

# Load MLP model with explicit architecture
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


class EmbeddingSGGModel(nn.Module):
    def __init__(self, input_dim=4, embed_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, text_feats, char_feats):
        text_embed = self.encoder(text_feats)
        char_embed = self.encoder(char_feats)
        return text_embed, char_embed

class FaceEmbeddingNet(nn.Module):
    def __init__(self, embedding_size=512):
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'efficientnet_b1', pretrained=False)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.embedding = nn.Linear(in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
        
    def forward(self, x):
        features = self.backbone(x)
        embeddings = self.bn(self.embedding(features))
        return embeddings


def cluster_characters(character_embeddings, eps=0.5, min_samples=1):
    if len(character_embeddings) == 0:
        return {}
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(character_embeddings)
    cluster_map = {i: int(label) for i, label in enumerate(clustering.labels_)}
    return cluster_map

def group_characters_by_cluster(cluster_map):
    groups = defaultdict(list)
    for char_idx, cluster_id in cluster_map.items():
        groups[cluster_id].append(char_idx)
    return groups.values()


def print_relations_with_clusters(relations, characters, cluster_map):
    print("\n--- True Positive Relations with Cluster Info ---")
    for rel in relations:
        text_idx = rel['text_index']
        char_idx = rel['character_index']
        cluster_id = cluster_map.get(char_idx, -1)
        char_bbox = characters[char_idx]
        print(f"Text #{text_idx} → Character #{char_idx} | BBox: {char_bbox} | Cluster ID: {cluster_id}")


def run_manga_pipeline(
    image_path: str,
    yolo_model,
    embed_model,
    classifier_model,
    ocr_model,
    output_dir: str,
    reading_order: str = "manga",
    character_mapper: CharacterMapper = None
):
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    yolo_results = yolo_model(image)
    detections = convert_yolo_to_magi_format(yolo_results, (img_height, img_width))

    panel_indices = sort_panels(detections["panels"], reading_order=reading_order)

    def extract_feats(bboxes):
        feats = []
        for x1, y1, x2, y2 in bboxes:
            center_x = (x1 + x2) / 2 / img_width
            center_y = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            feats.append([center_x, center_y, width, height])
        return feats

    text_feats = extract_feats([t["bbox"] for t in detections["texts"]])
    char_feats = extract_feats([c for c in detections["characters"]])

    text_tensor = torch.tensor(text_feats, dtype=torch.float32)
    char_tensor = torch.tensor(char_feats, dtype=torch.float32)

    with torch.no_grad():
        text_embeddings, char_embeddings = embed_model(text_tensor, char_tensor)

    text_embeddings_np = text_embeddings.numpy()
    char_embeddings_np = char_embeddings.numpy()

    char_clusters = cluster_characters(char_embeddings_np)
    char_groups = group_characters_by_cluster(char_clusters)

    detected_faces = [det for det in yolo_results[0].boxes.data.cpu().numpy() if int(det[5]) == 0]  # class 0 = face
    detected_bodies = [det for det in yolo_results[0].boxes.data.cpu().numpy() if int(det[5]) == 1]  # class 1 = body

    face_bboxes = [tuple(map(int, box[:4])) for box in detected_faces]
    body_bboxes = [tuple(map(int, box[:4])) for box in detected_bodies]

    face_embedding_model = FaceEmbeddingNet(embedding_size=512)
    face_embedding_model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "face_embedding_model.pth")
    )["model_state_dict"])
    face_embedding_model = face_embedding_model.to(device).eval()
    
    # Update character mapper with face embedding model
    character_mapper.embedding_model = face_embedding_model
    character_mapper.device = device

    # Get face embeddings and aligned faces
    face_embeddings = []
    aligned_faces = []
    for face_bbox in face_bboxes:
        face_crop = crop_image(image, face_bbox)
        aligned_face = cv2.resize(face_crop, (160, 160))
        aligned_faces.append(aligned_face)
        
        # Get embedding
        face_pil = Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
        input_tensor = character_mapper.transform(face_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = face_embedding_model(input_tensor).cpu().numpy()
        face_embeddings.append(embedding.flatten())

    face_embeddings, aligned_faces = face_processor.get_face_embeddings(image, face_bboxes)


    merged_clusters = {}
    face_embedding_map = {face_bbox: emb for face_bbox, emb in zip(face_bboxes, face_embeddings)}

    # Only process groups if we have character detections
    if detections["characters"]:
        for group in char_groups:
            found_global_id = None
            best_match_face = None
            best_match_body = None
            best_match = None
            
            # Try to find a face match for any character in the group
            for char_idx in group:
                char_bbox = detections["characters"][char_idx]
                for face_idx, face_bbox in enumerate(face_bboxes):
                    iou = calculate_iou(char_bbox, face_bbox)
                    if iou > 0.3:
                        best_match_face = face_idx
                        best_match = face_bbox
                        break
                if best_match_face is not None:
                    break
            
            # If we found a face match, get global ID
            if best_match_face is not None:
                # Use the aligned face for recognition
                aligned_face = aligned_faces[best_match_face]
                found_global_id = character_mapper.match_to_global_cluster(
                    face_embeddings[best_match_face],
                    crop=aligned_face,
                    cluster_type="face",
                    min_confidence=0.85
                )
            else:
                # Try to find a body match for any character in the group
                for char_idx in group:
                    char_bbox = detections["characters"][char_idx]
                    for body_idx, body_bbox in enumerate(body_bboxes):
                        iou = calculate_iou(char_bbox, body_bbox)
                        if iou > 0.3:
                            best_match_body = body_idx
                            break
                    if best_match_body is not None:
                        break
                
                if best_match_body is not None:
                    body_crop = crop_image(image, body_bboxes[best_match_body])
                    found_global_id = character_mapper.match_to_global_cluster(
                        features=np.zeros(128),  # Placeholder
                        crop=body_crop,
                        cluster_type="body"
                    )
            
            # If no face/body found, use character crop for the whole group
            if found_global_id is None:
                char_bbox = detections["characters"][group[0]]
                char_crop = crop_image(image, char_bbox)
                found_global_id = character_mapper.match_to_global_cluster(
                    features=np.zeros(128),  # Placeholder
                    crop=char_crop,
                    cluster_type="body"
                )
            
            # Assign the same global ID to all characters in the group
            for char_idx in group:
                merged_clusters[char_idx] = found_global_id

    relations = predict_relations(
        texts=detections["texts"],
        characters=detections["characters"],
        width=img_width,
        height=img_height,
        classifier_model=classifier_model,
        text_embeddings=text_embeddings_np,
        char_embeddings=char_embeddings_np
    )

    true_positives = [r for r in relations if r["label"] == 1]

    associations = associate_text_to_characters(
        texts=detections["texts"],
        characters=detections["characters"],
        relations=true_positives
    )

    print_relations_with_clusters(true_positives, detections["characters"], char_clusters)

    text_crops = [crop_image(image, t["bbox"]) for t in detections["texts"]]
    ocr_results = batch_ocr(text_crops, ocr_model)

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    transcript = generate_transcript(
        ocr_results,
        associations,
        merged_clusters,  # Use merged_clusters with global IDs
        char_embeddings_np,
        os.path.join(output_dir, f"{image_name}_transcript.txt"),
        character_mapper=character_mapper
    )

    visualize_results(
        image,
        detections,
        associations,
        os.path.join(output_dir, f"{image_name}_visualized.jpg")
    )

    return transcript


if __name__ == "__main__":
    yolo_bubble_char = YOLO(os.path.join(MODEL_DIR, "bubble_best.pt"))
    yolo_panel = YOLO(os.path.join(MODEL_DIR, "panel_best.pt"))
    face_processor = FaceProcessor(
        face_model_path=os.path.join(MODEL_DIR, "face_best.pt"),
        embedding_model_path=os.path.join(MODEL_DIR, "manga_embedding_model2.pth"),
        device="cuda"
    )

    classifier_model = RelationClassifier(input_dim=10)
    classifier_model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "relation_classifier3.pth")))
    classifier_model.eval()

    embed_model = EmbeddingSGGModel(input_dim=4, embed_dim=32)
    embed_model.load_state_dict(torch.load(EMBED_MODEL_PATH))
    embed_model.eval()

    ocr_model = DonutOCR(
        model_name="naver-clova-ix/donut-small",
        device="cuda",
        fp16=True
    )

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize character mapper for persistent recognition
    face_embedding_model = FaceEmbeddingNet(embedding_size=512)
    face_embedding_model.load_state_dict(torch.load(
        os.path.join(MODEL_DIR, "face_embedding_model.pth")
    )["model_state_dict"])
    face_embedding_model = face_embedding_model.to("cuda").eval()
    
    character_mapper = CharacterMapper(
        embedding_model=face_embedding_model,
        device="cuda"
    )
    character_mapper.load_cache("character_cache.pkl")
    
    # Process all images with the same mapper for cross-image recognition
    image_paths = ["page1.jpg", "page2.jpg", "page3.jpg"]  # Add your image paths
    for image_path in image_paths:
        transcript = run_manga_pipeline(
            image_path,
            yolo_bubble_char,
            embed_model,
            classifier_model,
            ocr_model,
            output_dir,
            reading_order="manga",
            character_mapper=character_mapper
        )
        print(f"Transcript for {image_path}:\n{transcript}\n")