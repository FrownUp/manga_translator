# face_processor.py
import cv2
import torch
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from PIL import Image
import torch.nn as nn

class FaceProcessor:
    def __init__(self, face_model_path, embedding_model_path, device="cuda"):
        self.face_detector = YOLO(face_model_path)
        self.device = device
        
        # Load face embedding model
        self.embedding_model = FaceEmbeddingNet().to(device)
        state_dict = torch.load(embedding_model_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in state_dict:
            self.embedding_model.load_state_dict(state_dict['model_state_dict'])
        else:  # Direct state dict
            self.embedding_model.load_state_dict(state_dict)
            
        self.embedding_model.eval()
        
        # Use same transform as training
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Landmark model for alignment
        self.landmark_model = YOLO("yolov8n-face.pt")  # Example model

    def detect_faces(self, image):
        results = self.face_detector(image)[0]
        face_bboxes = []
        for box in results.boxes:
            if int(box.cls) == 0:  # Face class
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                face_bboxes.append([x1, y1, x2, y2])
        return face_bboxes

    def detect_landmarks(self, face_crop):
        results = self.landmark_model(face_crop)[0]
        landmarks = []
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            for point in results.keypoints.xy[0]:
                landmarks.append(point.tolist())
        return landmarks

    def align_face(self, face_crop, landmarks):
        """Simple face alignment using eye positions"""
        if len(landmarks) < 2:
            return face_crop
            
        # Use first two points as eyes
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        
        # Calculate rotation angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Calculate center point between eyes
        center = ((left_eye[0] + right_eye[0]) // 2, 
                 (left_eye[1] + right_eye[1]) // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned = cv2.warpAffine(face_crop, M, (face_crop.shape[1], face_crop.shape[0]))
        return aligned

    def get_face_embeddings(self, image, face_bboxes):
        embeddings = []
        aligned_faces = []
        for bbox in face_bboxes:
            x1, y1, x2, y2 = [int(coord) for coord in bbox]
            face_crop = image[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
                
            # Detect landmarks and align face
            landmarks = self.detect_landmarks(face_crop)
            if landmarks:
                aligned_face = self.align_face(face_crop, landmarks)
            else:
                aligned_face = face_crop
                
            # Convert to PIL and apply transformations
            face_pil = Image.fromarray(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB))
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.embedding_model(face_tensor).cpu().numpy().flatten()
            
            embeddings.append(embedding)
            aligned_faces.append(aligned_face)
            
        return embeddings, aligned_faces

class MangaEmbeddingNet(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)