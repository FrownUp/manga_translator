import numpy as np
import cv2
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from skimage.metrics import structural_similarity as ssim
import torch
from torchvision import transforms
import pickle
import os
import time

class CharacterMapper:
    """
    Maps detected faces or bodies to consistent character IDs
    with caching, embeddings, similarity matching, and persistence.
    """

    def __init__(self, embedding_model=None, device="cpu", persistence_file=None):
        self.cluster_to_speaker = {}          # Local cluster → speaker ID
        self.global_cluster_registry = {"face": {}, "body": {}}  # Global cluster → features list
        self.cached_crops = {"face": {}, "body": {}}             # Store crops for visual SSIM matching
        self.cluster_to_global_id = {}        # Local cluster → global cluster ID
        self.next_speaker_id = 1
        self.next_global_id = {"face": 1, "body": 1}

        self.embedding_model = embedding_model
        self.device = device
        self.persistence_file = persistence_file

        # Transform for face embeddings
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Load from persistence if available
        if persistence_file and os.path.exists(persistence_file):
            self._load_state()

    # -----------------------------
    # Face Embedding Extraction
    # -----------------------------
    def get_face_embedding(self, face_crop: np.ndarray) -> np.ndarray:
        """Generate embedding from face crop using model"""
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.embedding_model(input_tensor).cpu().numpy()
        return embedding.flatten()

    # -----------------------------
    # Speaker Assignment
    # -----------------------------
    def get_speaker(self, cluster_type: str, cluster_id: int) -> str:
        key = f"{cluster_type}_{cluster_id}"
        if key not in self.cluster_to_speaker:
            global_id = self.cluster_to_global_id.get(key)
            speaker = f"CHAR{global_id}" if global_id else f"CHAR{self.next_speaker_id}"
            self.cluster_to_speaker[key] = speaker
            if not global_id:
                self.next_speaker_id += 1
        return self.cluster_to_speaker[key]

    # -----------------------------
    # SSIM Cache Matching
    # -----------------------------
    def _crop_matches_cached(self, new_crop_gray: np.ndarray, cluster_id: int,
                             cluster_type: str, threshold: float = 0.8) -> bool:
        for cached_crop in self.cached_crops[cluster_type].get(cluster_id, []):
            if cached_crop.shape != new_crop_gray.shape:
                cached_crop = cv2.resize(cached_crop, (new_crop_gray.shape[1], new_crop_gray.shape[0]))
            score = ssim(new_crop_gray, cached_crop)
            if score > threshold:
                return True
        return False

    # -----------------------------
    # Global Cluster Matching
    # -----------------------------
    def match_to_global_cluster(self, crop: np.ndarray, cluster_type: str = "face",
                                threshold: float = 0.7, min_confidence: float = 0.85) -> int:
        start_time = time.time()
        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Extract features
        if cluster_type == "face" and self.embedding_model:
            features = self.get_face_embedding(crop)
        else:
            features = np.zeros(128)  # fallback dummy features

        best_match_id = None
        best_similarity = -1

        # Compare to global clusters
        for gid, embeddings in self.global_cluster_registry[cluster_type].items():
            for stored_feat in embeddings:
                sim = cosine_similarity([features], [stored_feat])[0][0]
                if sim > best_similarity:
                    best_similarity = sim
                    best_match_id = gid

        # Evaluate match
        if best_similarity >= min_confidence:
            pass  # strong match, keep best_match_id
        elif best_similarity >= threshold:
            # Check visual similarity if moderate match
            if self._crop_matches_cached(crop_gray, best_match_id, cluster_type):
                pass
            else:
                best_match_id = None
        else:
            best_match_id = None

        # Create new global cluster if no match
        if best_match_id is None:
            best_match_id = self.next_global_id[cluster_type]
            self.global_cluster_registry[cluster_type][best_match_id] = []
            self.cached_crops[cluster_type][best_match_id] = []
            self.next_global_id[cluster_type] += 1

        # Update cluster registry & cache
        self.global_cluster_registry[cluster_type][best_match_id].append(features)
        self.cached_crops[cluster_type][best_match_id].append(crop_gray)

        # Optional: persist state
        if self.persistence_file:
            self._save_state()

        elapsed = time.time() - start_time
        # print(f"[CharacterMapper] match_to_global_cluster: {elapsed:.3f}s")

        return best_match_id

    # -----------------------------
    # Persistence
    # -----------------------------
    def _save_state(self):
        try:
            with open(self.persistence_file, "wb") as f:
                pickle.dump({
                    "cluster_to_speaker": self.cluster_to_speaker,
                    "global_cluster_registry": self.global_cluster_registry,
                    "cached_crops": self.cached_crops,
                    "cluster_to_global_id": self.cluster_to_global_id,
                    "next_speaker_id": self.next_speaker_id,
                    "next_global_id": self.next_global_id
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"[CharacterMapper] Persistence save failed: {e}")

    def _load_state(self):
        try:
            with open(self.persistence_file, "rb") as f:
                state = pickle.load(f)
                self.cluster_to_speaker = state["cluster_to_speaker"]
                self.global_cluster_registry = state["global_cluster_registry"]
                self.cached_crops = state["cached_crops"]
                self.cluster_to_global_id = state["cluster_to_global_id"]
                self.next_speaker_id = state["next_speaker_id"]
                self.next_global_id = state["next_global_id"]
        except Exception as e:
            print(f"[CharacterMapper] Persistence load failed: {e}")
