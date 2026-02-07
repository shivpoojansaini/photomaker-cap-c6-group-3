# face_utils.py

import numpy as np
import torch
from photomaker import FaceAnalysis2, analyze_faces


def load_face_detector(device):
    print("Loading face detector...")
    providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]

    detector = FaceAnalysis2(
        providers=providers,
        allowed_modules=["detection", "recognition"]
    )
    detector.prepare(ctx_id=0, det_size=(640, 640))
    return detector


def extract_left_right_embeddings(face_detector, pil_image):
    img_array = np.array(pil_image)[:, :, ::-1]

    faces = analyze_faces(face_detector, img_array)
    if len(faces) < 2:
        raise ValueError("Need at least two faces in the input image.")

    faces_sorted = sorted(faces, key=lambda f: f["bbox"][0])
    left_face, right_face = faces_sorted[:2]

    emb_left = torch.from_numpy(left_face["embedding"]).float()
    emb_right = torch.from_numpy(right_face["embedding"]).float()

    return torch.stack([emb_left]), torch.stack([emb_right])
