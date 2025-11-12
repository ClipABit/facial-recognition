import numpy as np
import cv2

class Face:
    def __init__(self, embedding: np.ndarray, face_image: np.ndarray):
        self.embedding = embedding
        self.face_image = face_image

    @classmethod
    def from_original_image(cls, embedding: np.ndarray, orig_image: np.ndarray | str, bbox: tuple[int, int, int, int]):
        if (type(orig_image) == str):
            image_np = cv2.imread(orig_image)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif (type(orig_image) == np.ndarray):
            image_np = orig_image

        x, y, w, h = bbox

        # Crop (OpenCV uses NumPy slicing)
        face_image = image_np[y:y+h, x:x+w]
        return cls(embedding, face_image)