import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_recognition import Face
import numpy as np
import cv2
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3)

# initailize Face using face image as np.ndarray
embedding = np.empty((512))
face_image = cv2.imread("photos/photo1.jpg")
f = Face(embedding, face_image)
axes[0].imshow(f.face_image)
axes[0].set_title("Face initialized with image array")

# initialize Face using original image path and bounding box
embedding2 = np.empty((512))
bbox = (50, 50, 300, 300)  # Example bounding box of face
f2 = Face.from_original_image(embedding2, "photos/photo1.jpg", bbox)
axes[1].imshow(f2.face_image)
axes[1].set_title("Face initialized from original image and bbox")

# initialize Face using original image as np.ndarray and bounding box
embedding3 = np.empty((512))
bbox = (50, 50, 300, 300)  # Example bounding box of face
face_image_np3 = cv2.imread("photos/photo1.jpg")
f3 = Face.from_original_image(embedding3, face_image_np3, bbox)
axes[2].imshow(f3.face_image)
axes[2].set_title("Face initialized from image array and bbox")

for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.show() 