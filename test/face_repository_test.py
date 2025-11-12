import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_recognition import FaceRepository
import matplotlib.pyplot as plt
import cv2

fr = FaceRepository()
# fr.add_images(0, ["../photos/photo1.jpg"])
# fr.add_images(1, ["../photos/trump_test.jpg"])
# print(fr.get_faces_in_clip(0))
# print(fr.get_faces_in_clip(1))

fr.add_images(0, [cv2.imread("../photos/photo1.jpg")])
print(fr.get_faces_in_clip(0))

fr.add_images(2, [
    "../photos/image.png", 
    "../photos/left.png", 
    "../photos/photo1.jpg",
    "../photos/right.png", 
    "../photos/trump_test.jpg", 
    "../photos/trump_test2.jpg", 
    "../photos/trump_test3.jpg", 
    "../photos/trump.jpg", 

])
print(fr.get_faces_in_clip(2))

face_images0 = fr.get_face_images_in_clip(2)
fig, axes = plt.subplots(1, len(face_images0))
for i, img in enumerate(face_images0):
    axes[i].imshow(img)
    axes[i].axis("off")
plt.suptitle("Face images in clip 2")
plt.show()