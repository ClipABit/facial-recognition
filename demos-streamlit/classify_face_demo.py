import streamlit as st
from PIL import Image
from io import BytesIO
import cv2
import tempfile
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

def detect_and_embed(img, detector_backend="mtcnn", model_name="ArcFace", enforce_detection=True, align=True):
    """
    Input:
        ...
    Return:
        List of dictionaries. Each dictionary represent a face containing the following fields:
        - embedding: the vector embedding of that face
        - facial_area: dictionary with info about face location
        - face_confidence: how confident this is a face
        - img_path: path of the source image
    """
    from deepface import DeepFace
    rep = DeepFace.represent(
        img_path = img,
        model_name = model_name,
        detector_backend = detector_backend,
        # model_name = "ArcFace",
        # detector_backend = "mtcnn",
        enforce_detection = enforce_detection,
        align = align
    )
    for r in rep:
        r["img_path"] = img
    return rep

def cluster(embeddings):
    """
    Input:
        embeddings: list of vector embeddings
    Return:
        A list of integers >= -1. -1 means outlier, each int >=0 represent a cluster that vector belongs to
            e.g. output=[0, 1, 0] indicates embeddings[0] and embeddings[2] belong to same cluster
    """
    clustering = DBSCAN(metric="cosine", eps=0.6, min_samples=2) # eps is the distance between vectors to be grouped together, min_samples is the min number of vectors needed for a cluster
    X = np.stack(embeddings)
    clustering.fit(X)
    return clustering.labels_

def classify_faces(img_lst):
    embeddings = []
    face_loc = []

    for img in img_lst:
        faces = detect_and_embed(img)

        for f in faces:
            embeddings.append(f["embedding"])
            face_loc.append({"img_path": img, "facial_area": f["facial_area"]})

    face_cluster = cluster(embeddings)

    print(face_cluster)
    
    classified_faces = []
    for i in range(len(face_cluster)):
        f = face_cluster[i]
        if f == -1:    # the face is an outlier: ignore
            continue
        if f >= len(classified_faces):
            classified_faces.append([face_loc[i]])
        else:
            classified_faces[f].append(face_loc[i])
    return classified_faces

def group_images(images):
    # print(images)
    grouped_images = []
    fc = classify_faces(images)
    for group in fc:
        g = []
        for image in group:
            region = image["facial_area"]
            image["img_path"] = image["img_path"].copy()
            cv2.rectangle(image["img_path"], (region['x'], region['y']),
                      (region['x'] + region['w'], region['y'] + region['h']),
                      (255, 0, 0), 2)
            g.append(image["img_path"])
        grouped_images.append(g)
    return grouped_images


# Streamlit app
st.title("Facial recognition")

uploaded_files = st.file_uploader(
    "Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if uploaded_files:
    # print("uploaded files:", uploaded_files)
    images = [Image.open(file) for file in uploaded_files]
    opencv_images = [cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR) for image in images]
    
    st.write(f"Uploaded {len(images)} images.")
    
    # Call your grouping function
    grouped_images = group_images(opencv_images)
    # print(grouped_images)
    
    st.write(f"Number of groups: {len(grouped_images)}")
    
    # Display grouped images
    for i, group in enumerate(grouped_images):
        st.subheader(f"Group {i+1}")
        for img in group:
            st.image(img, width=300)  # fixed size, keeps aspect ratio
