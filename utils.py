# from deepface import DeepFace
import numpy as np
from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
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

    # d1 = np.linalg.norm(np.array(embeddings[1])-np.array(embeddings[3]))
    # d2 = np.linalg.norm(np.array(embeddings[3])-np.array(embeddings[4]))
    # print(d1, d2)
    # d1 = np.linalg.norm(np.array(embeddings[0])-np.array(embeddings[1]))
    # print(d1)

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
