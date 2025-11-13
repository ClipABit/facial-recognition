from incdbscan import IncrementalDBSCAN
from deepface import DeepFace
import numpy as np
from sklearn.cluster import *

global all_embeddings

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

def cluster(clustering, embeddings):
    """
    Input:
        embeddings: list of vector embeddings
    Return:
        A list of integers >= -1. -1 means outlier, each int >=0 represent a cluster that vector belongs to
            e.g. output=[0, 1, 0] indicates embeddings[0] and embeddings[2] belong to same cluster
    """
    global all_embeddings
    
    X = np.stack(embeddings)
    print(type(X[0]))
    clustering.insert(X)
    # all_embeddings = np.vstack([all_embeddings, X])
    all_embeddings += embeddings
    # print(all_embeddings)
    labels = clustering.get_cluster_labels(np.array(all_embeddings))
    # print(labels)
    return labels

def classify_faces(clustering, img_lst):
    embeddings = []
    face_loc = []

    for img in img_lst:
        faces = detect_and_embed(img)

        for f in faces:
            embeddings.append(f["embedding"])
            face_loc.append({"img_path": img, "facial_area": f["facial_area"]})

    face_cluster = cluster(clustering, embeddings)

    print(face_cluster)
    

def main():
    global all_embeddings
    # all_embeddings = np.empty((0, 512))
    all_embeddings = []

    clustering = IncrementalDBSCAN(metric="cosine", eps=0.6, min_pts=2)

    classify_faces(clustering, [
        "../photos/image.png",
    ])

    classify_faces(clustering, [
        "../photos/image.png",
    ])

    classify_faces(clustering, [
        "../photos/photo1.jpg",
    ])

if __name__ == "__main__":
    main()