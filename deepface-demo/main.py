from config import *

def find_all_faces(img_path: str, db_path: str, model: str = 'VGG-Face'):
    """
    Find all the faces in the frames folder that match the given image.

    Args:
        img_path: Path to the input image file.
        db_path: Path to the database folder containing images to compare against.
        model: Face recognition model to use (e.g. VGG-Face, Facenet, OpenFace, DeepFace, etc)
    
    Returns:
        None
    """  
    dfs = DeepFace.find(img_path, db_path, model_name=model, enforce_detection=False)     # enforce_detection=False to skip face detection if no faces are detected
    print(dfs)

def analyze_face(img_path: str, actions: list = [''], detector: str = 'opencv'):
    """
    Analyze the face in the given image and print the results.

    Args:
        img_path: Path to the input image file.
        actions: List of actions to perform (e.g. age, gender, emotion, race)
        detector: Face detector backend to use (e.g. opencv, ssd, dlib, mtcnn, retinaface)

    Returns:
        None
    """
    objs = DeepFace.analyze(img_path, actions, detector_backend=detector, enforce_detection=False)   # enforce_detection=False to skip face detection if no faces are detected
    print(objs)

img_path = "your-image.jpg"     # can be jpg, jpeg, png
db_path = "./database"      # folder containing images to compare against

# models for face recognition
models = [
    "VGG-Face", "Facenet", "Facenet512", "OpenFace", "DeepFace",
    "DeepID", "ArcFace", "Dlib", "SFace", "GhostFaceNet",
    "Buffalo_L",
]
model = models[1]

# face detector backends
backends = [
    'opencv', 'ssd', 'dlib', 'mtcnn', 'fastmtcnn',
    'retinaface', 'mediapipe', 'yolov8n', 'yolov8m', 
    'yolov8l', 'yolov11n', 'yolov11s', 'yolov11m',
    'yolov11l', 'yolov12n', 'yolov12s', 'yolov12m',
    'yolov12l', 'yunet', 'centerface',
]
detector = backends[5]
actions=['gender', 'emotion']

# find_all_faces(img_path, db_path, model)
# analyze_face(img_path, actions, detector)
