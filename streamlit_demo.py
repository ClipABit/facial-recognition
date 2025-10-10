import streamlit as st
from PIL import Image
from io import BytesIO
from utils import *
import cv2
import tempfile

# Example grouping function
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
    # print("images:", images)
    # images = []
    # for file in uploaded_files:
    #     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    #         tmp.write(file.read())
    #         tmp.flush()
    #         images.append(tmp.name)
    # print(images)
    
    st.write(f"Uploaded {len(images)} images.")
    
    # Call your grouping function
    grouped_images = group_images(opencv_images)
    # print(grouped_images)
    
    st.write(f"Number of groups: {len(grouped_images)}")
    
    # Display grouped images
    for i, group in enumerate(grouped_images):
        st.subheader(f"Group {i+1}")
        cols = st.columns(len(group))
        for col, img in zip(cols, group):
            col.image(img, use_column_width=True)
