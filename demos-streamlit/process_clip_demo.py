import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from face_recognition import FaceRepository
import streamlit as st
import cv2
from PIL import Image
import numpy as np
import time

def process(fr: FaceRepository, clip_id, image_list):
    fr.add_images(clip_id, image_list)
    # print(fr.clip_faces_map[clip_id])x

def main():
    # Create the repository only once, make sure it doesn't re-created on every interaction
    if "face_repo" not in st.session_state:
        st.session_state.face_repo = FaceRepository()
    fr = st.session_state.face_repo  # short alias

    st.title("Process Clip Demo")

    # Clip ID (integer) inputed by user
    clip_id = st.number_input("Enter ID (integer)", min_value=0, step=1)

    # Upload multiple images for that clip id
    uploaded_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    # Convert uploaded files to PIL Images
    image_list = []
    if uploaded_files:
        images = [Image.open(file) for file in uploaded_files]
        # Keep images as RGB numpy arrays for consistent display in Streamlit
        # DeepFace can accept numpy arrays; if it requires BGR you can convert before passing.
        image_list = [np.array(image.convert("RGB")) for image in images]

    # Process button
    if st.button("Process"):
        if not image_list:
            st.error("Please upload at least one image.")
        else:
            process(fr, clip_id, image_list)
            # persist original uploaded images for this clip in session state so they are shown
            if "orig_images" not in st.session_state:
                st.session_state["orig_images"] = {}
            st.session_state["orig_images"].setdefault(str(clip_id), [])
            # extend stored originals with the newly uploaded images
            st.session_state["orig_images"][str(clip_id)].extend(image_list)
            st.success(f"Processed {len(image_list)} image(s) for clip {clip_id}.")

    # --- Show each clip as its own section: Clip N / clip N images / clip N faces ---
    st.markdown("---")


    # gather clip ids from repository and from uploaded originals
    clip_ids = set()
    try:
        clip_ids.update(list(fr.clip_faces_map.keys()))
    except Exception:
        pass
    if "orig_images" in st.session_state:
        for k in st.session_state["orig_images"].keys():
            try:
                clip_ids.add(int(k))
            except Exception:
                pass

    if not clip_ids:
        st.info("No clips available yet. Upload images and press Process to create clips.")
        return

    for cid in sorted(clip_ids):
        st.subheader(f"Clip {cid}")

        # --- Originals for this clip (one at a time) ---
        st.markdown(f"**clip {cid} images**")
        orig_images = []
        if "orig_images" in st.session_state:
            orig_images = st.session_state["orig_images"].get(str(cid), [])

        orig_idx_key = f"orig_idx_{cid}"
        if orig_idx_key not in st.session_state:
            st.session_state[orig_idx_key] = 0

        if not orig_images:
            st.info("No original images uploaded for this clip yet.")
        else:
            max_orig_idx = len(orig_images) - 1
            st.session_state[orig_idx_key] = max(0, min(st.session_state[orig_idx_key], max_orig_idx))
            if max_orig_idx == 0:
                st.image(orig_images[0], use_container_width=True, caption=f"clip {cid} original 0")
            else:
                cols = st.columns([1, 1, 6])
                with cols[0]:
                    if st.button("Prev original", key=f"prev_orig_{cid}"):
                        st.session_state[orig_idx_key] = max(0, st.session_state[orig_idx_key] - 1)
                with cols[1]:
                    if st.button("Next original", key=f"next_orig_{cid}"):
                        st.session_state[orig_idx_key] = min(max_orig_idx, st.session_state[orig_idx_key] + 1)

                cur = st.slider(f"Original image index (clip {cid})", min_value=0, max_value=max_orig_idx, value=st.session_state[orig_idx_key], key=f"slider_orig_{cid}")
                st.session_state[orig_idx_key] = cur
                st.image(orig_images[st.session_state[orig_idx_key]], use_container_width=True, caption=f"clip {cid} original {st.session_state[orig_idx_key]}")

        # --- Faces for this clip: show all detected face crops in a row ---
        st.markdown(f"**clip {cid} faces**")
        face_images = fr.get_face_images_in_clip(int(cid))
        if not face_images:
            st.info("No faces detected for this clip yet.")
            continue

        # render faces in rows, up to 8 per row
        per_row = 8
        for start in range(0, len(face_images), per_row):
            row = face_images[start:start + per_row]
            cols = st.columns(len(row))
            for c, img in zip(cols, row):
                with c:
                    st.image(img, use_container_width=True)

if __name__ == "__main__":
    main()