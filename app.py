import streamlit as st
import torch
import clip
from PIL import Image
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Function to load and preprocess images
def load_and_preprocess_images(image_paths):
    images = []
    for path in image_paths:
        image = Image.open(path)
        image = preprocess(image).unsqueeze(0).to(device)
        images.append(image)
    return images

# Function to compute image embeddings
def get_image_embeddings(images):
    with torch.no_grad():
        image_features = []
        for image in images:
            embedding = model.encode_image(image).cpu().numpy()
            image_features.append(embedding)
    return np.vstack(image_features)

def find_related_images(image_embeddings, target_index, top_k=5):
    target_embedding = image_embeddings[target_index].reshape(1, -1)
    similarities = cosine_similarity(target_embedding, image_embeddings)[0]
    related_indices = similarities.argsort()[-top_k-1:][::-1]
    related_indices = [idx for idx in related_indices if idx != target_index]
    return related_indices[:top_k]

def main():
    st.title("Related Image Finder Using CLIP")

    dir_path = st.text_input("Enter the directory path containing images:")
    image_name = st.text_input("Enter the image name:")

    if dir_path and image_name:
        if os.path.isdir(dir_path) and os.path.isfile(os.path.join(dir_path, image_name)):
            image_paths = [os.path.join(dir_path, img) for img in os.listdir(dir_path) if img.lower().endswith(('jpg', 'jpeg', 'png'))]

            if image_name in os.listdir(dir_path):
                images = load_and_preprocess_images(image_paths)
                image_embeddings = get_image_embeddings(images)
                target_index = image_paths.index(os.path.join(dir_path, image_name))
                related_image_indices = find_related_images(image_embeddings, target_index=target_index, top_k=3)

                st.header("Input Image")
                st.image(os.path.join(dir_path, image_name), caption="Input Image", width=200)

                st.header("Related Images")
                num_related = len(related_image_indices)
                cols = st.columns(num_related)  

                for i, idx in enumerate(related_image_indices):
                    with cols[i]:
                        st.image(image_paths[idx], caption=f"Related Image: {os.path.basename(image_paths[idx])}", width=200)
            else:
                st.error("Image not found in the specified directory.")
        else:
            st.error("Invalid directory path or image name.")

if __name__ == "__main__":
    main()
