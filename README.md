# Related Image Finder Using CLIP

This Streamlit application allows users to find and display images related to a specified input image using OpenAI's CLIP model. The app computes the similarity between the input image and other images in a directory, and displays the most similar images.

![demo](demo/demo.gif)

## Features

- **Image Upload**: Input image file is selected from a specified directory.
- **Image Processing**: The input image and images in the directory are processed and embedded using the CLIP model.
- **Related Images**: The app finds and displays the most similar images from the directory to the input image.
- **Grid Display**: Input image and related images are displayed in a grid format with fixed size for easy comparison.

## Requirements

- Python 3.7+
- Streamlit
- PyTorch
- CLIP
- PIL
- NumPy
- scikit-learn

You can install the required packages using `pip`:

```bash
pip install streamlit torch torchvision clip-by-openai pillow numpy scikit-learn
```

## Usage

1. **Clone or Download the Repository**: Ensure you have the script saved as `app.py` in your local directory.

2. **Run the Streamlit App**:
   ```bash
   streamlit run app.py
   ```

3. **Interact with the App**:
   - **Directory Path**: Enter the path to the directory containing images.
   - **Image Name**: Enter the name of the image file to be used as the input image.

4. **View Results**:
   - The app will display the input image at a fixed size of 200 x 200 pixels.
   - It will also show the most similar images from the directory in a grid format, with each related image sized to 200 x 200 pixels.

## Code Explanation

- **CLIP Model**: The CLIP model (`ViT-B/32`) is used for generating image embeddings and computing similarities between images.
- **Image Loading and Preprocessing**: Images are loaded and preprocessed to be compatible with the CLIP model.
- **Embedding Calculation**: Image embeddings are computed using the CLIP model to represent the images in a high-dimensional space.
- **Finding Related Images**: Similarity between the input image and other images is computed using cosine similarity, and the top similar images are selected.
- **Streamlit Interface**: Provides inputs for the directory path and image name, and displays the results in a user-friendly format.

## Example

```python
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
```