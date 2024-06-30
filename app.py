import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

# Import your classes from the file where they are defined
from models import GeneratorResNet, Discriminator, ResidualBlock  # Adjust the import according to your file structure

# Define the necessary transforms
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load the pre-trained models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update with your model paths
model_path_a2b = "saved_models/G_AB.pth"
model_path_b2a = "saved_models/G_BA.pth"

# Assuming the input shape and number of residual blocks are known
input_shape = (3, 100, 100)  # Adjust if necessary
num_residual_blocks = 9  # Adjust if necessary

# Instantiate the models
G_AB = GeneratorResNet(input_shape, num_residual_blocks).to(device)
G_BA = GeneratorResNet(input_shape, num_residual_blocks).to(device)

# Load state dicts
G_AB.load_state_dict(torch.load(model_path_a2b, map_location=device))
G_BA.load_state_dict(torch.load(model_path_b2a, map_location=device))

G_AB.eval()
G_BA.eval()

# Streamlit app
st.title("CycleGAN Image Translation")
st.write("Upload an image to see the translated results.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    option = st.selectbox("Choose translation direction:", ["Young to Old", "Old to Young"])

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    flag = 1

    # Generate images based on selected option
    with torch.no_grad():
        if option == "Young to Old":
            intermediate = G_AB(image_tensor)
            result_image_tensor = G_BA(intermediate)
            flag = 1
        else:
            intermediate = G_BA(image_tensor)
            result_image_tensor = G_AB(intermediate)
            flag = 0

    # Postprocess and save images
    intermediate_image = (intermediate * 0.5 + 0.5).squeeze().permute(1, 2, 0).cpu().numpy()
    result_image = (result_image_tensor * 0.5 + 0.5).squeeze().permute(1, 2, 0).cpu().numpy()

    # Display images with reduced dimensions
    col1, col2 = st.columns(2)
    if flag is 1:
        col1.image(intermediate_image, caption='Old', use_column_width=True)
        col2.image(result_image, caption='Reconstructed', use_column_width=True)
    else:
        col1.image(intermediate_image, caption='Young', use_column_width=True)
        col2.image(result_image, caption='Reconstructed', use_column_width=True)
        
