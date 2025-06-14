import argparse
from torchvision import transforms
from model import ResNet18, Residual
from PIL import Image
import torch
import os
import sys

"""
This script uses a trained ResNet18 model to predict whether a given image is a cat or a dog.
The image can be any external photo (not limited to the original dataset).
"""

def get_args():
    parser = argparse.ArgumentParser(description="Predict a single image using ResNet18")
    parser.add_argument('--img', type=str, default='cc.jpg', help='Path to the input image')
    parser.add_argument('--model', type=str, default='model/epochs_10-batch_size_16-lr_0_002-SEBlock_False-2.pth', help='Path to the trained model (.pth)')
    args = parser.parse_args()

    # Check if arguments are empty strings
    if not args.img.strip():
        print("Error: --img cannot be empty.")
        sys.exit(1)
    if not args.model.strip():
        print("Error: --model cannot be empty.")
        sys.exit(1)

    return args

def main(img_path, model_path):
    # Check if the image and model files exist
    if not os.path.isfile(img_path):
        print(f"Error: Image file not found at: {img_path}")
        sys.exit(1)
    if not os.path.isfile(model_path):
        print(f"Error: Model file not found at: {model_path}")
        sys.exit(1)

    # Choose the computation device
    device = "cuda" if torch.cuda.is_available() else 'cpu'

    print(f"📦 Loading model from: {model_path}")
    model = ResNet18(Residual)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    # Load and preprocess the image
    print(f"Loading image from: {img_path}")
    image = Image.open(img_path).convert("RGB")
    normalize = transforms.Normalize([0.48607032, 0.45353173, 0.4160252],
                                     [0.06886391, 0.06542894, 0.0667423])
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
        predicted_label = torch.argmax(output, dim=1).item()
        classes = ['cat', 'dog']
        print(f"Prediction result: {classes[predicted_label]}")

if __name__ == "__main__":
    args = get_args()
    main(args.img, args.model)
