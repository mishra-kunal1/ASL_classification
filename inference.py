# Import required libraries
import torchvision.transforms as transforms
from PIL import Image
import torch
from torchvision.models import resnet50
import numpy as np
import cv2

# Open the default camera (0) for video capture
cap = cv2.VideoCapture(0)

# Create a background subtractor using MOG2
fgbg = cv2.createBackgroundSubtractorMOG2()

# Define the set of labels for the prediction classes
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del','nothing', 'space']

# Set the number of classes
num_classes = 29

# Load the pre-trained ResNet-50 model and modify its final layer to output the desired number of classes
resnet_model = resnet50(pretrained=True)
in_features = resnet_model.fc.in_features
resnet_model.fc = torch.nn.Linear(in_features, num_classes)

# Load the pre-trained weights for the modified ResNet-50 model
resnet_model.load_state_dict(torch.load('trained_model_weights', map_location=torch.device('cpu')))

# Continuously capture frames from the video feed and perform predictions
while (cap.isOpened()):
    # Read a frame from the video feed
    ret, img = cap.read()

    # Initialize the predicted class as an empty string
    predicted_class = ''

    # Convert the frame to grayscale and apply thresholding to obtain a binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Apply background mask to thresholded image

    # Find contours in the thresholded image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # If any contours are detected, crop the region of interest and make a prediction
    if len(contours) > 0:
        # Define the region of interest
        x, y, w, h = 50, 200, 500, 500

        # Draw a rectangle around the region of interest
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the region of interest
        crop_img = img[y:y+h, x:x+w]

        # Apply the necessary data transforms to the cropped image
        data_transforms = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        final_img = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        final_img = data_transforms(final_img)
        final_img = final_img.to('cpu')

        # Put the model into evaluation mode and perform the prediction
        resnet_model.eval()
        prediction = resnet_model(final_img[None])
        index=torch.max(prediction, dim=1)[1]
        predicted_class = labels[index.item()]

    # Draw a rectangle around the region of interest and display the predicted class
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(img, "Model Prediction : " + predicted_class, (x, y+h-520), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 2)

    # Display the image
    cv2.imshow("Frame", img)

    # Wait for a key press and exit if the 'esc' key is pressed
    k = cv2.waitKey(10)
    if k == 27:
        break