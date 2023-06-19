import os
import cv2
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops
from skimage.transform import resize
from skimage.feature import hog
from sklearn.decomposition import PCA

# Specify the path to your image dataset
dataset_path = "./LungColon"

# Initialize empty lists for storing image features and labels
features = []
labels = []
image_paths = []

# Define the desired image size
image_size = (256, 256)

# Iterate through each folder in the dataset directory
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        # Iterate through each image file in the folder
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            if os.path.isfile(image_path):
                # Read the image and resize it to the desired size
                image = cv2.imread(image_path)
                image = cv2.resize(image, image_size)

                # Calculate average pixel value
                avg_pixel = np.mean(image.flatten())

                # Calculate color histograms
                hist_red = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
                hist_green = cv2.calcHist([image], [1], None, [256], [0, 256]).flatten()
                hist_blue = cv2.calcHist([image], [2], None, [256], [0, 256]).flatten()

                # Calculate texture features using GLCM
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                glcm = graycomatrix(gray_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast').mean()
                energy = graycoprops(glcm, 'energy').mean()
                correlation = graycoprops(glcm, 'correlation').mean()

                # Calculate shape descriptors
                contours, _ = cv2.findContours(gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_area = sum(cv2.contourArea(cnt) for cnt in contours)

                # Extract CNN features
                # Replace this part with your preferred method for extracting CNN features (e.g., using a pre-trained model)
                #cnn_features = np.zeros(100)

                # Extract edge features using Canny edge detection
                edges = cv2.Canny(gray_image, 100, 200)
                edge_count = np.sum(edges)

                # Concatenate all features into a single feature vector
                feature_vector = np.concatenate(([avg_pixel], hist_red, hist_green, hist_blue, [contrast, energy, correlation],
                                                 [contour_area], [edge_count]))

                # Add the feature vector to the features list
                features.append(feature_vector)
                # Add the label (parent folder name) to the labels list
                labels.append(folder_name)
                # Store the image path for later use
                image_paths.append(image_path)

# Convert the features and labels to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Perform label encoding on the class labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels, train_image_paths, test_image_paths = train_test_split(
    features, labels, image_paths, test_size=0.3, random_state=42)

# Create an instance of the decision tree classifier
clf = DecisionTreeClassifier()

# Train the decision tree classifier
clf.fit(train_features, train_labels)

# Make predictions on the test set
predictions = clf.predict(test_features)

# Convert labels and predictions back to original class names
test_labels = label_encoder.inverse_transform(test_labels)
predictions = label_encoder.inverse_transform(predictions)

print("Test Labels:", test_labels)
print("Predictions:", predictions)

# Print the image paths along with the corresponding predictions
for i in range(len(test_image_paths)):
    print("Image Path:", test_image_paths[i])
    print("Prediction:", predictions[i])
    print()

# Calculate and print metrics
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions, average='weighted')
recall = recall_score(test_labels, predictions, average='weighted')
f1 = f1_score(test_labels, predictions, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
