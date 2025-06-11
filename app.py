import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import streamlit as st
import cv2
import os
import requests
from sklearn.metrics import precision_score, recall_score, f1_score

# Load pre-trained model
model = load_model('C:/Users/hp/Desktop/Image Classification/Image_classify.keras')
data_cat = ['Fire', 'None', 'Smoke']  # Add 'None' category for no fire or smoke
detection_threshold = 0.8  # Increased threshold for more confident detection
img_height = 180
img_width = 180

# Telegram Bot Token and Chat ID
TOKEN =   # Replace with your bot's API token
CHAT_ID = # Replace with your chat ID

# Function to send a message via Telegram Bot
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    params = {
        "chat_id": CHAT_ID,
        "text": message
    }
    response = requests.post(url, params=params)
    if response.status_code == 200:
        print("Message sent successfully!")
    else:
        print(f"Failed to send message. Status code: {response.status_code}, Error: {response.text}")

# Streamlit Application
st.title("Integrated System For Forest Fire Recognition And Monitoring")
st.write("Choose an option below to proceed:")

option = st.selectbox("Select Mode", ["Upload an Image", "Real-Time Camera Detection"])

if option == "Upload an Image":
    st.write("Please upload an image of Fire or Smoke to classify it.")
    img_path = st.file_uploader("Upload an image of Fire or Smoke", type=["jpg", "jpeg", "png"])

    if img_path is not None:
        # Pre-process the image
        image_load = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
        img_arr = tf.keras.utils.img_to_array(image_load)
        img_bat = tf.expand_dims(img_arr, 0)

        # Make prediction
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict[0])

        # Show uploaded image
        st.image(img_path, caption="Uploaded Image", use_column_width=True)

        # Prediction Results
        predicted_class_index = np.argmax(score)
        predicted_class = data_cat[predicted_class_index]
        accuracy = np.max(score) * 100

        # Apply a more strict detection threshold
        if accuracy < detection_threshold * 100:
            st.write("**Prediction**: No Fire or Smoke detected in the image.")
        elif predicted_class == 'None':
            st.write("**Prediction**: No Fire or Smoke detected in the image.")
        else:
            st.write(f"**Prediction**: {predicted_class} detected with accuracy of {accuracy:.2f}%")

        # Plot Prediction Scores for each class
        fig, ax1 = plt.subplots(figsize=(5, 5))
        ax1.bar(data_cat, score.numpy() * 100, color=['red', 'orange', 'green'])
        ax1.set_title('Prediction Scores')
        ax1.set_ylabel('Confidence Score (%)')
        ax1.set_ylim(0, 100)

        st.pyplot(fig)

        # Assuming true label is provided for evaluation (for now, you have to manually provide it)
        true_label = st.selectbox("Select the true label of the uploaded image:", data_cat)
        true_label_index = data_cat.index(true_label)

        # Metrics Evaluation
        predicted_label = [predicted_class_index]  # Model's predicted label
        true_label_list = [true_label_index]  # User-specified true label for the uploaded image

        # Precision, Recall, F1-score
        precision = precision_score(true_label_list, predicted_label, average='weighted', zero_division=1)
        recall = recall_score(true_label_list, predicted_label, average='weighted', zero_division=1)
        f1 = f1_score(true_label_list, predicted_label, average='weighted', zero_division=1)

        st.write(f"**Precision**: {precision:.2f}")
        st.write(f"**Recall**: {recall:.2f}")
        st.write(f"**F1-Score**: {f1:.2f}")

    else:
        st.write("Please upload an image to classify.")

elif option == "Real-Time Camera Detection":
    st.write("Starting Real-Time Camera Detection...")

    # Open the video feed (using webcam here)
    video = cv2.VideoCapture(0)

    if not video.isOpened():
        st.error("Error: Could not open video feed.")
        st.stop()

    while True:
        ret, frame = video.read()
        if not ret:  # Check if there is a valid frame
            st.error("Error: Cannot read the video feed or end of the video reached.")
            break

        # Resize frame
        frame = cv2.resize(frame, (1000, 600))

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(frame, (15, 15), 0)

        # Convert to HSV color space
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Define lower and upper bounds for fire-like colors in HSV
        lower = [18, 85, 85]
        upper = [35, 255, 255]
        lower = np.array(lower, dtype='uint8')
        upper = np.array(upper, dtype='uint8')

        # Create a mask to detect regions with colors in the specified range
        mask = cv2.inRange(hsv, lower, upper)

        # Apply mask on original frame
        output = cv2.bitwise_and(frame, frame, mask=mask)
        number_of_total = cv2.countNonZero(mask)

        if int(number_of_total) > 15000:
            st.warning("🔥 Fire Detected!")
            send_telegram_message("⚠️ Fire Detected! Please check immediately.")

        # Display the output frame using OpenCV
        cv2.imshow("Video Result", output)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video and close all OpenCV windows
    video.release()
    cv2.destroyAllWindows()





































































































































