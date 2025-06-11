import cv2
import numpy as np
import requests

# Set up Telegram Bot Token and Chat ID
TOKEN = "7999244610:AAH0ENWbM_Lgu-DzXK-mBmlF0XKs2IrBL-c"  # Replace with your bot's API token
CHAT_ID = "1401877732"  # Replace with your chat ID

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

# Open the video feed (using webcam here)
video = cv2.VideoCapture(1)

if not video.isOpened():
    print("Error: Could not open video feed.")
    exit()

while True:
    ret, frame = video.read()
    if not ret:  # Check if there is a valid frame
        print("Error: Cannot read the video feed or end of the video reached.")
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
        print("Fire Detected")
        # Send a message to Telegram
        send_telegram_message("⚠️ Fire Detected! Please check immediately.")

    # Display output
    cv2.imshow("Video Result", output)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
