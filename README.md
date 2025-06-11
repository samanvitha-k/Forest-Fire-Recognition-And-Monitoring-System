# Forest Fire Recognition and Monitoring System 🔥🌲

A real-time CNN-based system for detecting and monitoring forest fires and smoke using image classification and  live video frame analysis using a webcam.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Web Interface](#web-interface)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)



## Features

✅ Upload images for classification as **Fire**, **Smoke**, or **Safe**  
✅ Real-time webcam-based fire and smoke detection  
✅ Deep learning-based classification using a custom CNN model  
✅ User-friendly web interface built with Streamlit  
✅ High model accuracy (~99.8%) on test data  
✅ Lightweight and efficient design for easy deployment  

## Installation

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- OpenCV and TensorFlow

### Steps

1️⃣ **Clone the repository**


git clone https://github.com/samanvitha-k/forest-fire-recognition-and-monitoring-system.git

cd forest-fire-recognition-and-monitoring-system

2️⃣ **Create and activate a virtual environment**

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

3️⃣ **Install required dependencies**


pip install -r requirements.txt


## Usage

### Streamlit Web App

For image-based fire and smoke classification:


cd code
streamlit run app.py

* Open your browser at [http://localhost:8501](http://localhost:8501)
* Upload an image and get instant classification results

### Real-Time Webcam Detection

For real-time detection of fire and smoke using webcam:

python code/camera.py
# or
python code/camera1.py


## Web Interface

The app includes:

* A user-friendly Streamlit interface
* Upload form for images
* Live webcam detection feed
* Displays prediction label for images/video frames

## Dataset

This project uses a **custom dataset** of **daytime fire and smoke images**, sourced from publicly available datasets and manually labeled:

* **Classes:** Fire, Smoke, Safe
* **Training images:** \~12,000
* **Validation images:** \~1,000
* **Test images:** \~1,000
* The dataset is not included in the GitHub repo to save space — available locally for development.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/new-feature).
3. Make your changes and commit (git commit -m 'Add new feature).
4. Push to the branch (git push origin feature/new-feature).
5. Open a pull request.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


## Contact

If you have any questions or issues with the project, feel free to reach out:

* **GitHub:** [samanvitha-k](https://github.com/samanvitha-k)
* **Email:** [samanvithak@gmail.com](mailto:samanvithak@gmail.com)
