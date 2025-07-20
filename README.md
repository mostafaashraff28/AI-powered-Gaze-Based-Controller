# AI-powered-Gaze-Based-Controller
Real-time gaze-controlled robotic car using TensorFlow Lite, MediaPipe, and Raspberry Pi. Detects right-eye movements and converts them into motion commands[Close, left, right, forward] for hands-free navigation. Includes dataset capture, model training (10-fold CV), and deployment code.
Absolutely! Here's your **final polished `README.md`** file — fully structured, professional, and ready to publish with your GitHub repository:

---


https://github.com/user-attachments/assets/2ff1b342-abd4-442a-9e50-d316b50129f4


```markdown
# 👁️ AI-Powered Gaze-Based Robotic Car

A real-time gaze-controlled robotic system designed to assist individuals with motor impairments by allowing them to control a robotic vehicle using only eye movements. This project demonstrates the practical integration of computer vision, deep learning, and embedded systems to create accessible hands-free control solutions.

---

## 🎓 Graduation Project – AAST College of Engineering

This project was developed as part of my Bachelor's degree in Computer Engineering at the **Arab Academy for Science, Technology and Maritime Transport (AAST)**. It highlights my passion for building inclusive technologies that make a meaningful difference in real-world accessibility.

---

## 🔍 Project Overview

| Feature                         | Description                                                  |
|---------------------------------|--------------------------------------------------------------|
| **Input**                       | Real-time video from Raspberry Pi camera                     |
| **Gaze Detection**              | Right-eye tracking via MediaPipe FaceMesh                    |
| **Model Architecture**          | MobileNetV2 + custom classification head                     |
| **Deployment**                  | TensorFlow Lite model on Raspberry Pi 4                      |
| **Output**                      | Directional commands via Bluetooth (forward, left, right)    |

---

## 🎯 Objectives

- **Develop** a hands-free robotic control system using gaze.
- **Enhance** detection accuracy with landmark filtering and smoothing.
- **Maintain** real-time performance on edge devices like Raspberry Pi 4.
- **Ensure** a calibration-free, user-friendly experience.
- **Validate** performance using 10-fold subject-independent cross-validation.

---

## 🧠 Technologies Used

- **Python 3.9**
- **TensorFlow 2.x + TensorFlow Lite**
- **OpenCV**
- **MediaPipe**
- **Raspberry Pi 4 (Linux)**
- **Bluetooth Serial Communication (`pyserial`)**
- **Keras, NumPy, Matplotlib, Seaborn**

---

## 📁 Project Structure

```

Gaze-Controlled-Robotic-Car/
├── dataset\_collection/
│   └── RIGHT_EYE.py                  # Eye image capture using MediaPipe 4 classes[Forward, close, left, right]
│
├── model\_training/
│   └── 20-10kfolds.ipynb             # Model training with 10-fold CV (MobileNetV2)
│
├── model\_integration/
│   └── Full\_code\_GUI\_DONE.py        # Real-time inference + Bluetooth control
│
├── best\_model\_fold1.tflite          # Optimized model for deployment
├── requirements.txt                 # Python dependencies
├── README.md                        # Full documentation
└── LICENSE                          # MIT License

````

---

## 🚀 How to Use

### 1. Collect Right-Eye Dataset
Capture labeled eye images with:
```bash
python dataset_collection/RIGHT_EYE.py
````

* Change `CLASS_LABEL` and `PERSON_ID` inside the script.

---

### 2. Train the Model

Run:

```bash
model_training/20-10kfolds.ipynb
```

* Trains MobileNetV2 in two phases (frozen head → fine-tuning)
* Uses cosine decay and early stopping
* Outputs `.keras` and `.tflite` models

---

### 3. Run Real-Time Controller

On Raspberry Pi (with camera + Bluetooth):

```bash
python model_integration/Full_code_GUI_DONE.py
```

* Loads `.tflite` model
* Detects right eye → predicts gaze direction
* Sends control command to robot via `/dev/rfcomm0`

---

## 📊 Model Performance

| Metric               | Result                       |
| -------------------- | ---------------------------- |
| Input Image Size     | 224 × 224 (right eye)        |
| Accuracy (Best Fold) | 98.9%                        |
| Evaluation           | 10-fold cross-validation     |
| Real-Time FPS        | \~8–10 frames per second     |
| Inference Speed      | \~90–100ms per frame (RPi 4) |

---

## 🔬 Model Training Details

* **Base Model**: MobileNetV2 pretrained on ImageNet
* **Classifier Head**: GlobalAveragePooling → Dropout → Dense(softmax)
* **Loss Function**: Categorical Crossentropy with label smoothing (0.1)
* **Optimizer**: Adam
* **Learning Rate**:

  * Phase 1: Cosine decay (1e-3 → 1e-6)
  * Phase 2: Fixed (2e-5)
* **Data Augmentation**:

  * Rotation ±15°
  * Zoom 0.8–1.2
  * Brightness 0.6–1.4
  * Width/Height shift ±10%
* **Evaluation Metrics**: Accuracy, F1-score, Confusion Matrix

---

## 📦 Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
tensorflow
opencv-python
mediapipe
numpy
pyserial
matplotlib
seaborn
scikit-learn
```

---

## 🧑‍💻 Author

**\Mostafa Ashraf Mostafa]**
B.Sc. in Computer Engineering – AAST
📫 Email: \[[mostefaashraf@gmail.com]]
🔗 LinkedIn: \[www.linkedin.com/in/mostafa-ashraf-074792324]
🌐 Portfolio: \[https://mostafa.mikawi.org/]

> Open to roles in:
> ⚙️ Computer Vision • 🧠 Embedded AI • 🤖 Robotics • ♿ Assistive Technologies

---

## 📽️ Demo Video

🎥 Watch the system in action:
🔗 \[www.linkedin.com/in/mostafa-ashraf-074792324]

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
You are free to use, modify, and distribute it with proper attribution.

---

## 🙏 Acknowledgments

Special thanks to my academic supervisors, peers, and everyone who supported me in this journey. This project is dedicated to using AI for good — empowering those who need it most.

---

