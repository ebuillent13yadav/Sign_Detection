# 🤟 Real-Time Sign Language Gesture Recognition

A real-time hand gesture recognition system that detects static sign language gestures from a webcam feed and converts them into spoken audio using a custom-trained CNN.

---

## 📁 Project Structure

```
├── dataset_sign_detection_3.0/
│   ├── train/
│   │   ├── approve/
│   │   ├── call_me/
│   │   ├── disapprove/
│   │   ├── fist/
│   │   ├── one/
│   │   ├── loser/
│   │   ├── ok/
│   │   ├── peace/
│   │   ├── rock/
│   │   └── stop/
│   └── val/
│       └── (same structure as train)
├── codes/
│   ├── data_collection.ipynb
│   ├── model_creation.ipynb
│   └── prediction.ipynb
├── gesture_model.keras
└── README.md
```

---

## ✋ Supported Gestures

| Gesture | Description |
|--------|-------------|
| `approve` | Thumbs up |
| `call_me` | Hand phone gesture |
| `disapprove` | Thumbs down |
| `fist` | Closed fist |
| `one` | index finger |
| `loser` | L-shape on forehead |
| `ok` | OK sign |
| `peace` | Two fingers up |
| `rock` | Rock on sign |
| `stop` | Open palm facing forward |

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/sign-language-recognition.git
cd sign-language-recognition
```

### 2. Create and activate a virtual environment (recommended)

```bash
conda create -n tf-gpu python=3.9
conda activate tf-gpu
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
tensorflow==2.10.0
numpy==1.23.5
opencv-python==4.8.0.76
pyttsx3==2.99
scikit-learn==1.3.0
```

### Full Dependency Versions

| Library | Version |
|--------|---------|
| Python | 3.9 |
| TensorFlow | 2.10.0 |
| NumPy | 1.23.5 |
| OpenCV (`opencv-python`) | 4.8.0.76 |
| pyttsx3 | 2.99 |
| scikit-learn | 1.3.0 |
| CUDA | 11.2 |
| cuDNN | 8.1 |

> **Note:** CUDA 11.2 and cuDNN 8.1 are required for GPU acceleration with TensorFlow 2.10. If you're on CPU only, CUDA/cuDNN are not needed.

---

## 🗃️ Dataset

- **1000 images per class** for training
- **200 images per class** for validation
- Images captured at 128×128 resolution via webcam
- Same background across captures; augmentation handles generalization

> 📥 Download the dataset and pre-trained model from [Google Drive](https://drive.google.com/drive/folders/1fs7hk-6hUVE1limRQXXNTDDXmLM87AYf?usp=sharing)

Place the downloaded dataset folder at the root of the project as `dataset_sign_detection_3.0/`.

---

## 🚀 Usage

### Step 1 — Collect Data (optional, dataset already provided)

```bash
python codes/data_collection.ipynb
```

- Press `S` to start/stop capturing images
- Press `Q` to quit
- Change `class_name` at the top of the script for each gesture

### Step 2 — Train the Model (optional, model already provided)

```bash
python codes/model_creation.ipynb
```

Training runs for 20 epochs with the following augmentation:
- Zoom (±10%)
- Brightness shift (0.8–1.2)
- Width/height shift (±10%)
- Horizontal flip

### Step 3 — Run Inference

```bash
python codes/prediction.ipynb
```

- Place your hand inside the **green ROI box** in the top-right corner
- Hold the gesture steady for ~10 frames to confirm
- Recognized gesture is displayed on screen and spoken aloud
- Press `ESC` to quit

---

## 🧠 Model Architecture

A custom CNN trained entirely from scratch (no pretrained backbones).

```
Input: (128, 128, 3)

Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Dropout(0.25)
Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Dropout(0.25)
Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Dropout(0.30)
Block 4: Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.30)
Block 5: Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Dropout(0.30)

Flatten
Dense(512) → BN → Dropout(0.5)
Dense(256) → Dropout(0.4)
Dense(10, softmax)
```

- **Optimizer:** Adam (lr=0.0001)
- **Loss:** Categorical Crossentropy
- **Epochs:** 20
- **Batch size:** 16

---

## 🔊 Text-to-Speech

Speech output is handled by `pyttsx3` — a fully offline TTS engine. No internet connection is required.

```python
import pyttsx3
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.say("hello")
engine.runAndWait()
```

---

## 🔍 Inference Logic

| Mechanism | Details |
|-----------|---------|
| **ROI** | Fixed 250×250 box in top-right corner of frame |
| **Confidence threshold** | Predictions below 0.6 are ignored |
| **Stability check** | Gesture must be predicted consistently for 10+ frames |
| **Duplicate suppression** | Same gesture is not spoken twice in a row |

---

## ⚠️ Known Limitations

- Works best with a plain/consistent background
- Sensitive to lighting changes outside the training distribution
- Static gestures only — no motion-based signs supported
- ROI is fixed; hand must be placed in the top-right box

---

## 👤 Author

**Divanshu Yadav**  
Real-Time Sign Language to Speech Translation
