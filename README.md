# 😊 Facial Expression Recognition

A deep learning model that recognizes human emotions from face images, served through an interactive **Streamlit** web app.

## 📌 Problem
Facial expressions are one of the main ways people communicate emotion. Recognizing them automatically is useful in human–computer interaction, customer-experience analysis, and behavioral research. The task is to classify a face image into one of six emotions: **Angry, Fear, Happy, Neutral, Sad, Surprise**.

## 📊 Dataset
- **[FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)**: 48×48 grayscale face images
- **Train:** 28,709 images · **Test:** 7,178 images
- The **"disgust"** class was removed. It had only 436 training images, about 6% of the largest class, which was too few to learn from reliably.
- The remaining classes were **balanced with random oversampling** to stop the model from favoring "Happy", the largest class.

## 🧠 Approach
1. **EDA:** counted images per class and visualized the class imbalance.
2. **Preprocessing:** rescaling plus data augmentation (rotation, shifts, zoom, horizontal flip) using `ImageDataGenerator`.
3. **Model:** a custom CNN with four convolutional blocks (32 → 64 → 128 → 512 → 512 filters), each with BatchNormalization, MaxPooling, and Dropout. L2 regularization on the deeper layers, then two dense layers (256, 512) and a softmax output.
4. **Training:** Adam optimizer (lr = 0.001), categorical cross-entropy, and **early stopping** (patience = 5) that restores the best weights.
5. **Evaluation:** accuracy, a per-class classification report, a confusion matrix, and visual checks on random test images.
6. **Deployment:** a Streamlit app that accepts uploaded images or an image URL.

## 📈 Results

| Metric | Value |
|---|---|
| Test accuracy | **57%** |
| Macro F1-score | 0.56 |

| Emotion | F1-score |
|---|---|
| Happy | 0.79 |
| Surprise | 0.77 |
| Neutral | 0.55 |
| Angry | 0.53 |
| Sad | 0.47 |
| Fear | 0.23 |

**Context:** FER-2013 is a hard benchmark. Images are small, low-resolution, and sometimes mislabeled, and human accuracy on it is only about **65%**. The model is strongest on clearly expressive emotions (Happy, Surprise). Its main weakness is **Fear**, which it often confuses with Sad and Surprise.

## 🖥️ Demo app
The Streamlit app (`img.py`) lets you:
- upload one or more `.jpg` face images, or
- paste an image URL

It shows the predicted emotion with a confidence score.

<!-- Add a screenshot of the app here: ![App screenshot](screenshot.png) -->

## 🚀 How to run
```bash
git clone https://github.com/youstinashraf/Facial-expression-recognition.git
cd Facial-expression-recognition
pip install -r requirements.txt

# 1. Train the model by running notebook02.ipynb; it saves model02.pkl
# 2. Launch the app
streamlit run img.py
```

## 🛠️ Tech stack
Python · TensorFlow / Keras · OpenCV · NumPy · Pandas · scikit-learn · imbalanced-learn · Matplotlib / Seaborn · Streamlit

## 🔮 Future improvements
- Transfer learning with a pre-trained backbone (e.g., VGG16, ResNet50) to push accuracy above 65%
- Face detection before classification, so the app works on full photos and not just cropped faces
- Real-time webcam emotion detection

## 📂 Files
| File | Description |
|---|---|
| `notebook02.ipynb` | EDA, preprocessing, model training, and evaluation |
| `img.py` | Streamlit web app |
| `FER.pdf` / `.pptx` | Project presentation |
