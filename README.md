---

````markdown
# 🧠 Image Caption Generator using CNN + LSTM

Automatically generate descriptive captions for images using Deep Learning, combining the power of Computer Vision (CNN) and Natural Language Processing (LSTM).  
Built with TensorFlow / Keras, trained on the MS COCO dataset, and capable of generating human-like image descriptions.

---

## 🎯 Project Overview

The Image Caption Generator is an AI model that takes an image as input and outputs a text caption that describes the image content.  
It replicates how platforms like Google Photos or Pinterest automatically describe uploaded images.

---

## ⚙️ Tech Stack

- Frameworks: TensorFlow, Keras  
- Languages: Python  
- Models Used: CNN (InceptionV3 / VGG16 / ResNet50), LSTM  
- Dataset: MS COCO  
- Libraries: NumPy, Pandas, Matplotlib, Pillow, OpenCV, NLTK  
- Evaluation Metric: BLEU Score  

---

## 🧩 Architecture

The system follows an Encoder–Decoder architecture:

1. CNN Encoder:  
   - Extracts visual features from images using a pretrained CNN model.  
   - Converts the image into a numerical feature vector.

2. LSTM Decoder:  
   - Takes the encoded image features and generates descriptive captions word by word.  
   - Learns contextual relationships between visual and textual data.

3. Combined Model:  
   - The CNN and LSTM are merged to form an end-to-end trainable network.  
   - Uses categorical cross-entropy loss for optimization.

---

## 🧠 Workflow

1. Image Feature Extraction: Use a pretrained CNN (InceptionV3 / VGG16 / ResNet50) to extract image features.  
2. Caption Preprocessing: Tokenize and clean captions using NLTK & Keras Tokenizer.  
3. Training: Combine image features with text sequences and train the CNN + LSTM model.  
4. Evaluation: Measure BLEU score to compare predicted and ground-truth captions.  
5. Inference: Generate captions for unseen images using the trained model.

---

## 🖼️ Example Output

Input: 🐶 Dog playing in the park  
Generated Caption: "A brown dog playing with a ball in the grass."

---

## 💻 Installation & Setup

1. Clone the Repository

```bash
git clone https://github.com/yourusername/Image-Caption-Generator.git
cd Image-Caption-Generator
````

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Download Dataset

Download the MS COCO dataset ([https://cocodataset.org/#home](https://cocodataset.org/#home)) and place it inside the /data folder.

---

## 🧠 Training the Model

```bash
python extract_features.py     # Extract image features using CNN
python preprocess_captions.py  # Clean and tokenize captions
python train.py                # Train the CNN + LSTM model
```

---

## 📷 Generate Captions

```bash
python predict.py --image test_images/example.jpg
```

Output Example:

```
Input Image: ![Sample Output](images/text_output.png)
Generated Caption: "A man riding a bike on a street."
```
---
## 📂 Project Structure

```
Image-Caption-Generator/
│
├── data/                    # Dataset (images + captions)
├── features/                # Extracted CNN features
├── models/                  # Saved model weights
├── notebooks/               # Jupyter notebooks
├── app.py                   # Streamlit/Flask app
├── train.py                 # Model training script
├── predict.py               # Caption generation
├── extract_features.py       # CNN feature extraction
├── preprocess_captions.py    # Text preprocessing
├── requirements.txt
└── README.md
```

---
## 🧠 Expected Outcome

* Generates accurate and meaningful captions for unseen images
* Demonstrates integration of Computer Vision and NLP
* Provides strong foundation in deep learning encoder–decoder models
* Optional GUI for real-time captioning

---

## 📚 References

* MS COCO Dataset: [https://cocodataset.org/#home](https://cocodataset.org/#home)
* Show and Tell: A Neural Image Caption Generator (Google Research): [https://arxiv.org/abs/1411.4555](https://arxiv.org/abs/1411.4555)
* TensorFlow Documentation: [https://www.tensorflow.org/](https://www.tensorflow.org/)
* Keras API Reference: [https://keras.io/api/](https://keras.io/api/)

---

## 👤 Author

Your Name
B.Tech ECE | SRM Institute of Science and Technology
[sg6165@srmist.edu.in]

---
