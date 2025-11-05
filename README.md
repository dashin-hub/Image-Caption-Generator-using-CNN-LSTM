# Image-Caption-Generator-using-CNN-LSTM
Deep learning project that generates natural-language captions for images using CNN for visual feature extraction and LSTM for text generation. Built with TensorFlow/Keras and trained on the MS COCO dataset for AI-powered image understanding.
Perfect 👍 — here’s a **ready-to-copy-paste `README.md`** for your **Image Caption Generator using CNN + LSTM** project.
It’s clean, professional, and formatted for **GitHub** — you can paste it directly into your repo and it will render beautifully.

---

````markdown
---

## 🎯 **Project Overview**

The Image Caption Generator is an AI model that takes an image as input and outputs a text caption that describes the image content.  
It replicates how platforms like **Google Photos** or **Pinterest** automatically describe uploaded images.

---

## ⚙️ **Tech Stack**

- **Frameworks:** TensorFlow, Keras  
- **Languages:** Python  
- **Models Used:** CNN (InceptionV3 / VGG16 / ResNet50), LSTM  
- **Dataset:** MS COCO  
- **Libraries:** NumPy, Pandas, Matplotlib, Pillow, OpenCV, NLTK  
- **Evaluation Metric:** BLEU Score  

---

## 🧩 **Architecture**

The system follows an **Encoder–Decoder** architecture:

1. **CNN Encoder:**  
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
4. Evaluation:Measure BLEU score to compare predicted and ground-truth captions.  
5. Inference: Generate captions for unseen images using the trained model.

---

## 🖼️ Example Output

Input: 🐶 Dog playing in the park  
Generated Caption: `"A brown dog playing with a ball in the grass."`

---

## 💻 **Installation & Setup**

1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/Image-Caption-Generator.git
cd Image-Caption-Generator
````

2. Install Dependencies

```bash
pip install -r requirements.txt
```

3. Download Dataset

Download the [MS COCO dataset](https://cocodataset.org/#home) and place it inside the `/data` folder.

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

---


```
 Input Image: <img width="1907" height="968" alt="image" src="https://github.com/user-attachments/assets/af9d5d2f-7ab6-4124-917d-cf5b1ad74348" />

Generated Caption: <img width="477" height="761" alt="image" src="https://github.com/user-attachments/assets/32413d04-3ef6-4fb6-a031-7c7a637ee5ed" />

```

---

📊 Evaluation Metric

Model performance is evaluated using **BLEU (Bilingual Evaluation Understudy)** score — comparing generated captions with human-written captions for accuracy and fluency.

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

 

 🧠 Expected Outcome

✅ Generates accurate and meaningful captions for unseen images
✅ Demonstrates integration of Computer Vision and NLP
✅ Provides strong foundation in deep learning encoder–decoder models
✅ Optional GUI for real-time captioning

---

📚 References

[MS COCO Dataset](https://cocodataset.org/#home)
[Show and Tell: A Neural Image Caption Generator (Google Research)](https://arxiv.org/abs/1411.4555)
[TensorFlow Documentation](https://www.tensorflow.org/)
[Keras API Reference](https://keras.io/api/)



👤 Author

SREEDHARSHAN G J
🎓 B.Tech ECE | SRM Institute of Science and Technology
📧 [sg6165@srmist.edu.in]


