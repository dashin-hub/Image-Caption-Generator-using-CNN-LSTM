



# 🧠 Image Caption Generator (CNN + LSTM)

Automatically generate descriptive captions for images using deep learning — combining a CNN encoder (InceptionV3 / VGG16 / ResNet50) with an LSTM decoder. Built with TensorFlow / Keras and trained on MS COCO, this project demonstrates an end-to-end encoder–decoder approach to image captioning.

---

## ⚡ Project Overview

This repo implements an encoder–decoder architecture that converts images into human-like captions. It’s suitable for demos, course projects, research baselines, and as a starting point for production systems (with further optimization).

Key features:

* Pretrained CNN for robust image feature extraction
* LSTM decoder with sequence modeling and teacher forcing training
* Tokenization, caption cleaning, and vocabulary management
* BLEU score evaluation and sample inference pipeline
* Optional attention mechanism extension (commented / experimental)

---

## 🧩 Why this project is useful

* Great learning resource for CV + NLP fusion
* Lightweight baseline for rapid experimentation
* Useful demo for portfolios, talks, and university projects

---

## 🧠 Architecture


1. **CNN Encoder**

   * Pretrained model (InceptionV3 / ResNet50 / VGG16)
   * Remove final classification layers and extract a dense visual feature vector (or spatial feature map for attention)

2. **Text Preprocessor**

   * Clean captions, lowercase, remove punctuation
   * Tokenize with Keras Tokenizer
   * Generate sequences and padding; build word-index mappings

3. **LSTM Decoder**

   * Sequence input (previous words) + image features
   * Embedding layer → LSTM → Dense (softmax) over vocabulary
   * Trained with categorical cross-entropy and teacher forcing

4. **(Optional) Attention**

   * Use spatial features + attention layer to attend to image regions during decoding

---

## 📁 Project Structure

```
Image-Caption-Generator/
├── data/                      # COCO subsets, preprocessed captions, tokenizers
├── models/                    # Trained weights / checkpoints
├── notebooks/                 # EDA and training notebooks
├── src/
│   ├── encoder.py             # CNN feature extractor
│   ├── decoder.py             # LSTM / attention decoder
│   ├── train.py               # Training loop
│   ├── inference.py           # Caption generation script
│   ├── utils.py               # helpers: tokenizers, loaders, metrics
│   └── evaluate.py            # BLEU / evaluation utilities
├── static/                    # logos, example images
│   └── image/
├── example_output.jpg         # example generated image (placeholder)
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Image-Caption-Generator.git
cd Image-Caption-Generator
```

### 2. Create environment & install

#### Using conda

```bash
conda create -n imgcap python=3.11 -y
conda activate imgcap
pip install -r requirements.txt
```

#### Using venv

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Prepare data (MS COCO)

* Download COCO images and annotations (or use a small subset for quick tests).
* Run preprocessing to build caption tokens and features:

```bash
python src/utils.py --prepare-coco --annotations /path/to/annotations --images /path/to/images
```

### 4. Train (small run)

```bash
python src/train.py --config configs/train_small.yaml
```

### 5. Evaluate

```bash
python src/evaluate.py --model models/last_checkpoint.h5 --data data/val_annotations.json
```

### 6. Inference / Generate a caption

```bash
python src/inference.py --image static/example.jpg --model models/last_checkpoint.h5
```

---

## 📊 Results & Evaluation

* **Dataset:** MS COCO (subset / full)
* **Evaluation Metric:** BLEU-1..4, METEOR (optional)
* **Typical baseline BLEU-4:** 0.25–0.45 (varies with training, beam size, attention)

> Note: These numbers are illustrative — report your real scores after training.

---

## 🛠️ Experiments & Tips

* Use **feature caching** (store CNN features) for faster training iterations.
* Start with a **small vocabulary** (top N words) then expand.
* Experiment with **beam search** (beam width 3–5) during inference for better captions.
* Add **attention** (Bahdanau / Luong) for improved localization-to-word mapping.
* Fine-tune the backbone CNN (careful: needs more GPU/RAM).

---

## 🔮 Future Work

* Integrate Transformer-based decoders (e.g., Vision Transformer + Transformer decoder).
* Add multilingual captioning and BLEU/CIDEr evaluation.
* Deploy as an API with FastAPI / Flask and a simple web demo.
* Quantize / prune models for mobile inference.

---

## ⚠️ Disclaimer

This project is for educational and research purposes only. Respect dataset terms (MS COCO) and licenses. Do not use this system for disallowed or infringing activities.

___

## 👨‍💻 Author & Contact

Sreedharshan G J
Electronics & Communication Engineering — SRM Institute of Science and Technology
📧 [sg6165@srmist.edu.in](mailto:your.email@example.com)

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/Awesome`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push and open a PR

Be sure to follow PEP8, add tests for new features, and update docs.

---

