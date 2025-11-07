from flask import Flask, request, render_template, jsonify
import os
import torch
from src.encoder import ImageEncoder
from src.decoder import TransformerDecoder
from src.utils import load_image, idx_to_word
from PIL import Image
import io
import base64
import logging
import torchvision.transforms as transforms

app = Flask(__name__)
app.logger.setLevel(logging.INFO)

# Global variables for model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
encoder = None
decoder = None
vocab = None

def load_model():
    """Load the trained model"""
    global model, encoder, decoder, vocab
    try:
        checkpoint = torch.load('models/image_captioning_model.pth', map_location=device)
        vocab = torch.load('data/dummy/vocab.pth')
        encoder = ImageEncoder(checkpoint['embed_size']).to(device)
        decoder = TransformerDecoder(
            checkpoint['embed_size'],
            checkpoint['hidden_size'],
            checkpoint['vocab_size'],
            1  # num_layers, assuming 1 from train_simple.py
        ).to(device)

        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])

        encoder.eval()
        decoder.eval()
        app.logger.info("Model loaded successfully")
    except Exception as e:
        app.logger.error(f"Error loading model: {str(e)}")
        raise

def preprocess_image(img):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    return transform(img)

def decode_caption(caption_indices):
    """Convert caption indices to words"""
    words = []
    for idx in caption_indices:
        word = idx_to_word(idx.item(), vocab)
        if word == '<END>':
            break
        if word not in ['<START>', '<PAD>']:
            words.append(word)
    return ' '.join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get image from request
            file = request.files['file']
            img = Image.open(file.stream)
            
            # Process image
            img = preprocess_image(img)
            
            # Generate caption
            with torch.no_grad():
                features = encoder(img.unsqueeze(0).to(device))
                caption_indices = decoder.generate(features)
                
            # Convert indices to words
            caption = decode_caption(caption_indices[0])
            
            # Convert image to base64 for display
            buffered = io.BytesIO()
            Image.open(file.stream).save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'caption': caption,
                'image': f'data:image/jpeg;base64,{img_str}'
            })
            
        except Exception as e:
            app.logger.error(f"Error processing request: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            })

def main():
    # Create upload directory if it doesn't exist
    os.makedirs('static/uploads', exist_ok=True)
    
    # Load model
    load_model()
    
    # Run app
    app.run(debug=True, port=5000)

if __name__ == '__main__':
    main()