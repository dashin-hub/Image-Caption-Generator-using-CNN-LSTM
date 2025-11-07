import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import logging
import sys
from src.encoder import ImageEncoder
from src.decoder import TransformerDecoder

# Setup logging
logging.basicConfig(level=logging.INFO)

def load_image(image_path):
    """Load and preprocess image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image

def main():
    parser = argparse.ArgumentParser(description='Generate caption for an image')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--model', default='models/image_captioning_model.pth',
                      help='Path to model checkpoint')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    try:
        # Load model
        checkpoint = torch.load(args.model)
        encoder = ImageEncoder(checkpoint['embed_size']).to(device)
        decoder = TransformerDecoder(
            checkpoint['embed_size'],
            checkpoint['hidden_size'],
            checkpoint['vocab_size'],
            num_layers=1
        ).to(device)
        
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        logging.info("Model loaded successfully")
        
        # Load and process image
        image = load_image(args.image).to(device)
        logging.info("Image processed")
        
        # Generate caption
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            features = encoder(image.unsqueeze(0))
            caption_indices = decoder.generate(features)
            
        # Convert indices to words
        idx_to_word = checkpoint.get('idx_to_word', {})
        caption_words = [idx_to_word.get(idx.item(), '<UNK>') 
                        for idx in caption_indices[0]]
        caption = ' '.join(word for word in caption_words 
                          if word not in ['<PAD>', '<START>', '<END>'])
        
        print("\nGenerated caption:", caption)
        
    except Exception as e:
        logging.error(f"Error during inference: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()