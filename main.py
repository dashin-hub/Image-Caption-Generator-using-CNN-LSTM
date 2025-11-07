#!/usr/bin/env python
import argparse
from PIL import Image
import torch
from src.encoder import ImageEncoder
from src.decoder import TransformerDecoder
from src.utils import load_image, process_image

def main():
    parser = argparse.ArgumentParser(description='Image Caption Generator')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, default='models/image_captioning_model.pth', help='Path to model checkpoint')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    try:
        checkpoint = torch.load(args.model, map_location=device)
        encoder = ImageEncoder(checkpoint['embed_size']).to(device)
        decoder = TransformerDecoder(
            checkpoint['embed_size'],
            checkpoint['hidden_size'],
            checkpoint['vocab_size']
        ).to(device)
        
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print("Model loaded successfully")
    except FileNotFoundError:
        print(f"Model file not found at {args.model}")
        print("Please train the model first using train.py")
        return
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # Load and process image
    try:
        image = load_image(args.image)
        image = process_image(image).to(device)
        print("Image processed successfully")
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return

    # Generate caption
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        features = encoder(image.unsqueeze(0))
        outputs = decoder.generate(features)
        caption = ' '.join([checkpoint.get('idx_to_word', {}).get(idx.item(), '<UNK>') 
                          for idx in outputs[0]])
        print(f"\nGenerated caption: {caption}")

if __name__ == '__main__':
    main()