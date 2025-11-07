import sys
import os
import logging
import torch
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def main():
    try:
        # Add the current directory to Python path
        sys.path.append(os.getcwd())
        
        from src.encoder import ImageEncoder
        from src.decoder import TransformerDecoder
        from src.dummy_data import create_dummy_data
        from torch.utils.data import DataLoader
        
        logging.info("Starting training process...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Create dummy dataset
        train_dataset = create_dummy_data(size=10)
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        logging.info("Created dummy dataset")
        
        # Initialize models
        encoder = ImageEncoder(256).to(device)
        decoder = TransformerDecoder(256, 512, 1000, 1).to(device)
        logging.info("Models initialized")
        
        # Training loop
        criterion = torch.nn.CrossEntropyLoss()
        encoder_optimizer = torch.optim.Adam(encoder.parameters())
        decoder_optimizer = torch.optim.Adam(decoder.parameters())
        
        logging.info("Starting training loop...")
        encoder.train()
        decoder.train()
        
        for batch_idx, (images, captions) in enumerate(train_loader):
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions)
            
            loss = criterion(outputs.view(-1, outputs.shape[-1]), captions.view(-1))
            
            # Backward pass
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            logging.info(f"Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
        
        # Save model
        os.makedirs('models', exist_ok=True)
        checkpoint = {
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'embed_size': 256,
            'hidden_size': 512,
            'vocab_size': 1000
        }
        torch.save(checkpoint, 'models/image_captioning_model.pth')
        logging.info("Model saved successfully")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()