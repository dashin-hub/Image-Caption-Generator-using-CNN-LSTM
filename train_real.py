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
        from src.dataset import ImageCaptioningDataset
        from torch.utils.data import DataLoader
        
        logging.info("Starting training process...")
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f"Using device: {device}")
        
        # Load real dataset
        train_dataset = ImageCaptioningDataset(data_dir='data', split='train')
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
        logging.info("Loaded real dataset")
        
        # Model parameters
        embed_size = 256
        hidden_size = 512
        vocab_size = len(train_dataset.word_to_idx)
        num_layers = 1
        
        # Initialize models
        encoder = ImageEncoder(embed_size).to(device)
        decoder = TransformerDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
        logging.info("Models initialized")
        
        # Training loop
        criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.word_to_idx['<PAD>'])
        encoder_optimizer = torch.optim.Adam(encoder.parameters())
        decoder_optimizer = torch.optim.Adam(decoder.parameters())
        
        logging.info("Starting training loop...")
        encoder.train()
        decoder.train()
        
        for epoch in range(1): # Just one epoch for now
            for batch_idx, (images, captions, _) in enumerate(train_loader):
                images = images.to(device)
                captions = captions.to(device)
                
                # Forward pass
                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])
                
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), captions[:, 1:].reshape(-1))
                
                # Backward pass
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()
                
                logging.info(f"Epoch {epoch+1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")
                if batch_idx > 10: # early stopping for this example
                    break

        # Save model
        os.makedirs('models', exist_ok=True)
        checkpoint = {
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'embed_size': embed_size,
            'hidden_size': hidden_size,
            'vocab_size': vocab_size,
            'word_to_idx': train_dataset.word_to_idx,
            'idx_to_word': train_dataset.idx_to_word,
        }
        torch.save(checkpoint, 'models/image_captioning_model.pth')
        logging.info("Model saved successfully")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
