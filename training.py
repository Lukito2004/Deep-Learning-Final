import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import os
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from tqdm import tqdm
import pickle
import cv2
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.add_word("<PAD>")
        self.add_word("<START>")
        self.add_word("<END>")
        self.add_word("<UNK>")

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __len__(self):
        return len(self.word2idx)

    def build_vocabulary(self, caption_list):
        frequencies = Counter()

        for caption in caption_list:
            tokens = self.tokenize(caption)
            frequencies.update(tokens)

        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.add_word(word)

        print(f"Vocabulary built with {len(self)} words")
        print(f"Words appearing >= {self.freq_threshold} times")

    @staticmethod
    def tokenize(text):
        return text.lower().split()

    def numericalize(self, text):
        tokens = self.tokenize(text)

        indices = [self.word2idx["<START>"]]

        for token in tokens:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx["<UNK>"])

        indices.append(self.word2idx["<END>"])

        return indices

class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocab=None, transform=None, build_vocab=False):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(captions_file)
        print(f"Loaded {len(self.df)} image-caption pairs")

        # Precompute image paths
        self.image_paths = [
            os.path.join(self.root_dir, 'Images', img_name)
            for img_name in self.df['image']
        ]

        # Precompute captions
        self.captions = self.df['caption'].tolist()

        if build_vocab:
            self.vocab = Vocabulary(freq_threshold=5)
            self.vocab.build_vocabulary(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        caption = self.captions[idx]

        try:
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
            if self.transform:
                image = self.transform(image)

        numericalized_caption = self.vocab.numericalize(caption)

        return image, torch.tensor(numericalized_caption, dtype=torch.long)


class CustomCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        images = torch.stack(images, dim=0)

        lengths = [len(cap) for cap in captions]

        captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return images, captions, lengths

class CNNEncoder(nn.Module):
    def __init__(self, encoded_image_size=7):
        super(CNNEncoder, self).__init__()

        self.enc_image_size = encoded_image_size

        # 3 -> 64
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 224 -> 112
        )

        # 64 -> 128
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )

        # 128 -> 256
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )

        # 256 -> 512
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 28 -> 14
        )

        # 512 -> 512
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 14 -> 7
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, images):
        x = checkpoint(self.block1, images, use_reentrant=False)
        x = checkpoint(self.block2, x, use_reentrant=False)
        x = checkpoint(self.block3, x, use_reentrant=False)
        x = checkpoint(self.block4, x, use_reentrant=False)
        x = checkpoint(self.block5, x, use_reentrant=False)

        # For attention: (batch, 512, 7, 7) -> (batch, 49, 512)
        batch_size = x.size(0)
        x = x.permute(0, 2, 3, 1)  # (batch, 7, 7, 512)
        x = x.view(batch_size, -1, 512)  # (batch, 49, 512)

        return x

class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()

        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)

        self.full_att = nn.Linear(attention_dim, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.relu(att1 + att2.unsqueeze(1))
        att = self.full_att(att)
        alpha = self.softmax(att.squeeze(2))
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)

        return context, alpha

class LSTMDecoder(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size,
                 encoder_dim=512, dropout=0.25):
        super(LSTMDecoder, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        self.fc = nn.Linear(decoder_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)

        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)

        return h, c

    def forward(self, encoder_out, captions, lengths):
        batch_size = encoder_out.size(0)
        num_pixels = encoder_out.size(1)

        lengths_sorted, sort_idx = torch.sort(torch.tensor(lengths), descending=True)
        encoder_out = encoder_out[sort_idx]
        captions = captions[sort_idx]

        embeddings = self.embedding(captions)

        h, c = self.init_hidden_state(encoder_out)

        decode_lengths = (lengths_sorted - 1).tolist()
        max_length = max(decode_lengths)

        predictions = torch.zeros(batch_size, max_length, self.vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_length, num_pixels).to(device)

        for t in range(max_length):
            batch_size_t = sum([l > t for l in decode_lengths])

            context, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])

            lstm_input = torch.cat([embeddings[:batch_size_t, t, :], context], dim=1)
            h_t, c_t = self.lstm_cell(lstm_input, (h[:batch_size_t], c[:batch_size_t]))

            preds = self.fc(self.dropout_layer(h_t))

            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

            h = h_t
            c = c_t

        return predictions, alphas, sort_idx

class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, attention_dim=512, embed_dim=256,
                 decoder_dim=512, encoder_dim=512, dropout=0.25):
        super(ImageCaptioningModel, self).__init__()

        self.encoder = CNNEncoder()
        self.decoder = LSTMDecoder(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=encoder_dim,
            dropout=dropout
        )

    def forward(self, images, captions, lengths):
        encoder_out = self.encoder(images)
        predictions, alphas, sort_idx = self.decoder(encoder_out, captions, lengths)
        return predictions, alphas, sort_idx


def prepare_data(root_dir='./caption_data', batch_size=32, num_workers=4, persistent_workers=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    print("Building vocabulary...")
    full_dataset = FlickrDataset(
        root_dir=root_dir,
        captions_file=os.path.join(root_dir, 'captions.txt'),
        transform=transform,
        build_vocab=True
    )

    vocab = full_dataset.vocab

    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset split:")
    print(f"Train: {len(train_dataset)} samples")
    print(f"Val: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    pad_idx = vocab.word2idx["<PAD>"]
    collate_fn = CustomCollate(pad_idx=pad_idx)

    # Set prefetch_factor and persistent_workers based on num_workers
    dataloader_kwargs = {
        'batch_size': batch_size,
        'collate_fn': collate_fn,
        'pin_memory': True,
    }

    if num_workers > 0:
        dataloader_kwargs['num_workers'] = num_workers
        dataloader_kwargs['persistent_workers'] = persistent_workers
        dataloader_kwargs['prefetch_factor'] = 2
    else:
        dataloader_kwargs['num_workers'] = 0

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **dataloader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **dataloader_kwargs
    )

    return train_loader, val_loader, test_loader, vocab


def train_epoch(model, train_loader, criterion, optimizer, vocab, epoch, scaler):
    model.train()
    total_loss = 0
    pad_idx = vocab.word2idx["<PAD>"]

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, captions, lengths) in enumerate(pbar):
        images = images.to(device)
        captions = captions.to(device)

        optimizer.zero_grad()

        with autocast():
            predictions, alphas, sort_idx = model(images, captions, lengths)
            targets = captions[:, 1:]
            predictions = predictions.contiguous().view(-1, predictions.size(-1))
            targets = targets.contiguous().view(-1)
            loss = criterion(predictions, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})

    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, vocab):
    model.eval()
    total_loss = 0
    pad_idx = vocab.word2idx["<PAD>"]

    with torch.no_grad():
        for images, captions, lengths in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            captions = captions.to(device)

            predictions, alphas, sort_idx = model(images, captions, lengths)

            targets = captions[:, 1:]  # Remove <START>

            predictions = predictions.contiguous().view(-1, predictions.size(-1))
            targets = targets.contiguous().view(-1)

            loss = criterion(predictions, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    return avg_loss


def generate_sample_captions(model, val_loader, vocab, num_samples=5):
    model.eval()
    samples_shown = 0

    with torch.no_grad():
        for images, captions, lengths in val_loader:
            if samples_shown >= num_samples:
                break

            image = images[0].unsqueeze(0).to(device)
            caption_gt = captions[0]

            encoder_out = model.encoder(image)

            generated_caption = []
            h, c = model.decoder.init_hidden_state(encoder_out)

            current_word = torch.tensor([vocab.word2idx['<START>']]).to(device)

            max_length = 20
            for _ in range(max_length):
                word_emb = model.decoder.embedding(current_word)

                context, alpha = model.decoder.attention(encoder_out, h)

                lstm_input = torch.cat([word_emb, context], dim=1)
                h, c = model.decoder.lstm_cell(lstm_input, (h, c))

                logits = model.decoder.fc(h)
                predicted_word_idx = torch.argmax(logits, dim=1).item()

                if vocab.idx2word[predicted_word_idx] == '<END>':
                    break

                generated_caption.append(vocab.idx2word[predicted_word_idx])
                current_word = torch.tensor([predicted_word_idx]).to(device)

            gt_caption = [vocab.idx2word[idx.item()] for idx in caption_gt
                         if vocab.idx2word[idx.item()] not in ['<START>', '<END>', '<PAD>']]

            print(f"Sample {samples_shown + 1}:")
            print(f"Generated: {' '.join(generated_caption)}")
            print(f"Ground Truth: {' '.join(gt_caption)}")

            samples_shown += 1

def train_model(model, train_loader, val_loader, vocab, num_epochs=20,
                learning_rate=3e-4, save_path='best_model.pth'):
    scaler = GradScaler()
    pad_idx = vocab.word2idx["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 5  # Early stopping

    print(f"Starting training on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, num_epochs + 1):
        print()
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, vocab, epoch, scaler)
        train_losses.append(train_loss)

        val_loss = validate(model, val_loader, criterion, vocab)
        val_losses.append(val_loss)

        scheduler.step(val_loss)
        print()
        print(f"Epoch {epoch} Summary:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        print()
        print("Sample captions:")
        generate_sample_captions(model, val_loader, vocab, num_samples=3)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab': vocab
            }
            torch.save(checkpoint, save_path)
            print(f"Model saved with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{patience_limit}")

        if patience_counter >= patience_limit:
            print()
            print(f"Early stopping triggered after {epoch} epochs")
            break

    print()
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")

    return train_losses, val_losses



if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 3e-4
    EMBED_DIM = 256
    ATTENTION_DIM = 512
    DECODER_DIM = 512
    ENCODER_DIM = 512
    DROPOUT = 0.25

    print("Preparing data...")
    train_loader, val_loader, test_loader, vocab = prepare_data(
        root_dir='./caption_data',
        batch_size=BATCH_SIZE,
        num_workers=4,
        persistent_workers=True
    )
    print()
    print(f"Vocabulary size: {len(vocab)}")

    # Create model
    print()
    print("Creating model...")
    model = ImageCaptioningModel(
        vocab_size=len(vocab),
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        encoder_dim=ENCODER_DIM,
        dropout=DROPOUT
    ).to(device)

    print()
    print("Model Architecture:")
    print(model)
    print()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Train model
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        vocab=vocab,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        save_path='best_model.pth'
    )

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    print()
    print("Training complete! Model saved as 'best_model.pth'")
    print("Training curves saved as 'training_curves.png'")

    print()
    print("Saving vocabulary...")
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocabulary saved as 'vocab.pkl'")