from glob import glob
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
import soundfile as sf
import numpy as np
import random
log = print
# -----------------------------
# Dataset для пар аудио
# -----------------------------
class AudioPairsDataset(Dataset):
    def __init__(self, file_paths, sr=11025, duration:int=10, n_mels:int=128):
        self.files = file_paths
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels

    def __len__(self):
        return 1000  # число пар на эпоху

    def __getitem__(self, idx):
        # Рандомно выбираем: одна песня (0) или разные (1)
        if random.random() < 0.5:
            file = random.choice(self.files)
            label = 0
            x1 = self.process_file(file, augment=False)
            x2 = self.process_file(file, augment=True)
        else:
            file1, file2 = random.sample(self.files, 2)
            label = 1
            x1 = self.process_file(file1, augment=True)
            x2 = self.process_file(file2, augment=True)

        return x1, x2, torch.tensor(label, dtype=torch.float32)

    def process_file(self, file, augment=True):
        y, _ = librosa.load(file, sr=self.sr, mono=True)

        # Берём чуть длиннее target_len
        crop_len = self.duration + 1
        crop_samples = crop_len * self.sr
        y = y[:crop_samples]
        if len(y) < crop_samples:
            y = np.pad(y, (0, crop_samples - len(y)))

        if augment:
            y = self.random_time_stretch(y, 0.95, 1.05)
            y = self.add_noise(y, snr_db=20)
            y = self.random_volume(y, min_db=-10, max_db=10)
        # Обрезаем до target_len
        target_samples = self.duration * self.sr
        y = y[:target_samples]
        if len(y) < target_samples:
            y = np.pad(y, (0, target_samples - len(y)))

        # log-Mel спектрограмма
        mel = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels)
        log_mel = librosa.power_to_db(mel)
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
        log_mel = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
        return log_mel

    @staticmethod
    def random_time_stretch(y, min_rate, max_rate):
        rate = np.random.uniform(min_rate, max_rate)
        return librosa.effects.time_stretch(y=y, rate=rate)

    @staticmethod
    def add_noise(y, snr_db):
        rms_signal = np.sqrt(np.mean(y**2))
        snr = 10**(snr_db / 10)
        rms_noise = rms_signal / np.sqrt(snr)
        noise = np.random.randn(len(y)) * rms_noise
        return y + noise
    @staticmethod
    def random_volume(y, min_db=-10, max_db=10, prevent_clipping=True):
        db_change = np.random.uniform(min_db, max_db)
        gain = 10 ** (db_change / 20)   
        y_augmented = y * gain  
        if prevent_clipping:
            max_val = np.max(np.abs(y_augmented))
            if max_val > 1.0:
                y_augmented = y_augmented / max_val * 0.95  
        
        return y_augmented
# -----------------------------
# CNN → embedding
# -----------------------------
class AudioEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -----------------------------
# Contrastive loss
# -----------------------------
def contrastive_loss(emb1, emb2, labels, margin=1.0):
    distance = F.pairwise_distance(emb1, emb2)
    loss = ((1 - labels) * distance**2 + labels * torch.clamp(margin - distance, min=0)**2).mean()
    return loss

# -----------------------------
# Пример тренировки на CPU
# -----------------------------
import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

def train_model(
    model,
    loader,
    optimizer,
    device,
    epochs,
    start_epoch,
    threshold,
    model_path
):
    """Training loop"""
    for epoch in range(start_epoch, epochs):
        model.train()
        total_correct = 0
        total_samples = 0

        for batch_idx, (x1, x2, labels) in enumerate(loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            emb1 = model(x1)
            emb2 = model(x2)

            distance = F.pairwise_distance(emb1, emb2)
            loss = ((1 - labels) * distance**2 + labels * torch.clamp(1.0 - distance, min=0)**2).mean()
            loss.backward()
            optimizer.step()
            preds = (distance >= threshold).float()
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            log(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, "
                  f"Acc: {total_correct/total_samples:.4f}")

        torch.save(model.state_dict(), model_path)
        log(f"Epoch {epoch} finished, model saved to {model_path}")

def validate_model(model, loader, device, threshold):
    """Validation loop to evaluate model performance"""
    model.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (x1, x2, labels) in enumerate(loader):
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)
            
            emb1 = model(x1)
            emb2 = model(x2)
            
            distance = F.pairwise_distance(emb1, emb2)
            preds = (distance >= threshold).float()
            
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            
            if batch_idx % 10 == 0:
                log(f"Validation batch {batch_idx}, Current accuracy: {total_correct/total_samples:.4f}")
    
    accuracy = total_correct / total_samples
    log(f"\n=== Validation Results ===")
    log(f"Total samples: {total_samples}")
    log(f"Correct predictions: {total_correct}")
    log(f"Accuracy: {accuracy:.4f}")
    return accuracy

def setup_dataloader(folder, batch_size, num_workers):
    """Setup dataset and dataloader"""
    files = glob.glob(os.path.join(folder, "*.wav"))
    log(f"Found {len(files)} files")
    dataset = AudioPairsDataset(files)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader

def setup_model(device, model_path, resume_training):
    """Setup model and load weights if requested"""
    model = AudioEmbeddingNet(embedding_dim=128).to(device)
    start_epoch = 0
    
    if resume_training and os.path.exists(model_path):
        log(f"Loading model from {model_path}...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            log("Model loaded successfully!")
        except Exception as e:
            log(f"Error while loading model: {e}")
            log("Starting from scratch...")
    else:
        log("Starting from scratch.")
    
    return model, start_epoch

def get_user_choice():
    """Get user input for training/validation and number of threads"""
    log("\n=== Audio Siamese Network ===")
    log("Choose mode:")
    log("  1 - Train model")
    log("  2 - Validate model")
    log("  3 - Train and then validate")
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        if choice in ['1', '2', '3']:
            break
        log("Invalid choice. Please enter 1, 2, or 3.")
    
    while True:
        try:
            num_workers = int(input("Enter number of worker threads: "))
            if num_workers > 0:
                break
            log("Please enter a positive number.")
        except ValueError:
            log("Please enter a valid number.")
    
    return int(choice), num_workers

if __name__ == "__main__":
    # Configuration
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    folder = os.path.join(BASE_DIR, "..", "data", "resampled_split", "training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.version.cuda)
    epochs = 100
    batch_size = 8
    threshold = 0.5
    model_path = os.path.join(BASE_DIR,"siamese_audio.pth")
    resume_training = True
    
    # Get user input
    choice, num_workers = get_user_choice()
    
    # Setup dataloader
    loader = setup_dataloader(folder, batch_size, num_workers)
    
    # Setup model
    model, start_epoch = setup_model(device, model_path, resume_training)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Execute user choice
    if choice == 1:  # Train only
        log("\n=== Starting Training ===")
        train_model(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            start_epoch=start_epoch,
            threshold=threshold,
            model_path=model_path
        )
        log("\n=== Training completed! ===")
        
    elif choice == 2:  # Validate only
        log("\n=== Starting Validation ===")
        if not os.path.exists(model_path):
            log(f"Error: Model file {model_path} not found. Please train the model first.")
        else:
            accuracy = validate_model(model, loader, device, threshold)
            log(f"\nValidation completed. Final accuracy: {accuracy:.4f}")
            
    elif choice == 3:  # Train then validate
        log("\n=== Starting Training ===")
        train_model(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            epochs=epochs,
            start_epoch=start_epoch,
            threshold=threshold,
            model_path=model_path
        )
        log("\n=== Training completed! Starting Validation ===")
        accuracy = validate_model(model, loader, device, threshold)
        log(f"\nFinal accuracy after training: {accuracy:.4f}")