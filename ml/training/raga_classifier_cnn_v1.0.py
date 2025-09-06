#!/usr/bin/env python3
"""
Honest Raga Classification System
A simple, working CNN-based classifier for Indian classical raga detection
No fake marketing, no misleading claims - just what actually works
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RagaDataset(Dataset):
    """Simple dataset class for raga classification"""
    
    def __init__(self, audio_files: List[str], labels: List[int], sr: int = 22050, duration: int = 10):
        self.audio_files = audio_files
        self.labels = labels
        self.sr = sr
        self.duration = duration
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            y, sr = librosa.load(audio_file, sr=self.sr, duration=self.duration)
            
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Ensure consistent dimensions (128, 431) for 10 seconds at 22050 Hz
            target_width = 431
            if mel_spec_db.shape[1] > target_width:
                mel_spec_db = mel_spec_db[:, :target_width]
            elif mel_spec_db.shape[1] < target_width:
                # Pad with zeros
                pad_width = target_width - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            
            # Convert to tensor
            mel_spec_tensor = torch.FloatTensor(mel_spec_db)
            
            return mel_spec_tensor, torch.LongTensor([label])
            
        except Exception as e:
            logger.error(f"Error loading {audio_file}: {e}")
            # Return zero tensor if loading fails
            return torch.zeros(128, 431), torch.LongTensor([0])

class SimpleRagaCNN(nn.Module):
    """Simple CNN for raga classification - honest and straightforward"""
    
    def __init__(self, num_classes: int = 2, input_height: int = 128):
        super(SimpleRagaCNN, self).__init__()
        
        # Simple CNN architecture
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate the size after convolutions and pooling
        # Input: (1, 128, 431) -> After 3 conv layers with pooling: (128, 16, 54)
        # But let's make it adaptive
        self.fc1 = nn.Linear(128 * 16 * 54, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Add channel dimension if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, height, width)
        
        # Convolutional layers
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Adaptive fully connected layer
        if not hasattr(self, 'fc1_adaptive') or self.fc1_adaptive.in_features != x.size(1):
            self.fc1_adaptive = nn.Linear(x.size(1), 512).to(x.device)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1_adaptive(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class HonestRagaClassifier:
    """Honest raga classifier - no fake claims, just what works"""
    
    def __init__(self, device: str = "auto"):
        # Set device honestly
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using Mac GPU (MPS)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.class_names = ['Carnatic', 'Hindustani']
        
        logger.info(f"Honest Raga Classifier initialized on {self.device}")
    
    def prepare_dataset(self, dataset_path: str, max_samples: int = 1000) -> Tuple[List[str], List[int]]:
        """Prepare dataset - honest about what we can actually process"""
        logger.info(f"Preparing dataset from {dataset_path}")
        
        audio_files = []
        labels = []
        
        # Find audio files
        carnatic_files = list(Path(dataset_path).glob("Carnatic/**/*.wav")) + \
                        list(Path(dataset_path).glob("Carnatic/**/*.mp3"))
        hindustani_files = list(Path(dataset_path).glob("Hindustani/**/*.wav")) + \
                          list(Path(dataset_path).glob("Hindustani/**/*.mp3"))
        
        # Sample files honestly
        carnatic_files = carnatic_files[:max_samples//2]
        hindustani_files = hindustani_files[:max_samples//2]
        
        # Add Carnatic files (label 0)
        for file_path in carnatic_files:
            audio_files.append(str(file_path))
            labels.append(0)  # Carnatic
        
        # Add Hindustani files (label 1)
        for file_path in hindustani_files:
            audio_files.append(str(file_path))
            labels.append(1)  # Hindustani
        
        logger.info(f"Dataset prepared: {len(audio_files)} files")
        logger.info(f"Carnatic: {labels.count(0)}, Hindustani: {labels.count(1)}")
        
        return audio_files, labels
    
    def train(self, dataset_path: str, epochs: int = 10, batch_size: int = 8, learning_rate: float = 0.001):
        """Train the model - honest about what we're doing"""
        logger.info("Starting honest training...")
        
        # Prepare dataset
        audio_files, labels = self.prepare_dataset(dataset_path)
        
        if len(audio_files) == 0:
            logger.error("No audio files found!")
            return
        
        # Split dataset
        train_files, val_files, train_labels, val_labels = train_test_split(
            audio_files, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Create datasets
        train_dataset = RagaDataset(train_files, train_labels)
        val_dataset = RagaDataset(val_files, val_labels)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize model
        self.model = SimpleRagaCNN(num_classes=2).to(self.device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # Training loop
        best_val_accuracy = 0.0
        train_losses = []
        val_accuracies = []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                train_total += target.size(0)
                train_correct += (predicted == target).sum().item()
                
                if batch_idx % 5 == 0:
                    logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Validation
            self.model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    target = target.squeeze()
                    
                    output = self.model(data)
                    _, predicted = torch.max(output.data, 1)
                    val_total += target.size(0)
                    val_correct += (predicted == target).sum().item()
            
            # Calculate metrics
            train_accuracy = 100 * train_correct / train_total
            val_accuracy = 100 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch+1} Results:")
            logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
            logger.info(f"  Val Accuracy: {val_accuracy:.2f}%")
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                self.save_model("honest_raga_model.pt")
                logger.info(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
            
            train_losses.append(avg_train_loss)
            val_accuracies.append(val_accuracy)
            
            scheduler.step()
        
        logger.info(f"Training complete! Best validation accuracy: {best_val_accuracy:.2f}%")
        
        # Plot training curves
        self._plot_training_curves(train_losses, val_accuracies)
        
        return best_val_accuracy
    
    def evaluate(self, dataset_path: str) -> Dict:
        """Evaluate the model honestly"""
        logger.info("Evaluating model...")
        
        if self.model is None:
            logger.error("Model not trained yet!")
            return {}
        
        # Prepare test dataset
        audio_files, labels = self.prepare_dataset(dataset_path)
        
        # Create test dataset
        test_dataset = RagaDataset(audio_files, labels)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        # Evaluation
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, target_names=self.class_names, output_dict=True)
        cm = confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': [int(p) for p in all_predictions],
            'true_labels': [int(l) for l in all_labels]
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(all_labels, all_predictions, target_names=self.class_names)}")
        
        # Plot confusion matrix
        self._plot_confusion_matrix(cm)
        
        return results
    
    def predict(self, audio_file: str) -> Dict:
        """Predict raga for a single audio file"""
        if self.model is None:
            logger.error("Model not trained yet!")
            return {'error': 'Model not trained'}
        
        try:
            # Load and preprocess audio
            y, sr = librosa.load(audio_file, sr=22050, duration=10)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Ensure consistent dimensions
            target_width = 431
            if mel_spec_db.shape[1] > target_width:
                mel_spec_db = mel_spec_db[:, :target_width]
            elif mel_spec_db.shape[1] < target_width:
                pad_width = target_width - mel_spec_db.shape[1]
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            
            mel_spec_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0).to(self.device)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                output = self.model(mel_spec_tensor)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)
            
            prediction = predicted.item()
            confidence = probabilities[0][prediction].item()
            
            return {
                'audio_file': audio_file,
                'predicted_tradition': self.class_names[prediction],
                'confidence': confidence,
                'probabilities': {
                    'Carnatic': probabilities[0][0].item(),
                    'Hindustani': probabilities[0][1].item()
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting {audio_file}: {e}")
            return {'error': str(e)}
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'class_names': self.class_names
            }, filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model = SimpleRagaCNN(num_classes=2).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.class_names = checkpoint['class_names']
            logger.info(f"Model loaded from {filepath}")
        else:
            logger.error(f"Model file not found: {filepath}")
    
    def _plot_training_curves(self, train_losses: List[float], val_accuracies: List[float]):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Training loss
        ax1.plot(train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Validation accuracy
        ax2.plot(val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Training curves saved to training_curves.png")
    
    def _plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        logger.info("Confusion matrix saved to confusion_matrix.png")

def main():
    """Main function - honest about what we're doing"""
    logger.info("🚀 Starting Honest Raga Classification System...")
    
    # Initialize classifier
    classifier = HonestRagaClassifier()
    
    # Train model
    dataset_path = "../../carnatic-hindustani-dataset"
    if Path(dataset_path).exists():
        logger.info(f"Training on dataset: {dataset_path}")
        
        # Train with honest parameters
        best_accuracy = classifier.train(
            dataset_path=dataset_path,
            epochs=10,
            batch_size=8,
            learning_rate=0.001
        )
        
        logger.info(f"Training complete! Best accuracy: {best_accuracy:.2f}%")
        
        # Evaluate model
        results = classifier.evaluate(dataset_path)
        
        # Save results
        with open('honest_evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Results saved to honest_evaluation_results.json")
        
    else:
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please ensure the dataset is available")

if __name__ == "__main__":
    main()
