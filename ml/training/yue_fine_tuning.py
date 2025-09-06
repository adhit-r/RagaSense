#!/usr/bin/env python3
"""
YuE Fine-tuning Script for RagaSense
Fine-tune the YuE foundation model specifically for Indian classical raga classification
"""

import os
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import librosa
import soundfile as sf
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.pytorch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RagaDataset(Dataset):
    """Dataset class for raga classification with YuE"""
    
    def __init__(self, audio_files: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.audio_files = audio_files
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        # Extract audio features
        features = self._extract_audio_features(audio_file)
        
        # Create text prompt for YuE
        prompt = self._create_raga_prompt(features)
        
        # Tokenize prompt
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _extract_audio_features(self, audio_file: str) -> Dict:
        """Extract audio features for prompt creation"""
        try:
            y, sr = librosa.load(audio_file, sr=22050)
            
            # Basic features
            tempo = librosa.beat.tempo(y=y, sr=sr)[0]
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            
            # Rhythm features
            onset_strength = librosa.onset.onset_strength(y=y, sr=sr)
            rhythm_regularity = 1.0 / (1.0 + np.std(onset_strength) / (np.mean(onset_strength) + 1e-8))
            
            # Pitch features
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, threshold=0.1)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            tonic = np.median(pitch_values) if pitch_values else 0.0
            
            return {
                'tempo': tempo,
                'spectral_centroid': spectral_centroid,
                'rhythm_regularity': rhythm_regularity,
                'tonic': tonic,
                'duration': len(y) / sr
            }
        except Exception as e:
            logger.error(f"Error extracting features from {audio_file}: {e}")
            return {
                'tempo': 0.0,
                'spectral_centroid': 0.0,
                'rhythm_regularity': 0.0,
                'tonic': 0.0,
                'duration': 0.0
            }
    
    def _create_raga_prompt(self, features: Dict) -> str:
        """Create optimized prompt for YuE raga classification"""
        tempo = features.get('tempo', 0)
        tonic = features.get('tonic', 0)
        rhythm_regularity = features.get('rhythm_regularity', 0)
        duration = features.get('duration', 0)
        
        prompt = f"""
        Analyze this Indian classical music audio:
        - Tempo: {tempo:.1f} BPM
        - Tonic frequency: {tonic:.1f} Hz
        - Rhythm regularity: {rhythm_regularity:.2f}
        - Duration: {duration:.1f} seconds
        
        Classify the raga and tradition (Carnatic or Hindustani).
        Consider melodic patterns, scale structure, and musical characteristics.
        
        The audio contains Indian classical music. Identify the specific raga.
        """
        
        return prompt

class YuEFineTuner:
    """Fine-tune YuE model for raga classification"""
    
    def __init__(self, model_name: str = "m-a-p/YuE-s1-7B-anneal-en-icl"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Add classification head
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)  # Carnatic vs Hindustani
        self.model.to(self.device)
        self.classifier.to(self.device)
        
        # Raga database
        self.raga_database = self._load_raga_database()
        
        logger.info(f"YuE Fine-tuner initialized on {self.device}")
    
    def _load_raga_database(self) -> Dict:
        """Load raga database for classification"""
        return {
            'carnatic_ragas': [
                'anandabhairavi', 'kalyani', 'bhairavi', 'kambhoji', 'sankarabharanam',
                'mohana', 'hindolam', 'madhuvanti', 'shankarabharanam', 'kharaharapriya'
            ],
            'hindustani_ragas': [
                'yaman', 'bageshri', 'kafi', 'bhairavi', 'khamaj',
                'bhairav', 'bilaval', 'kalyani', 'marwa', 'purvi'
            ]
        }
    
    def prepare_dataset(self, dataset_path: str, max_samples: int = 1000) -> Tuple[List[str], List[int]]:
        """Prepare dataset for fine-tuning"""
        logger.info(f"Preparing dataset from {dataset_path}")
        
        audio_files = []
        labels = []
        
        # Find audio files
        carnatic_files = list(Path(dataset_path).glob("Carnatic/**/*.wav")) + \
                        list(Path(dataset_path).glob("Carnatic/**/*.mp3"))
        hindustani_files = list(Path(dataset_path).glob("Hindustani/**/*.wav")) + \
                          list(Path(dataset_path).glob("Hindustani/**/*.mp3"))
        
        # Sample files
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
    
    def fine_tune(self, dataset_path: str, output_dir: str = "yue_fine_tuned", 
                  num_epochs: int = 3, batch_size: int = 4, learning_rate: float = 5e-5):
        """Fine-tune YuE model for raga classification"""
        
        # Setup MLflow
        mlflow.set_experiment("YuE_Raga_Classification")
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            
            # Prepare dataset
            audio_files, labels = self.prepare_dataset(dataset_path)
            
            # Split dataset
            train_files, val_files, train_labels, val_labels = train_test_split(
                audio_files, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            # Create datasets
            train_dataset = RagaDataset(train_files, train_labels, self.tokenizer)
            val_dataset = RagaDataset(val_files, val_labels, self.tokenizer)
            
            # Create data loaders
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Setup training
            optimizer = optim.AdamW([
                {'params': self.model.parameters(), 'lr': learning_rate},
                {'params': self.classifier.parameters(), 'lr': learning_rate * 10}
            ])
            
            criterion = nn.CrossEntropyLoss()
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            
            # Training loop
            best_val_accuracy = 0.0
            
            for epoch in range(num_epochs):
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                
                # Training
                self.model.train()
                self.classifier.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_idx, batch in enumerate(train_loader):
                    optimizer.zero_grad()
                    
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    # Forward pass
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_output = outputs.last_hidden_state.mean(dim=1)  # Global average pooling
                    logits = self.classifier(pooled_output)
                    
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(logits.data, 1)
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
                    
                    if batch_idx % 10 == 0:
                        logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                # Validation
                self.model.eval()
                self.classifier.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(self.device)
                        attention_mask = batch['attention_mask'].to(self.device)
                        labels = batch['labels'].to(self.device)
                        
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                        pooled_output = outputs.last_hidden_state.mean(dim=1)
                        logits = self.classifier(pooled_output)
                        
                        loss = criterion(logits, labels)
                        val_loss += loss.item()
                        
                        _, predicted = torch.max(logits.data, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                # Calculate metrics
                train_accuracy = 100 * train_correct / train_total
                val_accuracy = 100 * val_correct / val_total
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                
                logger.info(f"Epoch {epoch+1} Results:")
                logger.info(f"  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
                logger.info(f"  Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
                
                # Log metrics to MLflow
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_accuracy, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                
                # Save best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    self.save_model(output_dir)
                    logger.info(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
                
                scheduler.step()
            
            # Log final metrics
            mlflow.log_metric("best_val_accuracy", best_val_accuracy)
            
            logger.info(f"Fine-tuning complete! Best validation accuracy: {best_val_accuracy:.2f}%")
            
            return best_val_accuracy
    
    def save_model(self, output_dir: str):
        """Save fine-tuned model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save classifier
        torch.save(self.classifier.state_dict(), os.path.join(output_dir, "classifier.pt"))
        
        logger.info(f"Model saved to {output_dir}")
    
    def load_model(self, model_dir: str):
        """Load fine-tuned model"""
        self.model = AutoModel.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Load classifier
        classifier_path = os.path.join(model_dir, "classifier.pt")
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path))
        
        self.model.to(self.device)
        self.classifier.to(self.device)
        
        logger.info(f"Model loaded from {model_dir}")
    
    def evaluate(self, dataset_path: str) -> Dict:
        """Evaluate fine-tuned model"""
        logger.info("Evaluating fine-tuned YuE model...")
        
        # Prepare test dataset
        audio_files, labels = self.prepare_dataset(dataset_path, max_samples=200)
        
        # Create test dataset
        test_dataset = RagaDataset(audio_files, labels, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
        
        # Evaluation
        self.model.eval()
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.last_hidden_state.mean(dim=1)
                logits = self.classifier(pooled_output)
                
                _, predicted = torch.max(logits.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        
        # Classification report
        class_names = ['Carnatic', 'Hindustani']
        report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'predictions': all_predictions,
            'true_labels': all_labels
        }
        
        logger.info(f"Evaluation Results:")
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Classification Report:\n{classification_report(all_labels, all_predictions, target_names=class_names)}")
        
        return results

def main():
    """Main function for YuE fine-tuning"""
    logger.info("🚀 Starting YuE Fine-tuning for RagaSense...")
    
    # Initialize fine-tuner
    fine_tuner = YuEFineTuner()
    
    # Fine-tune model
    dataset_path = "carnatic-hindustani-dataset"
    if Path(dataset_path).exists():
        logger.info(f"Fine-tuning on dataset: {dataset_path}")
        
        best_accuracy = fine_tuner.fine_tune(
            dataset_path=dataset_path,
            output_dir="yue_fine_tuned_ragasense",
            num_epochs=3,
            batch_size=4,
            learning_rate=5e-5
        )
        
        logger.info(f"Fine-tuning complete! Best accuracy: {best_accuracy:.2f}%")
        
        # Evaluate model
        results = fine_tuner.evaluate(dataset_path)
        
        # Save results
        with open("yue_fine_tuning_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info("Results saved to yue_fine_tuning_results.json")
        
    else:
        logger.error(f"Dataset not found: {dataset_path}")
        logger.info("Please ensure the dataset is available")

if __name__ == "__main__":
    main()
