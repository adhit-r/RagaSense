# YuE Implementation Plan - Fixing the Integration

## 🎯 **Objective**

Transform the Advanced Raga Detection System from a sophisticated feature extractor into a true YuE-powered raga detection system that leverages the foundation model's musical understanding.

## 📋 **Current State Analysis**

### **What We Have** ✅
- Sophisticated cultural knowledge system (22-shruti, taals, gamakas)
- Advanced audio processing pipeline
- Well-designed modular architecture
- Proper error handling and fallback mechanisms

### **What's Missing** ❌
- Actual YuE model integration
- Audio-to-YuE interface
- Training pipeline
- Real musical understanding from YuE

## 🚀 **Implementation Roadmap**

### **Phase 1: Fix YuE Integration** (Priority: HIGH)

#### **Step 1.1: Research YuE Architecture**
```bash
# Research YuE's actual architecture and API
cd ml/YuE
python -c "from transformers import AutoModel; model = AutoModel.from_pretrained('m-a-p/YuE-s1-7B-anneal-en-icl'); print(model.config)"
```

**Tasks**:
- [ ] Understand YuE's input format (audio tokens, sample rate, etc.)
- [ ] Research YuE's audio encoder architecture
- [ ] Identify proper tokenization methods
- [ ] Test YuE model loading and basic inference

#### **Step 1.2: Implement Proper Audio Processing**
```python
# New implementation in advanced_raga_detector_v1.0.py
class YuEIndianExtension(nn.Module):
    def __init__(self, yue_model_path: str = "m-a-p/YuE-s1-7B-anneal-en-icl"):
        super().__init__()
        
        # Load YuE model with proper configuration
        self.yue_model = AutoModel.from_pretrained(yue_model_path)
        self.yue_tokenizer = AutoTokenizer.from_pretrained(yue_model_path)
        self.yue_sample_rate = 22050  # YuE's expected sample rate
        
        # Freeze YuE parameters (optional)
        for param in self.yue_model.parameters():
            param.requires_grad = False
    
    def _extract_yue_embeddings(self, audio_path: str) -> torch.Tensor:
        """Extract actual YuE embeddings from audio file"""
        # Load raw audio
        audio_waveform, sr = torchaudio.load(audio_path)
        
        # Resample to YuE's expected sample rate
        if sr != self.yue_sample_rate:
            audio_waveform = torchaudio.functional.resample(
                audio_waveform, sr, self.yue_sample_rate
            )
        
        # Convert to YuE's audio tokens
        audio_tokens = self.yue_tokenizer.encode(audio_waveform)
        
        # Get YuE embeddings
        with torch.no_grad():
            yue_outputs = self.yue_model(audio_tokens)
            yue_embeddings = yue_outputs.last_hidden_state.mean(dim=1)
        
        return yue_embeddings
```

#### **Step 1.3: Update Model Architecture**
```python
def forward(self, audio_path: str, cultural_context: Dict = None):
    """Forward pass with actual YuE integration"""
    # Extract YuE embeddings from raw audio
    yue_embeddings = self._extract_yue_embeddings(audio_path)
    
    # Extract cultural features from audio
    cultural_features = self._extract_cultural_features(audio_path)
    
    # Fuse YuE and cultural knowledge
    fused_features = self.cultural_fusion(yue_embeddings, cultural_features)
    
    # Final classification
    raga_logits = self.final_classifier(fused_features)
    
    return {
        'raga_logits': raga_logits,
        'yue_embeddings': yue_embeddings,
        'cultural_features': cultural_features
    }
```

### **Phase 2: Create Training Pipeline** (Priority: HIGH)

#### **Step 2.1: Create Training Dataset**
```python
class RagaDataset(Dataset):
    def __init__(self, audio_files: List[str], labels: List[str], 
                 yue_tokenizer, audio_processor):
        self.audio_files = audio_files
        self.labels = labels
        self.yue_tokenizer = yue_tokenizer
        self.audio_processor = audio_processor
        
        # Create label mapping
        unique_labels = list(set(labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load and process audio for YuE
        audio_waveform, sr = torchaudio.load(audio_path)
        
        # Resample if needed
        if sr != 22050:
            audio_waveform = torchaudio.functional.resample(audio_waveform, sr, 22050)
        
        # Convert to YuE tokens
        audio_tokens = self.yue_tokenizer.encode(audio_waveform)
        
        # Get label index
        label_idx = self.label_to_idx[label]
        
        return audio_tokens, torch.tensor(label_idx, dtype=torch.long)
```

#### **Step 2.2: Implement Training Loop**
```python
def train_raga_detector(model, dataset, epochs=100, batch_size=16):
    """Train the complete raga detection system"""
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup training
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (audio_tokens, labels) in enumerate(dataloader):
            # Move to device
            audio_tokens = audio_tokens.to(model.device)
            labels = labels.to(model.device)
            
            # Forward pass
            outputs = model(audio_tokens)
            loss = criterion(outputs['raga_logits'], labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        scheduler.step()
        print(f'Epoch {epoch} completed. Average Loss: {total_loss/len(dataloader):.4f}')
```

### **Phase 3: Testing and Validation** (Priority: MEDIUM)

#### **Step 3.1: Create Test Script**
```python
def test_yue_integration():
    """Test the YuE integration with real audio"""
    
    # Initialize system
    system = AdvancedRagaDetectionSystem()
    
    # Test with sample audio
    test_audio = "data/carnatic-hindustani/Carnatic/Shankarabharanam/sample.wav"
    
    if Path(test_audio).exists():
        print("Testing YuE integration...")
        
        # Analyze audio
        results = system.analyze_audio(test_audio, detailed=True)
        
        if 'error' not in results:
            print("✅ YuE integration successful!")
            print(f"Predicted raga: {results['primary_prediction']['raga']}")
            print(f"Confidence: {results['primary_prediction']['confidence']:.3f}")
        else:
            print(f"❌ Error: {results['error']}")
    else:
        print("❌ Test audio not found")

if __name__ == "__main__":
    test_yue_integration()
```

#### **Step 3.2: Performance Evaluation**
```python
def evaluate_yue_performance(dataset_path: str):
    """Evaluate YuE-powered system performance"""
    
    system = AdvancedRagaDetectionSystem()
    
    # Load test dataset
    audio_files = list(Path(dataset_path).glob("**/*.wav"))
    
    results = []
    for audio_file in audio_files[:10]:  # Test with first 10 files
        result = system.analyze_audio(str(audio_file))
        results.append(result)
    
    # Calculate metrics
    successful_results = [r for r in results if 'error' not in r]
    accuracy = len(successful_results) / len(results)
    
    print(f"Success rate: {accuracy:.2%}")
    print(f"Successful analyses: {len(successful_results)}/{len(results)}")
```

## 🔧 **Implementation Steps**

### **Week 1: YuE Research and Setup**
- [ ] Research YuE architecture and API
- [ ] Test YuE model loading
- [ ] Understand audio tokenization
- [ ] Set up development environment

### **Week 2: Fix YuE Integration**
- [ ] Implement proper audio processing
- [ ] Update model architecture
- [ ] Fix forward pass
- [ ] Test basic functionality

### **Week 3: Training Pipeline**
- [ ] Create training dataset class
- [ ] Implement training loop
- [ ] Add validation and metrics
- [ ] Test training on small dataset

### **Week 4: Testing and Optimization**
- [ ] Create comprehensive test suite
- [ ] Evaluate performance
- [ ] Optimize hyperparameters
- [ ] Document results

## 📊 **Expected Outcomes**

### **Before Implementation**
- **Accuracy**: ~60-70% (simple CNN)
- **YuE Usage**: 0% (not actually used)
- **Cultural Understanding**: High (hardcoded)
- **Generalization**: Limited

### **After Implementation**
- **Accuracy**: ~80-90% (expected)
- **YuE Usage**: 100% (properly integrated)
- **Cultural Understanding**: High (learned + hardcoded)
- **Generalization**: Excellent

### **Success Metrics**
- [ ] YuE model loads successfully
- [ ] Audio processing works without errors
- [ ] Training converges
- [ ] Accuracy improves over baseline
- [ ] System handles edge cases gracefully

## 🚨 **Risk Mitigation**

### **Technical Risks**
- **YuE model too large**: Use model quantization or smaller variants
- **Memory issues**: Implement batch processing and gradient checkpointing
- **Training instability**: Use proper learning rate scheduling and regularization

### **Data Risks**
- **Insufficient data**: Use data augmentation and transfer learning
- **Label quality**: Implement data validation and cleaning
- **Class imbalance**: Use stratified sampling and weighted loss

## 🎯 **Next Actions**

### **Immediate (Today)**
1. Research YuE architecture and API
2. Test YuE model loading in development environment
3. Understand audio tokenization requirements

### **This Week**
1. Implement proper YuE audio processing
2. Update model architecture
3. Create basic test script

### **Next Week**
1. Implement training pipeline
2. Test with small dataset
3. Evaluate performance

---

**Implementation Plan created by**: Adhithya Rajasekaran (@adhit-r)  
**Date**: September 6, 2025  
**Status**: 🔄 READY FOR IMPLEMENTATION
