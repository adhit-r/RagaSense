# YuE Architecture Research - Comprehensive Analysis

## 🎯 **Research Objective**

Understand YuE's architecture, API requirements, and integration points for our Advanced Raga Detection System to properly leverage its musical foundation model capabilities.

## 📋 **YuE Model Overview**

### **What is YuE?**
- **Full Name**: YuE (乐) - Chinese for "music" and "happiness"
- **Purpose**: Open-source foundation model for long-form music generation
- **Capability**: Transform lyrics into complete songs (lyrics2song)
- **Output**: Full songs with both vocal and instrumental tracks
- **License**: Apache 2.0 (commercial use allowed)

### **Model Variants**
- **YuE-s1-7B-anneal-en-cot**: 7B parameters, English, Chain-of-Thought
- **YuE-s1-7B-anneal-en-icl**: 7B parameters, English, In-Context Learning
- **YuE-s2-1B-general**: 1B parameters, Stage 2 upsampler
- **Multi-language variants**: Japanese/Korean, Chinese versions

## 🏗️ **Architecture Analysis**

### **Two-Stage Architecture**

#### **Stage 1: Music Generation (7B Model)**
- **Input**: Text prompts (genre, lyrics) + optional audio reference
- **Output**: Low-resolution audio tokens (xcodec format)
- **Purpose**: Generate musical structure and basic audio tokens
- **Context Length**: 16,384 tokens
- **Generation**: Up to 3,000 new tokens per segment

#### **Stage 2: Audio Upsampling (1B Model)**
- **Input**: Stage 1 audio tokens
- **Output**: High-resolution audio tokens
- **Purpose**: Upsample and refine audio quality
- **Batch Processing**: Configurable batch size for efficiency

### **Audio Processing Pipeline**

#### **1. Audio Encoding (xcodec)**
```python
# Audio is encoded using xcodec codec
def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes
```

**Key Details**:
- **Sample Rate**: 16kHz for encoding, 44.1kHz for final output
- **Codec**: xcodec (12 codebooks, 1024 vocab each)
- **Token Rate**: 50 tokens per second (tps)
- **Bitrate**: 0.5 target bandwidth

#### **2. Tokenization System**
```python
# Multi-modal tokenizer with special tokens
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
```

**Token Types**:
- **Text Tokens**: 0-31,999 (LLaMA tokenizer)
- **Special Tokens**: 32,000-32,021
  - `<SOA>`: Start of Audio
  - `<EOA>`: End of Audio
  - `<xcodec>`: Audio codec marker
  - `<stage_1>`: Stage 1 marker
  - `<stage_2>`: Stage 2 marker
- **Audio Tokens**: 32,022-83,733
  - **xcodec**: 45,334-57,621 (12 codebooks × 1024 vocab)

#### **3. Dual-Track Processing**
```python
# Interleaved vocal and instrumental tracks
ids_segment_interleaved = rearrange([np.array(vocals_ids), np.array(instrumental_ids)], 'b n -> (n b)')
```

**Format**: `[vocal_token, instrumental_token, vocal_token, instrumental_token, ...]`

## 🔧 **API Requirements**

### **Model Loading**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Stage 1 Model
model = AutoModelForCausalLM.from_pretrained(
    "m-a-p/YuE-s1-7B-anneal-en-icl",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)

# Stage 2 Model  
model_stage2 = AutoModelForCausalLM.from_pretrained(
    "m-a-p/YuE-s2-1B-general",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2"
)
```

### **Dependencies**
```bash
# Core requirements
torch>=2.0.0
transformers>=4.50.0
torchaudio
einops
numpy
omegaconf
sentencepiece
tqdm
scipy
accelerate>=0.26.0

# Audio processing
descript-audiotools>=0.7.2
descript-audio-codec

# Memory optimization
flash-attn  # Critical for long sequences
```

### **Hardware Requirements**
- **GPU Memory**: 24GB+ recommended for full generation
- **CUDA**: 11.8+ required
- **Flash Attention**: Mandatory for long audio sequences

## 🎵 **Audio Processing Workflow**

### **1. Input Processing**
```python
def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    # Convert to mono
    audio = torch.mean(audio, dim=0, keepdim=True)
    # Resample if needed
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio
```

### **2. Audio Tokenization**
```python
# Encode audio to xcodec tokens
raw_codes = encode_audio(codec_model, audio_prompt, device, target_bw=0.5)
code_ids = codectool.npy2ids(raw_codes[0])

# Add special tokens
audio_prompt_codec_ids = [mmtokenizer.soa] + codectool.sep_ids + audio_prompt_codec + [mmtokenizer.eoa]
```

### **3. Generation Process**
```python
# Generate with specific parameters
output_seq = model.generate(
    input_ids=input_ids,
    max_new_tokens=3000,
    do_sample=True,
    top_p=0.93,
    temperature=1.0,
    repetition_penalty=1.1,
    eos_token_id=mmtokenizer.eoa,
    guidance_scale=1.5
)
```

## 🔍 **Integration Points for Raga Detection**

### **Current Issues in Our System**
1. **Wrong Input Type**: We pass feature vectors instead of raw audio
2. **Missing Audio Processing**: No xcodec encoding
3. **No Tokenization**: Not using YuE's multi-modal tokenizer
4. **Fake Integration**: `_extract_yue_embeddings()` just projects features

### **Required Changes**

#### **1. Audio Input Processing**
```python
def _extract_yue_embeddings(self, audio_path: str) -> torch.Tensor:
    """Extract actual YuE embeddings from audio file"""
    # Load raw audio
    audio_waveform, sr = torchaudio.load(audio_path)
    
    # Convert to mono and resample to 16kHz
    audio_waveform = torch.mean(audio_waveform, dim=0, keepdim=True)
    if sr != 16000:
        resampler = Resample(orig_freq=sr, new_freq=16000)
        audio_waveform = resampler(audio_waveform)
    
    # Encode to xcodec tokens
    raw_codes = self.codec_model.encode(audio_waveform.to(self.device), target_bw=0.5)
    raw_codes = raw_codes.transpose(0, 1).cpu().numpy().astype(np.int16)
    
    # Convert to token IDs
    code_ids = self.codectool.npy2ids(raw_codes[0])
    
    # Add special tokens
    audio_tokens = [self.mmtokenizer.soa] + self.codectool.sep_ids + code_ids + [self.mmtokenizer.eoa]
    
    # Get YuE embeddings
    with torch.no_grad():
        token_ids = torch.tensor(audio_tokens).unsqueeze(0).to(self.device)
        yue_outputs = self.yue_model(token_ids)
        yue_embeddings = yue_outputs.last_hidden_state.mean(dim=1)
    
    return yue_embeddings
```

#### **2. Model Architecture Updates**
```python
class YuEIndianExtension(nn.Module):
    def __init__(self, yue_model_path: str = "m-a-p/YuE-s1-7B-anneal-en-icl"):
        super().__init__()
        
        # Load YuE model and components
        self.yue_model = AutoModelForCausalLM.from_pretrained(
            yue_model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2"
        )
        
        # Load tokenizer and codec
        self.mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")
        self.codectool = CodecManipulator("xcodec", 0, 1)
        
        # Load xcodec model for audio encoding
        self.codec_model = self._load_codec_model()
        
        # Freeze YuE parameters (optional)
        for param in self.yue_model.parameters():
            param.requires_grad = False
```

#### **3. Training Pipeline**
```python
class RagaDataset(Dataset):
    def __init__(self, audio_files: List[str], labels: List[str], 
                 mmtokenizer, codectool, codec_model):
        self.audio_files = audio_files
        self.labels = labels
        self.mmtokenizer = mmtokenizer
        self.codectool = codectool
        self.codec_model = codec_model
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load and process audio
        audio_waveform = load_audio_mono(audio_path, sampling_rate=16000)
        
        # Encode to xcodec tokens
        raw_codes = encode_audio(self.codec_model, audio_waveform, device)
        code_ids = self.codectool.npy2ids(raw_codes[0])
        
        # Create audio tokens
        audio_tokens = [self.mmtokenizer.soa] + self.codectool.sep_ids + code_ids + [self.mmtokenizer.eoa]
        
        return torch.tensor(audio_tokens), torch.tensor(label)
```

## 📊 **Performance Characteristics**

### **Generation Speed**
- **H800 GPU**: 30s audio in 150 seconds
- **RTX 4090**: 30s audio in 360 seconds
- **Memory Usage**: 24GB+ for full generation

### **Audio Quality**
- **Stage 1**: Low-resolution structure generation
- **Stage 2**: High-resolution audio upsampling
- **Final Output**: 44.1kHz stereo audio

### **Context Limitations**
- **Max Context**: 16,384 tokens
- **Generation Limit**: 3,000 new tokens per segment
- **Window Slicing**: Used for long sequences

## 🚀 **Implementation Strategy**

### **Phase 1: Basic Integration**
1. **Install Dependencies**: Set up YuE environment
2. **Load Models**: YuE-s1-7B + xcodec + tokenizer
3. **Audio Processing**: Implement xcodec encoding
4. **Basic Inference**: Test with sample audio

### **Phase 2: Raga-Specific Adaptation**
1. **Cultural Extensions**: Integrate with our cultural knowledge
2. **Training Pipeline**: Create dataset and training loop
3. **Fine-tuning**: Adapt YuE for Indian classical music
4. **Evaluation**: Test performance on raga dataset

### **Phase 3: Production Optimization**
1. **Memory Optimization**: Implement gradient checkpointing
2. **Batch Processing**: Optimize for multiple audio files
3. **Model Quantization**: Reduce memory requirements
4. **API Integration**: Create production-ready interface

## ⚠️ **Critical Considerations**

### **Memory Requirements**
- **7B Model**: Requires significant GPU memory
- **Flash Attention**: Mandatory for long sequences
- **Batch Size**: Limited by available memory

### **Audio Format Requirements**
- **Input**: Raw audio waveforms (not features)
- **Sample Rate**: 16kHz for encoding
- **Format**: Mono audio preferred
- **Duration**: Up to several minutes

### **Tokenization Complexity**
- **Multi-modal**: Text + audio tokens
- **Special Tokens**: Complex token structure
- **Codec Integration**: xcodec encoding required

## 🎯 **Next Steps**

### **Immediate Actions**
1. **Set up YuE environment** with all dependencies
2. **Test basic audio processing** with xcodec
3. **Implement proper audio-to-YuE interface**
4. **Create simple test script** for validation

### **Development Priorities**
1. **Fix audio processing pipeline** (HIGH)
2. **Implement proper tokenization** (HIGH)
3. **Create training dataset** (MEDIUM)
4. **Test with raga audio** (MEDIUM)

## 📚 **Resources**

### **Official Documentation**
- **GitHub**: https://github.com/multimodal-art-projection/YuE
- **Paper**: https://arxiv.org/abs/2503.08638
- **Demo**: https://map-yue.github.io/
- **Hugging Face**: https://huggingface.co/m-a-p/YuE-s1-7B-anneal-en-icl

### **Key Files to Study**
- `inference/infer.py`: Main inference pipeline
- `inference/mmtokenizer.py`: Multi-modal tokenizer
- `inference/codecmanipulator.py`: Audio codec handling
- `finetune/`: Training and fine-tuning code

---

**Research completed by**: Adhithya Rajasekaran (@adhit-r)  
**Date**: September 6, 2025  
**Status**: ✅ COMPLETE
