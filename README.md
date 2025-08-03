# Medical Fine-Tuning Project

A comprehensive project for fine-tuning large language models (LLMs) on medical datasets using QLoRA (Quantized Low-Rank Adaptation) technique. This project focuses on adapting pre-trained language models for medical applications and healthcare-related tasks.

## 🏥 Project Overview

This project implements fine-tuning of the Llama-2-7b-chat model on medical datasets to create a specialized medical language model. The implementation uses:

- **QLoRA (Quantized Low-Rank Adaptation)** for efficient fine-tuning
- **Medical datasets** for domain-specific training
- **4-bit quantization** for memory efficiency
- **PEFT (Parameter-Efficient Fine-Tuning)** for reduced computational requirements

## 🚀 Features

- **Efficient Fine-tuning**: Uses QLoRA to reduce memory requirements by 75%
- **Medical Domain Specialization**: Trained on medical flashcards and healthcare conversations
- **Quantized Models**: 4-bit quantization for reduced memory footprint
- **Modular Design**: Easy to adapt for different models and datasets
- **Comprehensive Logging**: TensorBoard integration for training monitoring

## 📋 Prerequisites

### System Requirements
- **GPU**: Google Colab A100 GPU (40GB VRAM) - Recommended for optimal performance
- **Alternative**: Any NVIDIA GPU with CUDA support (minimum: 8GB+ VRAM)
- **Google Colab Pro+** (for A100 access)

### Software Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU acceleration)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Med-Fine-Tuning
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install additional packages for fine-tuning**
   ```bash
   pip install accelerate peft bitsandbytes transformers trl datasets fsspec
   ```

## 📊 Datasets

The project uses the following medical datasets:

1. **Medical Meadow Medical Flashcards** (`medalpaca/medical_meadow_medical_flashcards`)
   - Medical question-answer pairs
   - Healthcare terminology and concepts
   - Clinical scenarios and explanations

2. **OpenAssistant Guanaco** (`timdettmers/openassistant-guanaco`)
   - General conversation data for baseline training
   - Helps maintain conversational abilities

## ⚙️ Configuration

### Model Configuration
```python
# Base model
model_name = "NousResearch/Llama-2-7b-chat-hf"

# Fine-tuned model name
new_model = "llama-2-7b-chat-finetune"
```

### QLoRA Parameters
```python
# LoRA attention dimension
lora_r = 64

# Alpha parameter for LoRA scaling
lora_alpha = 16

# Dropout probability for LoRA layers
lora_dropout = 0.1
```

### Training Parameters
```python
# Training epochs
num_train_epochs = 1

# Batch sizes
per_device_train_batch_size = 1
per_device_eval_batch_size = 4

# Learning rate
learning_rate = 2e-4

# Gradient accumulation
gradient_accumulation_steps = 8
```

## 🚀 Usage

### 1. Data Preparation
```python
# Load medical dataset
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards")
dataset = dataset["train"].shuffle(seed=42).select(range(1000))
```

### 2. Model Setup
```python
# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={"": 0}
)
```

### 3. LoRA Configuration
```python
# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### 4. Training
```python
# Set training arguments
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False
)

# Start training
trainer.train()
```

### 5. Inference
```python
# Load fine-tuned model
model = PeftModel.from_pretrained(model, "path/to/adapter")

# Generate responses
prompt = "What are the symptoms of diabetes?"
result = pipe(f"<s>[INST] {prompt} [/INST]")
```

## 📁 Project Structure

```
Med-Fine-Tuning/
├── Med_Fine_Tuning.ipynb          # Main fine-tuning notebook
├── README.md                      # This file
```

## 🔧 Troubleshooting

### Common Issues

1. **PackageNotFoundError: bitsandbytes**
   ```bash
   pip install bitsandbytes
   ```

2. **CUDA out of memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Use smaller model or enable gradient checkpointing

3. **Invalid Notebook Error**
   ```bash
   python fix_notebook.py "path/to/notebook.ipynb"
   ```

### Memory Optimization

- **Enable gradient checkpointing**: `gradient_checkpointing = True`
- **Use 4-bit quantization**: `load_in_4bit = True`
- **Reduce batch size**: `per_device_train_batch_size = 1`
- **Increase gradient accumulation**: `gradient_accumulation_steps = 8`

## 📈 Monitoring

### TensorBoard Integration
```bash
tensorboard --logdir ./results
```

### Training Metrics
- Loss curves
- Learning rate scheduling
- Gradient norms
- Memory usage

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the transformers library
- **Microsoft** for the LoRA technique
- **Medical Alpaca** for the medical dataset
- **Nous Research** for the base model


**Note**: This project is for educational and research purposes. Always verify medical information with qualified healthcare professionals. 
