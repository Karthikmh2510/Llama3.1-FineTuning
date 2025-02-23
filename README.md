# MedTerm Fine-Tuning with Llama-3.1-8B-Instruct

## Overview
This repository contains the fine-tuning process for **Llama-3.1-8B-Instruct** on **medical terminology datasets**. The model has been trained to generate precise and context-aware responses for medical questions.

## Features
- **Fine-Tuned Llama-3.1-8B Model**: Optimized for medical terminology understanding and generation.
- **Efficient Quantization**: Uses **4-bit quantization** with `BitsAndBytesConfig` for reduced memory consumption.
- **LoRA (Low-Rank Adaptation)**: Enables efficient training with fewer resources.
- **Medical Dataset Utilization**: Trained on `dmedhi/wiki_medical_terms` for improved domain-specific knowledge.
- **Evaluation with BIOSSES**: Assesses biomedical sentence similarity for model performance.
- **Streamlined Deployment**: Supports **Hugging Face Model Hub** for easy sharing and usage.

## Tech Stack
- **Python**
- **Hugging Face Transformers** (for LLM fine-tuning)
- **TRL (Transformers Reinforcement Learning Library)**
- **PEFT (Parameter Efficient Fine-Tuning)**
- **BitsAndBytes (bnb) for Quantization**
- **Google Colab** (Training Environment)
- **Hugging Face Hub** (Model Storage and Deployment)

## Installation
### Prerequisites
- Python 3.8+
- Hugging Face API Token

### Steps
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Karthik2510/Llama3.1-FineTuning.git
   cd Llama3.1-FineTuning
   ```
2. **Install Dependencies**
   ```bash
   pip install bitsandbytes peft trl accelerate datasets transformers huggingface_hub
   ```
3. **Set Environment Variables**
   Create a `.env` file and add your API keys:
   ```env
   HF_Finegrained_Token=your_hf_finegrained_token
   HF_Write_Token=your_hf_write_token
   ```

## Training the Model
### Model Configuration
- **Base Model**: `meta-llama/Llama-3.1-8B-Instruct`
- **Quantization**: 4-bit precision with **NF4** format.
- **LoRA Configuration**:
  - Rank (`r`): 16
  - Alpha: 32
  - Dropout: 0.05
  - Target Modules: `q_proj, o_proj, k_proj, v_proj, gate_proj, up_proj, down_proj`

### Fine-Tuning Execution
```python
from datasets import load_dataset

dataset = load_dataset("dmedhi/wiki_medical_terms", split="train")

def format_data(entry):
    return {
        "text": f"Question: {entry['medical_term']}\nAnswer: {entry['wiki_description']}"
    }

formatted_dataset = dataset.map(format_data)
```

```python
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args = transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
)

trainer.train()
```

### Saving & Uploading Model
```python
model.save_pretrained("Karthik2510/Medi_terms_Llama3.1_8B_instruct_model")
```
```python
from huggingface_hub import notebook_login
notebook_login()

model.push_to_hub("Karthik2510/Medi_terms_Llama3.1_8B_instruct_model", token=HF_WRITE_TOKEN)
```

## Inferencing the Model
```python
text = "What is Paracetamol poisoning?"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Model Evaluation using BIOSSES
### Evaluation Setup
- **BIOSSES Dataset**: Biomedical Sentence Similarity dataset for evaluating text coherence.
- **Sentence Transformers**: `all-MiniLM-L6-v2` used for similarity scoring.

```python
from sentence_transformers import SentenceTransformer, util
sim_model = SentenceTransformer("all-MiniLM-L6-v2")
```

### Evaluation Process
```python
similarity_scores = []
num_samples = min(10, len(dataset_BIOSSES))
for i in range(num_samples):
    s1 = dataset_BIOSSES[i]["sentence1"]
    s2 = dataset_BIOSSES[i]["sentence2"]
    generated_text = generate_response(s1)
    similarity = compute_similarity(generated_text, s2)
    similarity_scores.append(similarity)
    print(f"Example {i+1}:\nGenerated: {generated_text}\nReference: {s2}\nScore: {similarity:.4f}\n")

average_similarity = sum(similarity_scores) / len(similarity_scores)
print(f"Model's Average Similarity Score on BIOSSES: {average_similarity:.4f}")
```

## Project Structure
```
|-- fine_tune.py              # Fine-tuning script
|-- inference.py              # Model inference script
|-- evaluation.py             # Model evaluation on BIOSSES
|-- requirements.txt          # Required dependencies
|-- .env.example             # Example environment file
|-- data/                     # Optional dataset storage
```

## Future Enhancements
- Expand training dataset with more **medical literature**.
- Implement **multi-turn dialogue capability**.
- Deploy as an **API service for real-time querying**.
- Improve performance with **adapter tuning and hyperparameter optimization**.

## Contributors
- **Karthik Manjunath Hadagali** – AI Research Engineer

## License
This project is licensed under the **MIT License**. Feel free to contribute!

---

**🚀 If you found this project useful, give it a star ⭐ on GitHub!**

