# 🤖 ShannsGPT — Fine-Tuned Mistral 7B for YouTube Comment Responses

A QLoRA fine-tuned version of **mistralai/Mistral-7B-Instruct-v0.2** trained to generate intelligent, context-aware replies to YouTube comments — in a personalized creator style ending with *"–ShannsGPT"*.

---

## 🔗 Links

| Resource | URL |
|----------|-----|
| 🧠 Model on Hugging Face | [shannu1247/shannsGPT](https://huggingface.co/shannu1247/shannsGPT) |
| 📦 Dataset | Custom Parquet dataset (train/test split) |

---

## 📊 Training Results

The model was trained for **10 epochs** with consistent loss reduction across both training and validation sets:

| Epoch | Training Loss | Validation Loss |
|-------|--------------|-----------------|
| 1     | 4.221443     | 3.774627        |
| 2     | 3.495335     | 3.127225        |
| 3     | 2.951283     | 2.662432        |
| 4     | 2.496840     | 2.275134        |
| 5     | 2.174563     | 1.937306        |
| 6     | 1.711387     | 1.713325        |
| 7     | 1.503116     | 1.579444        |
| 8     | 1.473471     | 1.512485        |
| 9     | 1.424919     | 1.487753        |
| 10    | 1.328061     | 1.478584        |


---

## ⚙️ Training Configuration

### LoRA Configuration

| Parameter | Value |
|-----------|-------|
| Rank (r) | 8 |
| Alpha | 32 |
| Target Modules | `["q_proj"]` |
| Dropout | 0.05 |
| Bias | None |
| Task Type | CAUSAL_LM |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-4 |
| Batch Size | 4 (per device) |
| Epochs | 10 |
| Weight Decay | 0.01 |
| Gradient Accumulation Steps | 4 |
| Warmup Steps | 2 |
| Optimizer | paged_adamw_8bit |
| Precision | FP16 |
| Max Sequence Length | 512 tokens |

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers peft datasets bitsandbytes accelerate optimum huggingface_hub
```

### 2. Load the Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

base_model = "mistralai/Mistral-7B-Instruct-v0.2"
lora_model = "shannu1247/shannsGPT"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=False,
)
model = PeftModel.from_pretrained(model, lora_model)
```

### 3. Generate a Response

```python
instructions_string = """ShannsGPT, functioning as a virtual data science consultant on YouTube, communicates in clear, accessible language, escalating to technical depth upon request. \
It reacts to feedback aptly and ends responses with its signature '-ShannsGPT'. \
ShannsGPT will tailor the length of its responses to match the viewer's comment, providing concise acknowledgments to brief expressions of gratitude or feedback, \
thus keeping the interaction natural and engaging.

Please respond to the following comment.
"""

prompt_template = lambda comment: f'''[INST] {instructions_string} \n{comment} \n[/INST]'''

comment = "Great content, thank you!"
prompt = prompt_template(comment)

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"), max_new_tokens=140)
print(tokenizer.batch_decode(outputs)[0])
```

---

## 🧠 Model Behavior

- **Adaptive Length** — Tailors response length to match the viewer's comment
- **Technical Depth** — Escalates to technical detail upon request
- **Signature** — Always ends with `–ShannsGPT`

---

## 📁 Dataset

The fine-tuning dataset consists of custom YouTube comment-reply pairs stored as Parquet files.

- **Split**: Train / Test
- **Domain**: YouTube comments from data science and technical content
- **Field used**: `example` column for tokenization

---

## 🎯 Use Cases

- YouTube content creator comment responses
- Data science consultation chatbot
- Educational content engagement
- Automated community management
- Customer support for technical content

---

## ⚠️ Model Limitations

- Optimized specifically for the **YouTube comment format**
- Requires the **specific prompt template** shown above for best results
- Performance may vary with **out-of-domain** comments
- Requires a CUDA-compatible GPU with at least **15GB VRAM** (Colab T4 works)

---

## 📄 License

This project is for educational and research purposes. Please refer to the base model's license: [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2).

---

*Made with ❤️ by [Shanmuk](https://huggingface.co/shannu1247)*
