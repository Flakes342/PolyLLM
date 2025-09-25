# %%
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))

# %%
from huggingface_hub import notebook_login
notebook_login()

# %%
import torch
import json
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import numpy as np
from datasets import load_dataset, Dataset
from accelerate import Accelerator
from transformers import TrainerCallback
import torch.nn as nn

# %%
# Initialize accelerator
accelerator = Accelerator()
DEVICE = accelerator.device

# %%
# Configuration
teacher_model_name = "meta-llama/Llama-2-7b-hf"
student_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
MAX_LENGTH = 512
BATCH_SIZE = 2
TEMPERATURE = 2.0
ALPHA = 0.5

# %%
# Load teacher model
teacher_model = AutoModelForCausalLM.from_pretrained(
    teacher_model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

# %%
# Load student model
student_model = AutoModelForCausalLM.from_pretrained(
    student_model_name,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).to(device)

# %%
# Load tokenizers
teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
student_tokenizer = AutoTokenizer.from_pretrained(student_model_name)

# Ensure both have pad tokens
if teacher_tokenizer.pad_token is None:
    teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
if student_tokenizer.pad_token is None:
    student_tokenizer.pad_token = student_tokenizer.eos_token

# %%
# Load Alpaca dataset
dataset = load_dataset("tatsu-lab/alpaca")
dataset = dataset["train"].train_test_split(test_size=0.1, seed=42)

print(f"Train samples: {len(dataset['train'])}")
print(f"Test samples: {len(dataset['test'])}")

# %%
def preprocess(examples, tokenizer, max_length=512):
    """Preprocess examples for training"""
    texts = []
    for instr, inp, out in zip(examples["instruction"], examples["input"], examples["output"]):
        if inp:
            text = f"Instruction: {instr}\nInput: {inp}\nOutput: {out}"
        else:
            text = f"Instruction: {instr}\nOutput: {out}"
        texts.append(text)

    model_inputs = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    return model_inputs

# %%
# Create datasets
train_dataset = dataset["train"].map(
    lambda x: preprocess(x, student_tokenizer, MAX_LENGTH),
    batched=True,
    remove_columns=dataset["train"].column_names,
)

eval_dataset = dataset["test"].map(
    lambda x: preprocess(x, student_tokenizer, MAX_LENGTH),
    batched=True,
    remove_columns=dataset["test"].column_names,
)

# %%
def distillation_loss(student_logits, teacher_logits, labels, temperature=2.0, alpha=0.5):
    """Compute distillation loss combining CE and KL divergence"""
    # CrossEntropy on ground-truth labels
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    student_loss = loss_fct(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1)
    )

    # KL Divergence between teacher & student soft predictions
    T = temperature
    kd_loss = nn.KLDivLoss(reduction="batchmean")(
        torch.nn.functional.log_softmax(student_logits / T, dim=-1),
        torch.nn.functional.softmax(teacher_logits / T, dim=-1)
    ) * (T * T)

    return alpha * student_loss + (1 - alpha) * kd_loss

# %%
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, device=None, **kwargs):
        if 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.device = device or torch.device("cuda:0")
        
        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model = self.teacher_model.to(self.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Explicitly move everything to the same device
        model = model.to(self.device)
        
        # Move all inputs to device
        device_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                device_inputs[key] = value.to(self.device)
            else:
                device_inputs[key] = value
        
        labels = device_inputs.get("labels")
        
        # Forward pass (student)
        outputs_student = model(**device_inputs)
        student_logits = outputs_student.logits

        # Forward pass (teacher) 
        with torch.no_grad():
            self.teacher_model = self.teacher_model.to(self.device)
            outputs_teacher = self.teacher_model(**device_inputs)
            teacher_logits = outputs_teacher.logits

        # Align dimensions
        min_length = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_length, :]
        teacher_logits = teacher_logits[:, :min_length, :]
        labels = labels[:, :min_length]

        loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)
        
        return (loss, outputs_student) if return_outputs else loss

# %%
training_args = TrainingArguments(
    output_dir="./distilled_model",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=4,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps", 
    save_steps=1000,
    logging_steps=100,
    num_train_epochs=1,
    learning_rate=5e-5,
    warmup_steps=50,
    weight_decay=0.01,
    
    # FP16 fixes
    fp16=False,  # Disable FP16 to avoid gradient issues
    bf16=True,   # Use BF16 instead (more stable)
    # OR use fp16_full_eval=True if you want to keep fp16
    
    dataloader_num_workers=0,
    remove_unused_columns=True,
    report_to="none",
    seed=42,
    
    # Gradient clipping fixes
    max_grad_norm=None,  # Disable gradient clipping temporarily
    # OR use a smaller value like 0.1
    
    dataloader_pin_memory=False,
    local_rank=-1,
)

# %%
class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, device=None, **kwargs):
        if 'tokenizer' in kwargs:
            kwargs['processing_class'] = kwargs.pop('tokenizer')
        
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.device = device or torch.device("cuda:0")
        
        if self.teacher_model is not None:
            self.teacher_model.eval()
            self.teacher_model = self.teacher_model.to(self.device)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        model = model.to(self.device)
        
        device_inputs = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                device_inputs[key] = value.to(self.device)
            else:
                device_inputs[key] = value
        
        labels = device_inputs.get("labels")
        
        # Forward pass (student)
        outputs_student = model(**device_inputs)
        student_logits = outputs_student.logits

        # Forward pass (teacher) 
        with torch.no_grad():
            self.teacher_model = self.teacher_model.to(self.device)
            outputs_teacher = self.teacher_model(**device_inputs)
            teacher_logits = outputs_teacher.logits

        # Align dimensions
        min_length = min(student_logits.size(1), teacher_logits.size(1))
        student_logits = student_logits[:, :min_length, :]
        teacher_logits = teacher_logits[:, :min_length, :]
        labels = labels[:, :min_length]

        loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)
        
        # Ensure loss is float32 for stable gradients
        if loss.dtype == torch.float16:
            loss = loss.float()
        
        return (loss, outputs_student) if return_outputs else loss

trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=student_tokenizer,
    teacher_model=teacher_model,
)

print("Starting training with stable precision...")
print(f"Total training steps: {len(trainer.get_train_dataloader()) * training_args.num_train_epochs}")

trainer.train()

# %%
# Create test dataloader for evaluation
def create_test_dataloader():
    test_dataset_torch = eval_dataset.with_format("torch")
    return DataLoader(test_dataset_torch, batch_size=BATCH_SIZE, shuffle=False)

test_loader = create_test_dataloader()

def calculate_perplexity(model, dataloader):
    """Calculate perplexity of model on dataset"""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Count non-padded tokens
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

print("Calculating perplexities...")
student_ppl = calculate_perplexity(student_model, test_loader)

# For teacher perplexity, we need to be careful with tokenization
def calculate_teacher_perplexity():
    teacher_model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in test_loader:
            # Decode and re-tokenize for teacher
            texts = student_tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)
            teacher_inputs = teacher_tokenizer(
                texts,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(DEVICE)

            outputs = teacher_model(
                input_ids=teacher_inputs["input_ids"],
                attention_mask=teacher_inputs["attention_mask"],
                labels=teacher_inputs["input_ids"]
            )
            loss = outputs.loss

            valid_tokens = (teacher_inputs["input_ids"] != teacher_tokenizer.pad_token_id).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

teacher_ppl = calculate_teacher_perplexity()

def token_kl_divergence(student_model, teacher_model, dataloader):
    """Calculate KL divergence between student and teacher predictions"""
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    total_kl = 0
    count = 0

    student_model.eval()
    teacher_model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # Student predictions
            student_outputs = student_model(input_ids=input_ids, attention_mask=attention_mask)
            student_logits = student_outputs.logits

            # Teacher predictions (re-tokenize)
            texts = student_tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            teacher_inputs = teacher_tokenizer(
                texts,
                max_length=MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(DEVICE)

            teacher_outputs = teacher_model(**teacher_inputs)
            teacher_logits = teacher_outputs.logits

            # Align dimensions
            min_length = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_length, :]
            teacher_logits = teacher_logits[:, :min_length, :]

            kl = kl_loss_fn(
                torch.nn.functional.log_softmax(student_logits, dim=-1),
                torch.nn.functional.softmax(teacher_logits, dim=-1)
            )
            total_kl += kl.item()
            count += 1

    return total_kl / count if count > 0 else float('inf')

print("Calculating KL divergence...")
token_kl = token_kl_divergence(student_model, teacher_model, test_loader)

def measure_generation_speed(model, tokenizer, prompt="Explain quantum computing in simple terms.", max_new_tokens=50):
    """Measure generation speed"""
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    end_time = time.time()

    return max_new_tokens / (end_time - start_time)

print("Measuring generation speeds...")
student_speed = measure_generation_speed(student_model, student_tokenizer)
teacher_speed = measure_generation_speed(teacher_model, teacher_tokenizer)

def calculate_model_size(model):
    """Calculate model size in GB"""
    temp_file = "temp_model.pt"
    torch.save(model.state_dict(), temp_file)
    size = os.path.getsize(temp_file) / 1e9  # Convert to GB
    os.remove(temp_file)
    return size

print("Calculating model sizes...")
student_size = calculate_model_size(student_model)

# For teacher size, we'll estimate based on parameters since it's quantized
teacher_params = sum(p.numel() for p in teacher_model.parameters())
teacher_size = teacher_params * 4 / 1e9  # Assuming fp32 for comparison

# Create comprehensive metrics table
print("\n" + "="*60)
print("KNOWLEDGE DISTILLATION RESULTS")
print("="*60)

metrics = {
    "Metric": [
        "Perplexity",
        "Token KL Divergence",
        "Generation Speed (tokens/sec)",
        "Model Size (GB)",
        "Compression Ratio",
        "Speed Improvement"
    ],
    "Teacher (LLaMA-2 7B)": [
        f"{teacher_ppl:.2f}",
        "0.00 (baseline)",
        f"{teacher_speed:.2f}",
        f"{teacher_size:.2f}",
        "1.0x",
        "1.0x"
    ],
    "Student (OPT-350M)": [
        f"{student_ppl:.2f}",
        f"{token_kl:.4f}",
        f"{student_speed:.2f}",
        f"{student_size:.2f}",
        f"{teacher_size/student_size:.1f}x",
        f"{student_speed/teacher_speed:.1f}x"
    ],
    "Performance Notes": [
        "Lower is better",
        "Lower = better alignment",
        "Higher is better",
        "Smaller = more efficient",
        "Higher = better compression",
        "Higher = faster inference"
    ]
}

df = pd.DataFrame(metrics)
print(df.to_string(index=False))

print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
print(f"✓ Successfully distilled {teacher_model_name} into {student_model_name}")
print(f"✓ Achieved {teacher_size/student_size:.1f}x model compression")
print(f"✓ Achieved {student_speed/teacher_speed:.1f}x speed improvement")
print(f"✓ Perplexity degradation: {((student_ppl/teacher_ppl - 1) * 100):.1f}%")
print(f"✓ Model saved to: ./distilled_model")

# Save metrics to file
with open("distillation_results.json", "w") as f:
    results = {
        "teacher_model": teacher_model_name,
        "student_model": student_model_name,
        "teacher_perplexity": teacher_ppl,
        "student_perplexity": student_ppl,
        "kl_divergence": token_kl,
        "teacher_speed": teacher_speed,
        "student_speed": student_speed,
        "teacher_size_gb": teacher_size,
        "student_size_gb": student_size,
        "compression_ratio": teacher_size/student_size,
        "speed_improvement": student_speed/teacher_speed
    }
    json.dump(results, f, indent=2)

print(f"✓ Detailed results saved to: distillation_results.json")


