# -----------------------------
# 1️⃣ Imports
# -----------------------------
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline
)
from peft import LoraConfig, get_peft_model

# -----------------------------
# 2️⃣ Load Dataset
# -----------------------------
dataset = Dataset.from_dict({"text": texts})
dataset = dataset.train_test_split(test_size=0.05)

# -----------------------------
# 3️⃣ Tokenizer Setup
# -----------------------------
model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset.map(tokenize, batched=True, remove_columns=["text"])

# -----------------------------
# 4️⃣ Model & LoRA Setup
# -----------------------------
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -----------------------------
# 5️⃣ Data Collator
# -----------------------------
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -----------------------------
# 6️⃣ Training Arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="/home/ubuntu/experiments/lora_checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
)

# -----------------------------
# 7️⃣ Trainer Setup
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
)

# -----------------------------
# 8️⃣ Train Model
# -----------------------------
trainer.train()

# -----------------------------
# 9️⃣ Save Model & Tokenizer
# -----------------------------
model.save_pretrained("/home/ubuntu/experiments/trained")
tokenizer.save_pretrained("/home/ubuntu/experiments/trained")

# -----------------------------
# 🔟 Quick Test Generation
# -----------------------------
pipe = pipeline(
    "text-generation",
    model="/home/ubuntu/experiments/trained",
    tokenizer=tokenizer,
    device_map="auto"
)

prompt = "Your prompt goes here"

response = pipe(
    prompt,
    max_new_tokens=250,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    top_k=50,
    truncation=True,
    return_full_text=False  # Returns only generated tokens
)

print("\n✅ Generated Answer:\n", response[0]["generated_text"])
