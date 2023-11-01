import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datetime import datetime
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

train_dataset = load_dataset('json', data_files='/kaggle/working/training_dataset.jsonl' , split='train')
train_dataset
base_model_id = "mistralai/Mistral-7B-v0.1"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(base_model_id, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    padding_side="left",
    add_eos_token=True,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

# Assuming you have `train_dataset` defined somewhere
def generate_and_tokenize_prompt(prompt):
    return tokenizer(prompt, padding=True, truncation=True, max_length=512)

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, config)

if torch.cuda.device_count() > 1:  # If more than 1 GPU
    model.is_parallelizable = True
    model.model_parallel = True

project = "journal-finetune"
base_model_name = "Mistral"
run_name = base_model_name + "-" + project
output_dir = "./" + run_name

trainer = Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    args=TrainingArguments(
        output_dir=output_dir,
        warmup_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=500,
        learning_rate=2.5e-5,
        fp16=True,
        optim="paged_adamw_8bit",
        logging_steps=25,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=50,
        run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
