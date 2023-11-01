import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datetime import datetime
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model


def formatting_func(example):
    text = f"###Human:\nYou are an AI assistant who is good at chitchat. You have engaged on a conversation with an user. You have a Persona and a Memory. Your Memory comprises of two previous Conversations you had with the user. Generate assistant response to the user in the Current Conversation. Take into consideration assistant Persona, user Persona and Memory.\n\n{example['input']}\n\n###Assistant:\n{example['output']}"
    return text

def generate_and_tokenize_prompt2(prompt):
    result = tokenizer(
        formatting_func(prompt),
        truncation=True,

        padding=True,
    )
    result["labels"] = result["input_ids"].copy()
    return result


train_dataset = load_dataset('json', data_files='/kaggle/working/training_dataset.jsonl' , split='train')



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
