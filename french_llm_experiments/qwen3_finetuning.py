# -*- coding: utf-8 -*-
"""Qwen3_finetuning.ipynb

"""

from google.colab import drive
drive.mount('/content/drive')

import os

base_path = "/content/drive/MyDrive"

for root, dirs, files in os.walk(base_path):
    if "hipe-ocrepair-bench_v0.9_icdar2017_v1.1_train_fr.jsonl" in files:
        print(root)
        break

import json
import random

train_file = "/content/drive/MyDrive/fr/hipe-ocrepair-bench_v0.9_icdar2017_v1.1_train_fr.jsonl"

data = []

with open(train_file, encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print("Loaded:", len(data))

print(data[0]["ocr_hypothesis"])
print(data[0]["ground_truth"])

sample = random.choice(data)

ocr_text = sample["ocr_hypothesis"]["transcription_unit"]
gt_text = sample["ground_truth"]["transcription_unit"]

print("OCR\n")
print(ocr_text[:3000])

print("\n\nGROUND TRUTH\n")
print(gt_text[:3000])

from difflib import ndiff

diff = ndiff(ocr_text, gt_text)

changes = []

for d in diff:
    if d[0] != " ":
        changes.append(d)

print("".join(changes[:500]))

!pip install -q jiwer

from jiwer import cer

scores = []

for idx, sample in enumerate(data):

    ocr = sample["ocr_hypothesis"]["transcription_unit"]
    gt = sample["ground_truth"]["transcription_unit"]

    score = cer(gt, ocr)

    scores.append((score, idx))

scores = sorted(scores, reverse=True)

print(scores[:10])

worst_idx = 172

sample = data[worst_idx]

ocr_text = sample["ocr_hypothesis"]["transcription_unit"]
gt_text = sample["ground_truth"]["transcription_unit"]

print("OCR \n")
print(ocr_text[:4000])

print("\n\n GROUND TRUTH \n")
print(gt_text[:4000])

from difflib import SequenceMatcher

ocr_words = ocr_text[:3000].split()
gt_words = gt_text[:3000].split()

matcher = SequenceMatcher(None, ocr_words, gt_words)

count = 0

for tag, i1, i2, j1, j2 in matcher.get_opcodes():

    if tag != "equal":


        print("TYPE:", tag)

        print("\nOCR:")
        print(" ".join(ocr_words[i1:i2]))

        print("\nGT:")
        print(" ".join(gt_words[j1:j2]))

        count += 1

    if count >= 40:
        break



!pip install -q transformers datasets peft accelerate bitsandbytes trl

from peft import LoraConfig

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model_name = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

!pip uninstall -y torchao

!pip install -q torchao>=0.16.0



from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

from peft import prepare_model_for_kbit_training, get_peft_model

model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, lora_config)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

model = prepare_model_for_kbit_training(model)

model = get_peft_model(model, lora_config)

model.print_trainable_parameters()

def tokenize(example):

    prompt = (
        "You are an expert in OCR post-correction for historical French documents.\n\n"
        "Your task is ONLY to correct obvious OCR recognition errors.\n\n"
        "Rules:\n"
        "- Preserve the original historical French spelling.\n"
        "- Do NOT modernize the language.\n"
        "- Do NOT rewrite or paraphrase sentences.\n"
        "- Do NOT change punctuation unless it is clearly an OCR mistake.\n"
        "- Do NOT change capitalization unless it is clearly incorrect because of OCR.\n"
        "- If a word is uncertain, leave it unchanged.\n"
        "- When in doubt, prefer the original text.\n"
        "- Output only the corrected text.\n\n"
        f"OCR text:\n{example['prompt']}\n\n"
        "Corrected text:\n"
    )

    target = example["answer"]

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=128,
        add_special_tokens=True,
    )

    target_tokens = tokenizer(
        target,
        truncation=True,
        padding="max_length",
        max_length=128,
        add_special_tokens=False,
    )

    input_ids = prompt_tokens["input_ids"] + target_tokens["input_ids"]

    attention_mask = (
        prompt_tokens["attention_mask"]
        + target_tokens["attention_mask"]
    )

    labels = (
        [-100] * len(prompt_tokens["input_ids"])
        + target_tokens["input_ids"]
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

formatted_data = []

for sample in data:
    formatted_data.append({
        "prompt": sample["ocr_hypothesis"]["transcription_unit"],
        "answer": sample["ground_truth"]["transcription_unit"],
    })

from datasets import Dataset

dataset = Dataset.from_list(formatted_data)

dataset = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = dataset["train"].map(tokenize)
eval_dataset = dataset["test"].map(tokenize)

import transformers
print(transformers.__version__)

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./ocr-lora",

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    learning_rate=1e-4,

    num_train_epochs=3,

    logging_steps=10,

    save_strategy="epoch",
    eval_strategy="epoch",

    load_best_model_at_end=True,

    fp16=True,

    optim="paged_adamw_8bit",

    warmup_ratio=0.1,

    lr_scheduler_type="cosine",
)

from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

save_path = "/content/drive/MyDrive/ocr-lora"

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("Model saved!")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

model_name = "Qwen/Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_name)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    base_model,
    "/content/drive/MyDrive/ocr-lora"
)

def split_text(text, chunk_size=500):

    chunks = []

    for i in range(0, len(text), chunk_size):

        chunks.append(text[i:i+chunk_size])

    return chunks

ocr_text = sample["ocr_hypothesis"]["transcription_unit"]
gt_text = sample["ground_truth"]["transcription_unit"]

chunks = split_text(ocr_text, chunk_size=500)

print(len(chunks))

def correct_ocr(ocr_text):

    prompt = ("""
        "You are an expert in OCR post-correction for historical French documents.\n\n"
        "Your task is ONLY to correct obvious OCR recognition errors.\n\n"
        "Rules:\n"
        "- Preserve the original historical French spelling.\n"
        "- Do NOT modernize the language.\n"
        "- Do NOT rewrite or paraphrase sentences.\n"
        "- Do NOT change punctuation unless it is clearly an OCR mistake.\n"
        "- Do NOT change capitalization unless it is clearly incorrect because of OCR.\n"
        "- If a word is uncertain, leave it unchanged.\n"
        "- When in doubt, prefer the original text.\n"
        "- Output only the corrected text.\n\n"

OCR text:

{ocr_text}

Corrected text:
"""
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False
    )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    prediction = decoded.split("Corrected text:")[-1].strip()

    return prediction

from jiwer import cer, wer
import numpy as np

from tqdm import tqdm
import torch

ocr_cers = []
ocr_wers = []

model_cers = []
model_wers = []

for sample in tqdm(data[:20]):

    ocr_text = sample["ocr_hypothesis"]["transcription_unit"][:500]

    gt_text = sample["ground_truth"]["transcription_unit"][:500]

    with torch.no_grad():

        prediction = correct_ocr(ocr_text)

    # OCR baseline
    ocr_cers.append(cer(gt_text, ocr_text))
    ocr_wers.append(wer(gt_text, ocr_text))

    # Model
    model_cers.append(cer(gt_text, prediction))
    model_wers.append(wer(gt_text, prediction))

    print("\n PREDICTION \n")

    print(prediction[:300])





