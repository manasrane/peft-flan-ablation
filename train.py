import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import (
    LoraConfig,
    PrefixTuningConfig,
    PromptTuningConfig,
    get_peft_model
)
import yaml
import argparse
from data.agnews import load_agnews, preprocess_agnews
from data.cnn_dm import load_cnn_dm, preprocess_cnn_dm
from utils import count_trainable_params

def build_model(method, config):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    if method == "lora":
        peft_config = LoraConfig(
            r=config.get('r', 8),
            lora_alpha=config.get('lora_alpha', 32),
            target_modules=["q", "v"],
            lora_dropout=config.get('lora_dropout', 0.1),
            task_type="SEQ_2_SEQ_LM"
        )
        model = get_peft_model(model, peft_config)

    elif method == "prefix":
        peft_config = PrefixTuningConfig(
            task_type="SEQ_2_SEQ_LM",
            num_virtual_tokens=config.get('num_virtual_tokens', 10),
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)

    elif method == "prompt":
        peft_config = PromptTuningConfig(
            task_type="SEQ_2_SEQ_LM",
            num_virtual_tokens=config.get('num_virtual_tokens', 20),
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)

    # full_ft = no PEFT
    return model, tokenizer

def tokenize_function(example, tokenizer):
    model_inputs = tokenizer(example["input_text"], max_length=512, truncation=True)
    labels = tokenizer(example["target_text"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    method = config['method']
    task = config['task']

    model, tokenizer = build_model(method, config)

    if task == "classification":
        train_dataset, eval_dataset = load_agnews(config)
        preprocess_fn = preprocess_agnews
    elif task == "summarization":
        train_dataset, eval_dataset = load_cnn_dm(config)
        preprocess_fn = preprocess_cnn_dm

    train_dataset = train_dataset.map(preprocess_fn, batched=True)
    eval_dataset = eval_dataset.map(preprocess_fn, batched=True)

    if "label" in train_dataset.column_names:
        train_dataset = train_dataset.remove_columns("label")
        eval_dataset = eval_dataset.remove_columns("label")

    train_dataset = train_dataset.map(lambda x: tokenize_function(x, tokenizer))
    eval_dataset = eval_dataset.map(lambda x: tokenize_function(x, tokenizer))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=f"./results/{method}_{task}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    # Save model
    trainer.save_model(f"./models/{method}_{task}")

    # Count params
    trainable, total = count_trainable_params(model)
    print(f"Trainable params: {trainable}, Total: {total}")

if __name__ == "__main__":
    main()