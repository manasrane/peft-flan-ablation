import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import yaml
import argparse
from data.agnews import load_agnews, preprocess_agnews
from data.cnn_dm import load_cnn_dm, preprocess_cnn_dm
from utils import count_trainable_params
import pandas as pd
import os
from sklearn.metrics import accuracy_score
from rouge_score import rouge_scorer

def load_model(method, task):
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")

    if method != "full_ft":
        if method in ["prefix", "prompt"]:
            model = PeftModel.from_pretrained(model, f"./models/{method}_{task}")
        else:
            model = PeftModel.from_pretrained(model, f"./models/{method}_{task}", is_trainable=True)

    return model, tokenizer

def evaluate_classification(model, tokenizer, eval_dataset):
    model.eval()
    predictions = []
    references = []

    for example in eval_dataset:
        input_text = example["input_text"]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=10)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(example["target_text"])

    # Map back to labels
    label_map = {"World": 0, "Sports": 1, "Business": 2, "Sci/Tech": 3}
    pred_labels = [label_map.get(p, -1) for p in predictions]
    ref_labels = [label_map.get(r, -1) for r in references]
    acc = accuracy_score(ref_labels, pred_labels)
    return acc

def evaluate_summarization(model, tokenizer, eval_dataset):
    model.eval()
    predictions = []
    references = []

    for example in eval_dataset:
        input_text = example["input_text"]
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(**inputs, max_length=128)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        predictions.append(pred)
        references.append(example["target_text"])

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        scores.append(score)
    rouge1 = sum(s['rouge1'].fmeasure for s in scores) / len(scores)
    rougeL = sum(s['rougeL'].fmeasure for s in scores) / len(scores)
    return rouge1, rougeL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    method = config['method']
    task = config['task']

    model, tokenizer = load_model(method, task)

    if task == "classification":
        _, eval_dataset = load_agnews(config)
        eval_dataset = eval_dataset.map(preprocess_agnews, batched=True)
        metric = evaluate_classification(model, tokenizer, eval_dataset)
        result = {"accuracy": metric}
    elif task == "summarization":
        _, eval_dataset = load_cnn_dm(config)
        eval_dataset = eval_dataset.map(preprocess_cnn_dm, batched=True)
        rouge1, rougeL = evaluate_summarization(model, tokenizer, eval_dataset)
        result = {"rouge1": rouge1, "rougeL": rougeL}

    trainable, total = count_trainable_params(model)

    # Save to CSV
    df = pd.DataFrame([{
        "method": method,
        "task": task,
        "trainable_params": trainable,
        "total_params": total,
        **result
    }])

    if not os.path.exists("experiments/results.csv"):
        df.to_csv("experiments/results.csv", index=False)
    else:
        existing = pd.read_csv("experiments/results.csv")
        combined = pd.concat([existing, df])
        combined.to_csv("experiments/results.csv", index=False)

if __name__ == "__main__":
    main()