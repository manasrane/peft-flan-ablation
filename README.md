# PEFT Ablation on Flan-T5-small

This project compares Parameter-Efficient Fine-Tuning (PEFT) methods on Flan-T5-small for classification and summarization tasks.
Conducted a systematic parameter-efficiency study on Flan-T5-small comparing full fine-tuning, LoRA, prefix-tuning, and prompt-tuning across classification and summarization tasks; demonstrated that LoRA achieves near-full performance with <3% trainable parameters, supported by reproducible experiments and performance–parameter tradeoff analysis.

## Methods Compared
- Full Fine-Tuning
- LoRA
- Prefix Tuning
- Prompt Tuning

## Tasks
- **Classification**: AG News dataset (accuracy)
- **Summarization**: CNN/DailyMail dataset (ROUGE scores)

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python train.py --method <method> --task <task>`
3. Evaluate: `python evaluate.py`
4. Plot results: `python plot_results.py`

## Results
See `experiments/results.csv` and the analysis notebook.
