import argparse
import json
import csv
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--test_path", type=str, default="data/processed/test.jsonl")
parser.add_argument("--output_path", type=str, default="")
args = parser.parse_args()

MODEL_PATH = args.model_path
TEST_PATH = args.test_path
run_name = Path(MODEL_PATH).name
OUTPUT_PATH = args.output_path or f"data/predictions/{run_name}_predictions.csv"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")
print(f"Model: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()

Path("data/predictions").mkdir(parents=True, exist_ok=True)

def extract_decision(text):
    text = text.strip()
    if "DECISION:" in text:
        after = text.split("DECISION:")[1].strip()
        return after.split("\n")[0].strip()
    return ""

rows = []

with open(TEST_PATH, "r", encoding="utf-8") as f:
    for line in tqdm(f):
        ex = json.loads(line)

        paper_id = ex["paper_id"]           # ✅ guaranteed
        input_text = ex["input_text"]
        true_output = ex["target_text"]

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=768
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=4
            )

        pred_text = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        rows.append({
            "paper_id": paper_id,
            "input_text": input_text,
            "true_decision": extract_decision(true_output),
            "pred_decision": extract_decision(pred_text),
            "true_meta_review": true_output,
            "pred_meta_review": pred_text
        })

with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved predictions to {OUTPUT_PATH}")