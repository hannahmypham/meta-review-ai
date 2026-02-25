# Meta-Review AI

Automated generation of ICLR meta-reviews and accept/reject decisions using fine-tuned seq2seq models.

Given a paper's reviews (ratings, strengths, weaknesses, summaries, questions), the model produces a meta-review and a final decision — mimicking what an Area Chair would write.

## Dataset

**Source:** ICLR 2025 official reviews scraped from the [OpenReview API](https://openreview.net/).

| Split | Papers |
|-------|--------|
| Train | 6,890  |
| Val   | 862    |
| Test  | 862    |
| **Total** | **8,614** |

Label distribution: 57% Reject / 43% Accept.

Raw data: [Google Drive](https://drive.google.com/file/d/16M1sQygtZo7zKGpfb5b81FUx2mTRRD4Y/view?usp=sharing)

See [DATADICTIONARY_README.md](DATADICTIONARY_README.md) for column descriptions.

## Models

| Run | Model | Config | Status |
|-----|-------|--------|--------|
| run1 | `google/flan-t5-base` | `configs/run1_finetune.yaml` | Done |
| run2 | `facebook/bart-large-cnn` | `configs/run2_bart.yaml` | Training |
| run3 | `google/pegasus-large` | `configs/run3_pegasus.yaml` | Planned |

All models are fine-tuned with **LoRA** (via PEFT) for parameter-efficient training.

## Architecture

```mermaid
flowchart TB
    subgraph Data["Data Pipeline"]
        A[OpenReview API] -->|OpenReviewDataExtract.py| B[(Raw CSV<br>~8.6K papers, ~30K reviews)]
        B -->|build_dataset.py| C1[Group reviews by paper_id]
        C1 --> C2[Build structured prompts<br>title + abstract + aggregates + reviews]
        C2 --> C3[Build targets<br>DECISION + META_REVIEW]
        C3 --> C4[Stratified split 80/10/10]
        C4 --> D[(train.jsonl / val.jsonl / test.jsonl)]
    end

    subgraph Train["Training (train_flan_t5.py)"]
        E[Load YAML config] --> F[Load pretrained model<br>FLAN-T5 / BART / Pegasus]
        E --> T0[Load tokenizer]
        T0 --> J[Tokenize dataset ds.map<br>input_ids + labels]
        F --> G{LoRA enabled?}
        G -- Yes --> H[Apply LoRA adapter<br>BART/Pegasus: q_proj, v_proj<br>T5: q, v]
        G -- No --> I[Use full model]
        H --> K[Seq2SeqTrainer<br>TrainingArgs: eval/save steps,<br>grad accumulation, predict_with_generate]
        I --> K
        J --> K
        K --> L[(Saved run folder<br>runs/run_name/<br>weights + tokenizer + config)]
    end

    subgraph Predict["Inference (generate_predictions.py)"]
        L --> M[Load trained model + tokenizer]
        D --> N[Read test.jsonl]
        N --> O[Generate predictions<br>num_beams=4, max_new_tokens=256]
        M --> O
        O --> P[Extract decision + meta-review]
        P --> Q[(predictions.csv<br>paper_id / true / pred)]
    end

    subgraph Eval["Evaluation"]
        Q --> R[Decision metrics<br>Accuracy / F1 / Precision / Recall]
        Q --> S[Text metrics<br>ROUGE-1 / ROUGE-2 / ROUGE-L / BERTScore]
        R --> EVAL_OUT[(eval_metrics.json)]
        S --> EVAL_OUT
        EVAL_OUT --> U[Compare models<br>select best]
    end

    D --> Train
```

## Project Structure

```
meta-review-ai/
├── configs/                        # YAML training configs per model
├── data/
│   ├── raw/                        # Raw review CSV (not tracked)
│   ├── processed/                  # Train/val/test JSONL (not tracked)
│   └── predictions/                # Model predictions CSV
├── src/
│   ├── preprocessing/
│   │   └── build_dataset.py        # CSV → structured JSONL (one row per paper)
│   ├── train_flan_t5.py            # Fine-tune any seq2seq model with LoRA
│   ├── generate_predictions.py     # Run inference on test set
│   └── eval_flan_t5.py             # Evaluation metrics (TODO)
├── runs/                           # Saved model checkpoints (not tracked)
├── requirements.txt
└── DATADICTIONARY_README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection (optional — raw CSV already available)

```bash
python data/predictions/OpenReviewDataExtract.py
```

### 2. Preprocessing

Converts the review-level CSV into paper-level JSONL with structured prompts and train/val/test splits.

```bash
python src/preprocessing/build_dataset.py --csv_path data/raw/iclr_2025_detailed_reviews.csv
```

### 3. Training

Pass a config file to select which model to train:

```bash
# FLAN-T5-base
python src/train_flan_t5.py --config configs/run1_finetune.yaml

# BART-large-CNN
python src/train_flan_t5.py --config configs/run2_bart.yaml
```

### 4. Generate Predictions

```bash
python src/generate_predictions.py --model_path runs/flan_t5_run1
python src/generate_predictions.py --model_path runs/bart_large_cnn_run2
```

## Input/Output Format

**Input** (structured prompt per paper):
```
TASK:
Write an ICLR meta-review based on the paper and reviewer feedback.
Also output a final decision.

PAPER TITLE: ...
ABSTRACT: ...
AGGREGATES:
- NumReviews: 4
- MeanRating: 5.25
- RatingRange: 3 to 8

REVIEWS:
REVIEW 1:
FinalRating: 3
Summary: ...
Strengths: ...
Weaknesses: ...
```

**Output** (model target):
```
DECISION: REJECT
META_REVIEW:
The paper proposes ... however reviewers raised concerns about ...
```

## Requirements

- Python 3.10+
- PyTorch
- Transformers, PEFT, Datasets (HuggingFace)
- See `requirements.txt` for full list
