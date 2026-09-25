# Data

This directory holds evaluation data and generated reports.

## Dataset used for model evaluation

| Field | Value |
|---|---|
| Name | TweetEval — `sentiment` task (SemEval-2017 Task 4A) |
| Source | https://github.com/cardiffnlp/tweeteval (dataset files under `datasets/sentiment/`) |
| Mirror | https://huggingface.co/datasets/tweet_eval (`sentiment` config) |
| License | Creative Commons Attribution 3.0 Unported (CC BY 3.0) |
| Classes | `negative` (0), `neutral` (1), `positive` (2) |
| Splits | train 45,615 / validation 2,000 / test 12,284 |
| Language | English (Twitter) |

The label mapping in this dataset (0=negative, 1=neutral, 2=positive) is
**identical** to the model's own label mapping, so **no label remapping or
manipulation is performed** during evaluation.

Downloaded automatically by `scripts/evaluate_model.py` into `data/dataset/`
(git-ignored). Generated reports are written to `data/`:

- `model_evaluation_report.txt` — accuracy, precision, recall, F1, class-wise metrics
- `confusion_matrix.csv` — confusion matrix as CSV
