# AGENTS.md

Guidance for AI coding agents working in this repo. Keep edits minimal and aligned with the patterns below.

## Project Snapshot

Single-script demo of a DistilBERT sentiment-classification MVP using Hugging Face `transformers` + `datasets` on top of PyTorch. Everything lives in `app.py`; there is no package layout, no tests, no CLI, no module boundaries.

- `app.py` — end-to-end script: inline toy dataset → `train_test_split` → `Dataset.from_pandas` → tokenize with `DistilBertTokenizer` → load `DistilBertForSequenceClassification` → (training/eval/save are commented-out `Trainer` blocks intended to be re-enabled).
- `requirements.txt` — pinned versions; treat these as authoritative.
- `ReadMe.md` — setup commands (Python 3.14 venv).

## Critical Conventions (project-specific)

1. **Script-style, top-level execution.** Code runs at import time — no `if __name__ == "__main__":` guard, no functions wrapping the pipeline. New steps go inline in `app.py` in pipeline order. Only `tokenize_function` is factored out because `Dataset.map` requires a callable.
2. **Logging, not print.** Use the module `logger` (`logging.basicConfig(level=INFO, format="%(asctime)s | %(levelname)s | %(message)s")`). For DataFrames/Datasets use `logger.info("Label:\n%s", obj)` (lazy `%s` formatting); f-strings are used only for scalars. Do not reintroduce `print` — note the commented `Trainer` blocks still use `print` and should be converted to `logger` when re-enabled.
3. **Determinism.** Always set both `np.random.seed(42)` and `torch.manual_seed(42)` near the top; reuse `random_state=42` in any sklearn split.
4. **Model + tokenizer pairing.** Use `DistilBertTokenizer` / `DistilBertForSequenceClassification` from `'distilbert-base-uncased'` with `num_labels=2`. The `AutoTokenizer`/Llama line is intentionally left commented as an alternative — keep it that way unless the user asks to switch.
5. **Tokenization contract.** `tokenize_function` uses `padding="max_length"`, `truncation=True`, `max_length=128`, and is applied via `dataset.map(..., batched=True)`. Preserve these args; downstream tensor shapes depend on them.
6. **Trainer is intentionally disabled.** The `TrainingArguments` / `Trainer` / `trainer.train()` / evaluation / `save_pretrained` blocks are commented out (see lines ~79–129). When re-enabling, keep `report_to="none"` (wandb must stay off) and the explicit `IntervalStrategy.EPOCH` / `SaveStrategy.EPOCH` imports already present at the top of the file.

## Dev Workflow

Python 3.14 venv (per `ReadMe.md`):

```bash
python3.14 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
export HF_TOKEN=...   # only if pulling gated models
python app.py
```

There are no tests, linters, or build steps configured. "Run = `python app.py`." When adding deps, pin them in `requirements.txt` in the same `package==x.y.z` style as the existing entries.

## When Extending

- Keep new code in `app.py` unless the user explicitly asks to modularize.
- Follow the section banner style already in the file (`# Part 1: Building an MVP from Scratch` + `# ---------` underline + matching `logger.info`).
- Reuse existing variables (`df`, `train_df`, `test_df`, `tokenized_train`, `tokenized_test`, `model`, `tokenizer`) rather than renaming.
- If you uncomment the `Trainer` block, also uncomment the matching evaluation/metrics/save blocks together — they form one logical unit and reference the same `trainer` / `predictions` variables.

