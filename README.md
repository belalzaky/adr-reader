# ADR Reader — can a model read drug safety from text?

Most adverse-drug-event (ADE) signals aren't in tidy tables — they're buried in free text: case reports, patient notes, literature. This project asks a simple question with a surprising answer: **which kind of model best decides whether a sentence describes a drug side effect — and can it pull out the drug and the effect?**

I benchmarked four approaches on the [ADE Corpus](https://huggingface.co/datasets/SetFit/ade_corpus_v2_classification), building each one up lap by lap and measuring honestly.

## The result

| Approach | What it is | F1 (ADE class) |
|---|---|---|
| **Domain AI** | PubMedBERT embeddings + logistic regression | **0.83** |
| Keyword model | TF-IDF + logistic regression | 0.79 |
| General AI | MiniLM (general-purpose) embeddings + logistic regression | 0.73 |
| LLM (zero-shot) | llama3.2 via Ollama, prompted for JSON | 0.71 |

**The headline: bigger and newer isn't automatically better.** A five-line keyword model beat a general-purpose transformer, and the winner was the model trained on the *right kind of text* (biomedical) — not the biggest one. Domain fit + task fit beat raw size.

## What each lap taught

- **Lap 1 — baseline (TF-IDF + logistic regression).** Met the accuracy paradox: on imbalanced data (only ~29% of sentences are ADEs), a model that mostly shrugs "no" scores high accuracy while missing a third of real harms. Judged on precision/recall/F1 for the ADE class instead; `class_weight="balanced"` traded precision for recall (0.66 → 0.85) — the right trade in drug safety, where a missed harm costs more than a false alarm.
- **Lap 2 — transformer embeddings (same classifier, smarter features).** A *general* transformer's embeddings **lost** to keywords (F1 0.73). Swapping in a **biomedical** model (PubMedBERT) won (0.83). The variable that mattered was domain fit, not model size.
- **Lap 3 — zero-shot LLM extraction.** Prompted an LLM to return structured JSON (is-ADE, drug, effect). It came **last** at classification, over-flagged (low precision), and occasionally hallucinated — e.g. reporting a side effect while unable to name a drug. But it's the only approach that can *extract* the drug and effect, not just classify. Wrong tool for classification; right tool for structuring.

## Honest caveats

- The LLM was **zero-shot** — it saw none of the training data the others learned from, so this is not a like-for-like race; its score is on a 150-sentence sample. Given that, landing close is genuinely impressive.
- The gold labels are themselves noisy — some "correct" answers are debatable, which caps how good any model can look.

## Live demo

An interactive **example explorer** — pick a real sentence and watch the four models agree or disagree, with the drug and side effect highlighted: **[belalzaky.uk](https://belalzaky.uk)** (Projects → ADR Reader).

## Stack

Python · pandas · scikit-learn · sentence-transformers (MiniLM, PubMedBERT) · Ollama (llama3.2)

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install pandas scikit-learn datasets sentence-transformers ollama jupyterlab
jupyter lab   # open adr-reader.ipynb
```

*Built in public — part of an ongoing series turning drug-safety data into things you can argue with.*
