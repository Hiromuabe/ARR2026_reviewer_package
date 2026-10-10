# Semantic judgments and memory updates: code and data

This package contains the experiment code and synthetic data used to evaluate conflict judgments and JSON memory updates. It includes the 40 base problems, all 120 conflict cases, three wording families, actual prompts, saved model outputs, and scripts to recompute the reported results or replay inference. It contains no manuscript, reviewer correspondence, or earlier exploratory experiments.

## Recompute results without a GPU

Python 3.10 or newer is sufficient. Reanalysis uses only the Python standard library; no installation or network access is required.

```sh
python3 code/recompute.py --output-dir /tmp/memory-update-results
```

This checks file hashes, regenerates the 40 problems, derives the 120 labels and expected states, reconstructs all 2,720 prompts, re-parses and re-scores the saved outputs, checks the controlled intervention inputs and seeds, and reproduces counts and bootstrap intervals. Outputs are `recomputed.json`, `judgment.csv`, `arms.csv`, and `paired.csv`. The original source archive is not required. This compact package checks its own files; tracing exports to original source-file lines was checked separately before packaging.

## Files

| Path | Contents |
| --- | --- |
| `code/` | Problem construction, event catalogue, prompt templates, scoring, model loading, inference replay, and result recomputation |
| `data/basic_problems.json` | 40 synthetic again problems with initial states, utterances, intervals, aliases and three new-information types |
| `data/challenge_cases.json` | 120 cases and their expected updated JSON states |
| `data/variants.json` | Original wording, new-information paraphrase, and new-information plus instruction paraphrase |
| `data/prompts.jsonl` | Actual system/user messages and tokenizer-rendered inputs, with conditions, per-call seeds and token budgets |
| `data/records.jsonl` | Saved raw outputs, final answers and evaluations for 2,720 actual calls |
| `configs/experiments.json` | Effective generation settings and exact model revisions |
| `configs/expected_results.json` | Expected reported aggregates for checking reanalysis |
| `manifest.json` | SHA-256 checksums |

IDs and original source labels in saved records are provenance labels, not paths required at runtime. Inputs use synthetic people and situations. Four Qwen3-8B thinking shards partition the same 40 problems; they are not four independent repetitions. The three wording families also reuse those same 40 problems.

## Replay inference

First check the input plan without a GPU or model libraries:

```sh
python3 code/replay_inference.py --experiment intervention_original --dry-run
```

For new inference, install the recorded core versions from `requirements-inference.txt` in a suitable CUDA environment and obtain the exact model revision listed in `configs/experiments.json`. Model weights are not included. The loader reads an existing Hugging Face cache with `local_files_only=True` and does not download weights.

```sh
python3 code/replay_inference.py --experiment intervention_original --cache-dir /path/to/cache --output-dir /path/to/new-run
```

Use `intervention_paraphrase` and `intervention_paraphrase_instructions` for the other primary wording conditions. Other experiments are `qwen32_sampled_off`, `qwen32_sampled_on`, `qwen8_sampled_off`, `qwen8_sampled_on`, `qwen32_greedy`, and `llama8_greedy`. `--limit 1` selects a smoke run. Write new outputs outside this package.

By default, Self updates replay the historical saved judgment. `--fresh-self` generates independent judgments first and passes the new judgment to each Self update; an unparseable judgment causes the dependent update to be skipped. This changes the input if the new judgment differs. Gold/Flipped inputs use the stipulated labels.

## Experimental conventions

The primary intervention uses Qwen3-32B with thinking enabled, temperature 0.6, top-p 0.95, top-k 20, min-p 0, and 8,192 maximum output tokens including thinking. Its per-problem seed is `42 + 100 * numeric_problem_id`, shared by Gold and Flipped. Sampled judge/update comparisons use maximum lengths 4,096/8,192 and derive seeds from the base seed plus problem, update and condition offsets. Deterministic comparisons use maximum lengths 32/2,048 for judgment/update. Saved inputs specify every call's effective seed and budget.

The primary bootstrap uses 10,000 base-problem resamples, RNG seed 42, and historical sorted-draw indices 249/9749. Deterministic comparisons use 10,000 resamples, RNG seed 271828, and interpolated 2.5/97.5 percentiles. These conventions are preserved by the recomputation script.

Presupposition loss means failure to retain the target proposition in the presuppositions array under the registered interval ontology. It does not establish erasure from every state field or failure on later questions. Invalid outputs are counted separately; paired effects use cases valid in both conditions, with conservative bounds for the full 40 problems.

## Validation and environment

The recorded inference environment was Python 3.10.12, PyTorch 2.5.1+cu121, Transformers 5.17.0, CUDA 12.1, bfloat16, batch size 1, and NVIDIA A100-SXM4-80GB. The requirements file records core versions, not a complete historical dependency lock. Matching seeds do not guarantee identical newly sampled outputs across environments.

CPU reanalysis and input-plan dry runs are checked during packaging. The adapted replay runner has not been executed on a GPU. The compact operational config removes stale inherited thinking flags and uses the effective per-experiment setting; the archived outputs are unchanged.
