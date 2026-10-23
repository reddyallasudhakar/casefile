# Compute costs: options and estimate

Written 2026-10-23, before creating the first GPU VM. Prices are approximate us-central1
on-demand rates from third-party trackers (they disagree by up to ~30%); check the
[official Compute Engine GPU pricing](https://cloud.google.com/compute/gpus-pricing) before
any big run. Paid from Policygaido's GCP credits; budget alerts are set on `casefile-ml`.

## What we actually need to run

| Work | Model size | GPU memory needed | Estimated GPU time |
|---|---|---|---|
| F2 baselines still to run (Qwen few-shot, free generation) | 0.5B | ~4 GB | ~10 min |
| F2 fine-tune (LoRA or full) on 569 pages × ~1.3k tokens × 3 epochs | 0.5B | ~8 GB | ~10–20 min |
| F2 evaluation of the fine-tuned model | 0.5B | ~4 GB | ~5 min |
| One-time VM setup (drivers, `uv sync`, model download) | — | — | ~20–30 min |
| **F2 total** | | | **~1–1.5 hours** |
| Later: QLoRA at 7B | 7B | ~16–20 GB | ~3–5 h |
| Later: 7B teacher for distillation | 7B | ~20+ GB | ~5 h |
| Later: GRPO on 1.5B | 1.5B | ~20+ GB | ~10–20 h |
| **Whole plan** | | | **~30–50 GPU-hours** |

## The options

| Option | ~$/hour | Good for | Catch |
|---|---|---|---|
| **T4** VM (n1-standard-4 + T4, 16 GB) | ~$0.35–0.55 | Cheap inference of small models | Old; no bf16 (training in fp16 is fragile); 16 GB is too small for 7B QLoRA later |
| **L4** VM (g2-standard-4, 24 GB), on-demand | ~$0.70 | Everything in this plan up to 7B QLoRA | — |
| **L4 VM, Spot** | ~$0.25–0.42 | Same, at roughly half price | Google can stop it with 30 s notice; fine for jobs that take minutes and can resume |
| **A100 40 GB** (a2-highgpu-1g) | ~$3.67 (Spot ~$1.10–2.20) | GRPO and the 7B teacher, if the L4 is too slow | 5× the L4's price; overkill for F2 |
| Vertex AI custom training jobs | VM price + a small markup | Reproducible runs that stop themselves | Needs a container image; worth it later, not now |
| Colab Enterprise (Vertex) | ~VM price | Exploring in notebooks; idle shutdown built in | Work lives in notebooks, not in git |
| Cloud Run with GPU | per second | Serving a model, scaling to zero | GPU for *jobs* wasn't confirmed generally available; Mumbai is invite-only |
| Kaggle notebooks | free (T4 ×2, ~30 h/week) | Zero-cost experiments | Outside GCP; less reproducible |

## Hidden costs to watch

- **Idle VMs.** A forgotten on-demand L4 costs ~$17 a day. The main risk to the budget is
  forgetting, not the training itself.
- **Disks bill even when the VM is stopped**: ~$0.10/GB-month for a balanced disk, so a
  50 GB disk is ~$5/month. Delete the VM when a phase is over.
- **Gemini API calls**: generating the 800 pages used roughly 4M output and thinking
  tokens (exact figure in the billing report). Classifying the test set costs cents.

## Decision

**An L4 Spot VM (g2-standard-4) in us-central1, with a 50 GB disk and a hard 4-hour
maximum run time**, after which Google stops it automatically.

- **L4 over T4:** about the same cost per finished job (it's ~2× faster), supports bf16 for
  stable training, and its 24 GB covers every model in the plan up to 7B QLoRA.
- **Spot over on-demand:** about half the price, and every F2 job takes minutes and can be
  re-run if Google reclaims the VM.
- **The 4-hour cap** means a forgotten VM costs at most ~$1–2, not ~$17 a day.
- **us-central1:** cheapest region with L4s and our quota (16). The data is synthetic, so
  there's no residency reason to use Mumbai.
- **Stopped between sessions, deleted after each phase.**
- **Revisit** at GRPO time: an A100 Spot may finish faster and cost less overall.

**Estimate:** F2 ≈ **$1–2** of GPU time. The whole plan ≈ **$15–40** on L4 Spot, plus
perhaps $20–40 if GRPO moves to an A100, plus disks and Gemini calls. That's well within the
budget alert.
