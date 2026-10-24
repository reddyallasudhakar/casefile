# RL primer: from policy gradients to GRPO

Written for F5, where GRPO is trained on top of a fine-tuned extraction model. Each idea
builds on the previous one; the code is [bench/grpo.py](../bench/grpo.py).

## 1. Why RL after fine-tuning at all?

**Supervised fine-tuning (SFT)** shows the model one correct answer per page and says "make
this more likely". It never sees its *own* mistakes, so it never learns which of its habits
cause errors.

**Reinforcement learning** flips that: the model writes answers itself, a program scores them,
and training pushes up whatever scored well. That's useful when:

- **a program can check the answer** (here: did it extract ₹1,53,700 exactly?), and
- **the model already gets it right sometimes**, so there's something to reinforce.

That's the "RL with verifiable rewards" idea behind DeepSeek-R1 and recent reasoning models,
applied here to reading claim documents.

## 2. The language model is a policy

In RL terms, the model is a **policy** π: given a state (the prompt plus the tokens written so
far), it gives a probability for each possible action (the next token). A whole answer is a
sequence of actions, and its probability is the product of each token's probability, or in
logs, the **sum of token log-probabilities**. That's `completion_logprob` from Module 1.

## 3. Policy gradient (REINFORCE)

Goal: maximise the expected reward of the model's answers. The policy-gradient theorem says the
gradient of that expectation is

```
∇ E[reward] = E[ reward × ∇ log π(answer) ]
```

In words: **sample an answer, then nudge its log-probability up in proportion to its reward.**
High-reward answers become more likely; low-reward ones barely move.

**Problem:** if every reward is positive (our rewards are between 0 and 1.1), every sampled
answer gets pushed up, including bad ones. Learning is slow and noisy.

## 4. Baselines and advantages

The fix is to subtract a **baseline** b, the reward you'd expect anyway:

```
∇ ≈ (reward − b) × ∇ log π(answer)
```

`reward − b` is the **advantage**: how much better this answer was than expected. Better than
expected → pushed up; worse → pushed *down*. Subtracting a baseline doesn't bias the gradient,
but it cuts the noise dramatically.

## 5. PPO, briefly

PPO (used in classic RLHF) adds two things:
- a **critic**, a second network trained to predict the expected reward, used as the baseline;
- **ratio clipping**, which limits how far one update can move the policy when the same batch
  of samples is reused for several gradient steps.

It works, but it means training and holding a second model (the critic) in memory.

## 6. GRPO: the group is the baseline

GRPO (from DeepSeekMath) removes the critic. For each prompt, sample a **group** of G answers.
The baseline is simply the group's average reward, normalised by its spread:

```
advantage_i = (reward_i − mean(group)) / std(group)
```

If 8 attempts at the same page score [1.1, 1.1, 0.7, 1.1, 0.5, 1.1, 1.1, 0.7], the answers that
got everything right get positive advantages and the ones that slipped get negative ones: the
model learns *exactly what distinguishes its good attempts from its bad attempts on that page.*

Two consequences:
- **No critic needed.** Cheaper, simpler.
- **A group where every answer scores the same teaches nothing.** If the model gets a page
  right 8 times out of 8 (or wrong 8 out of 8), all advantages are 0. GRPO learns only from
  pages where the model is *inconsistent*. A model that's already near-perfect has little
  left to learn from.

## 7. What `bench/grpo.py` does

1. Start from the SFT model (merge its adapter), and add a fresh LoRA to train.
2. Each step: 8 training pages × 8 sampled answers at temperature 1.
3. Reward each answer: share of gold fields correct + 0.1 for valid JSON.
4. Advantages per group; skip groups with no spread.
5. Loss = −advantage × mean token log-probability of the answer; one gradient step per batch.
   Each batch is used once, so PPO's clipping has nothing to clip and is left out. There's no
   KL penalty towards the starting model (following Dr. GRPO and DAPO); a small learning rate
   keeps updates gentle instead.
6. Every 20 steps: validation page accuracy (OCR, greedy decoding), and save the adapter.

## 8. Reward hacking: what to watch

A model optimises *the reward you wrote*, not the goal you meant. Weak spots in ours:

- **Fields without a gold value aren't checked.** If a policy has no room-rent limit, nothing
  penalises the model for inventing one. So: track how often it fills fields that should be null.
- **The JSON bonus** could be farmed by outputting valid but empty JSON; the field share keeps
  that from paying off, but watch the valid-JSON rate against field accuracy.
- **Length drift:** answers getting longer or shorter over training is an early warning sign.

`bench/grpo.py` logs mean reward, sample accuracy, informative groups and answer length every
step, and saves sample answers at the end, for exactly this check.

## Further reading (in llm-training-lab/papers)

- PPO (2017) · InstructGPT (2022): classic RLHF
- DeepSeekMath (2024): introduces GRPO · DeepSeek-R1 (2025): RL with verifiable rewards at scale
- Dr. GRPO, DAPO (2025): fixes to GRPO's biases; why we drop the KL term and length-normalise carefully
