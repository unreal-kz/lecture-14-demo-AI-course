# Lecture 14.1 demo — from policy gradients to RLHF, DPO and GRPO

Companion to **Lecture 14.1, Policy Optimization and Reinforcement Learning for Language Models** (CSS4007 Artificial
Intelligence, Narxoz University). It is the toys from the lecture, in code you can change. **Nothing is graded and
nothing is handed in.**

**No API key. No GPU. No installation. No cost.** The notebook needs numpy only, which Colab already has.

## Open it

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/unreal-kz/lecture-14-demo-AI-course/blob/main/lecture14_policy_llm_rl.ipynb)

Run the cells in order. A **✏️ cell** is one you are meant to edit. The first code cell holds all the toys
(`policy_toys.py`, pasted unchanged); you only need to run it.

## What is in it

1. One customer, as REINFORCE sees them: returns, baseline, which way the policy is pushed.
2. What a baseline does to the noise of the gradient.
3. REINFORCE with and without the baseline.
4. One small batch reused for many steps, with and without PPO's clip. ✏️ Try other clip values.
5. A reward model fitted to comparisons between answers.
6. Optimising against that reward model with a KL penalty.
7. DPO: the same answer probabilities from a single loss.
8. GRPO's group-relative advantage. ✏️ Make the question harder.
9. LoRA parameter arithmetic.

Every number about these toys on the lecture slides came from this code with the seeds in the notebook.

## What these toys are not

The "language model" has five possible answers and its probabilities are a table. The toys show how the methods
behave, and where they can go wrong, in a case small enough to check by hand. They say nothing about how large models
behave or how well any method works on them.

## About the numbers

Everything is synthetic: a firm and answers that do not exist. The reference run (all cells) took about 10 seconds on
an Apple-silicon Mac with numpy 2.5.3. A Colab run has not been timed.

## Files

| File | What it is |
|---|---|
| `lecture14_policy_llm_rl.ipynb` | The demo notebook, no outputs saved |
| `policy_toys.py` | The toys; the same text is the first code cell of the notebook |
| `README.md` | This file |
