# LongMemEval Task Adaptation Guide

This page shows how to turn PlugMem from its task-agnostic general mode into a LongMemEval-specific setup with only light task adaptation.

The key message is simple: we do not rebuild the whole framework. We keep the PlugMem pipeline, but we make a few targeted changes so the system matches the user-agent conversation setting of LongMemEval much better. With these changes, our adapted setup reaches **90.2 Accuracy** on LongMemEval, which is higher than the **82.8** score of the task-agnostic general backbone. In other words, the same framework becomes stronger once it is adapted to the task.

We provided an adapted example in `eval/longmemeval/eval_longmemeval_sota.py`.

## Step 1: Switch the Extraction Prompt

In general mode, PlugMem extracts propositional knowledge with a broad prompt that works across tasks. For LongMemEval, we want the prompt to focus on user-agent conversational facts.

### What to change

Replace the general extraction prompt with a task-specific prompt that explicitly tells the agent to extract propositions about:

- user’s preferences,
- fact about user,
- what user has done,
- what agent has done

### Why it helps

In user-agent conversation task, whether the memory system stores the right kind of facts is important. A task-specific prompt reduces noise and pushes the extractor toward exactly the memories that matter in user-agent dialogue.

### Example prompt idea

You can adapt the prompt along these lines:

- offering few-shot example showing how to extract useful knowledge from converdation,
- separate user facts from agent facts,
- avoid summarizing the whole conversation when a smaller proposition is enough.

## Step 2: Split Text More Finely

The second change is about granularity. Instead of extracting from a large chunk of text, we split the conversation more carefully.

### Recommended strategy

Use one message as the basic extraction unit whenever possible.

That means the pipeline should consider each user turn and each agent turn separately, rather than waiting for a large block of dialogue to accumulate.

### Why it helps

Fine-grained segmentation makes the extracted propositions cleaner:

- fewer unrelated facts get mixed together,
- each memory item maps to a single conversational move,

For LongMemEval, this is especially useful because many questions depend on one specific turn.

## Step 3: Upgrade the Models

Once we have better extracted propositions, we should structure them with a stronger model and reason over them with a stronger backbone agent.

### Recommended choices

Use a stronger structuring model such as **Qwen3.5-27B**.

Use a stronger reasoning model and backbone agent such as **GPT-5.4**.
