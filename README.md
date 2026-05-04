# PlugMem
<p align="center">
  <img src="assets/plugmem_name_card.png" alt="PlugMem"/>
</p>

**PlugMem** is a **plug-and-play long-term memory system for LLM agents**. Instead of storing and retrieving raw interaction histories, PlugMem organizes experience into **compact, reusable knowledge units**, allowing agents to recall what matters to agent decision-making with minimal context overhead.

The module is **task-agnostic by design** and can be integrated into existing agent pipelines with minimal effort, serving as a general memory backbone for diverse environments such as dialogue agents, knowledge-intensive QA, and web automation. 

For more details, please see the full paper: [https://arxiv.org/abs/2603.03296](https://arxiv.org/abs/2603.03296)

<p align="center">
  <img src="assets/plugmem_pipeline.png" alt="PlugMem Pipeline"/>
</p>

## Table of Contents

- [Updates](#updates)
- [Features](#features)
  - [Plug-in](#plug-in)
  - [Memory](#memory)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducibility](#reproducibility)
- [Citation](#citation)

## Updates

- **[2026-05]** 🚀 **Plugin release** — PlugMem now ships as installable plugins for AI coding agents.
  Integrations available for **[OpenClaw](openclaw-plugmem-plugin/)** and **Claude Code** (see `plugin` branch).
  Highlights: inspect your memory graph, test retrieval interactively, and replay past agent sessions.

  <p align="center">
    <img src="assets/plugmem_promotion_headline.png" alt="PlugMem Plugin" width="700"/>
  </p>

- **[2026-05]** 🏆 **New SOTA on LongMemEval & HotpotQA** — With light task adaptation, PlugMem reaches
  **90.2 Acc** on LongMemEval and **79.1 F1 / 91.1% LLM-Judge Acc** on HotpotQA (multi-hop), both
  state-of-the-art results. Because the framework is task-agnostic, it can serve as a drop-in backbone for other work on these benchmarks. → [Step-by-step reproduction guide](examples/task-adaptation/)

- **[2026-04]** 🎉 **PlugMem accepted to ICML 2026!**

## Features
### Plug-in
- **Enhance your agent with 6 lines of code**
```python
# init PlugMem memory graph
mg = MemoryGraph()
# init memory sequence
mem = Memory(...)
mem.append(...)
mem.close()
# insert memory sequence into memory graph
mg.insert(mem)
# retrieve memory and perform reasoning on retrieved nodes
mg.retrieve_and_reason(...)
```
- **Easy to modify**: Apply adaptive strategies by defining different value functions and reasoning prompts.
- **Agent integrations**: Native plugins available for **[OpenClaw](openclaw-plugmem-plugin/)** and **Claude Code** (see `plugin` branch), with a built-in **Memory Inspector** UI for visualizing the memory graph, browsing individual memories, testing retrieval, and replaying agent trajectories.

<p align="center">
  <img src="assets/plugmem_memory_inspector.png" alt="Memory Inspector — Graph view" width="800"/>
  <br/><em>Graph view: explore the full memory graph across semantic, procedural, and episodic nodes</em>
</p>

<p align="center">
  <img src="assets/plugmem_memory_inspector_2.png" alt="Memory Inspector — Browse view" width="800"/>
  <br/><em>Browse view: inspect, filter, and manage individual memory entries</em>
</p>

### Memory
- **Three Memory Types**: 
  - **Semantic** (facts, concepts): User preferences, factual information
  - **Procedural** (workflows, procedures): How-to knowledge, step-by-step processes
  - **Episodic** (interaction sequences): Long interaction sessions stored on disk, referenced by ID
- **Graph Structure**: Maintain hierarchical knowledge units to illustrate the relationship between memories.
- **LLM Enhancement**: Use LLMs for intelligent knowledge extraction, memory retrieval, and reasoning
- **Memory Compression and Evolution**: Naively support updating and evolving the memory graph.

<p align="center">
  <img src="assets/plugmem_structuring.png" alt="PlugMem Structuring"/>
</p>

## Installation
1. Install benchmarks in `src/` and follow their installation docs to set up the environment.
2. Install/upgrade `openai==2.6.1`.
3. Additional modifications:
- **WebArena**
```bash
# under src/
cd src
# clone modified AgentOccam
git clone https://github.com/jizej/AgentOccam
# clone 
git clone https://github.com/web-arena-x/webarena
# Enable Scriptbrowserenv to run under async loop (if needed)
cp src/webarena_patch/envs.py src/webarena/browser_env/envs.py
# Enable OPENAI_API_KEY + AZURE_ENDPOINT for trajectory evaluation (if needed)
cp src/webarena_patch/openai_utils.py src/webarena/llms/providers/openai_utils.py
```

## Quick Start
1. Export Parameters
```bash
export OPENAI_API_KEY=<your_openai_api_key>
export AZURE_ENDPOINT=<your_azure_endpoint>
export DIR_PATH="/<your_path_to_PlugMem>/data"
export QWEN_BASE_URL="http://<your_qwen_host>:8000/v1"
export EMBEDDING_BASE_URL="http://<your_embedding_host>:8001/v1/embeddings"
```
2. Host local inference servers (Qwen + Embedding)
```bash
cd host_local_inference
# Qwen (vLLM) server
bash vllm_deploy.sh
# NV-Embed-v2 server
bash nv_embed_v2_deploy.sh
```
3. Make needed directory
```bash
mkdir -p "$DIR_PATH/episodic_memory" \
         "$DIR_PATH/semantic_memory" \
         "$DIR_PATH/procedural_memory" \
         "$DIR_PATH/tag" \
         "$DIR_PATH/subgoal"
```
4. Run examples for different benchmarks
   ### WebArena
   ```bash
   cd src/eval/webarena
   python eval_agentoccam.py
   ```
   Options for `eval_agentoccam.py`:
   - `--config`: Path to the YAML config file (required).
   - `--replay-trajectory/--no-replay-trajectory`: Replay a saved trajectory before evaluation.
   - `--trajectory-dir`: Directory containing trajectory JSON files for replay.
   - `--load_memory_graph/--no-load_memory_graph`: Load a persisted memory graph from disk.
   - `--refresh-embeddings/--no-refresh-embeddings`: Refresh embeddings when loading the memory graph.
   - `--read-only-memory/--no-read-only-memory`: Use the memory graph without inserting new memories.
   - `--disable-memory-graph/--no-disable-memory-graph`: Turn off all memory-graph operations.
   ### LongMemEval
   ```bash
   cd src/eval/longmemeval
   python eval_longmemeval_all.py
   ```
   ### HotpotQA(HippoRAG 2)
   ```bash
   cd src/eval/hotpotqa
   # It may take several hours to structure memory for hotpotqa_corpus.json.
   python build.py
   #Rebuild the memory graph from structuring result and run test
   python eval_hotpotqa_all.py
   ```

## Reproducibility
- We release agent trajectories and memory graph artifacts for all three tasks.
- We release human demonstrations used for WebArena (Under License CC BY 4.0).
- Data available in Google Drive: https://drive.google.com/drive/folders/15feC6xYsONJhJAb2n1kPjGrjSt0weHXi?usp=sharing

## Citation
If you use our code or data, or otherwise found our work helpful, please cite our paper:

```
@misc{yang2026plugmemtaskagnosticpluginmemory,
      title={PlugMem: A Task-Agnostic Plugin Memory Module for LLM Agents}, 
      author={Ke Yang and Zixi Chen and Xuan He and Jize Jiang and Michel Galley and Chenglong Wang and Jianfeng Gao and Jiawei Han and ChengXiang Zhai},
      year={2026},
      eprint={2603.03296},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2603.03296}, 
}
```
