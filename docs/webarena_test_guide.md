# WebArena 测试流程

这份文档总结本仓库中 WebArena 集成的测试方法。

README 中 WebArena 使用的是 `src/` 下的旧版评测栈，不是 `plugmem/`
下较新的 FastAPI 服务化包。

## 相关文件

- 入口脚本：`src/eval/webarena/eval_agentoccam.py`
- Agent 集成：`src/eval/webarena/plugmem_agent.py`
- Prompt：`src/eval/webarena/AgentOccamWithMemory_prompt.py`
- 配置文件：`src/eval/webarena/configs/*.yml`
- 记忆图：`src/memory_retrieving/memory_graph.py`
- 记忆结构化：`src/memory_structuring/memory.py`
- LLM / embedding 工具：`src/utils.py`
- WebArena 补丁：
  - `src/webarena_patch/envs.py`
  - `src/webarena_patch/openai_utils.py`

## 1. 安装外部仓库

`eval_agentoccam.py` 期望下面两个目录存在：

```text
src/AgentOccam
src/webarena
```

在 `src/` 下安装：

```bash
cd ~/PlugMemV3/src
git clone https://github.com/jizej/AgentOccam
git clone https://github.com/web-arena-x/webarena
```

然后按这两个仓库各自的 README 安装依赖。一般需要：

```bash
pip install -e ./AgentOccam
pip install -e ./webarena
pip install beartype gymnasium aiolimiter pyyaml openai
playwright install chromium
```

## 2. 应用 WebArena 补丁

从仓库根目录执行：

```bash
cd ~/PlugMemV3
cp src/webarena_patch/envs.py src/webarena/browser_env/envs.py
cp src/webarena_patch/openai_utils.py src/webarena/llms/providers/openai_utils.py
```

这两个补丁的作用：

- 让 `ScriptBrowserEnv.close()` 在当前执行流程中更稳定；
- 让 WebArena evaluator 支持 `OPENAI_API_KEY` + `AZURE_ENDPOINT`。

## 3. 配置环境变量

配置 LLM 相关变量：

```bash
export OPENAI_API_KEY="<your_openai_or_azure_key>"
export AZURE_ENDPOINT="<your_azure_endpoint>"   # 如果不用 Azure，可以 unset
export QWEN_BASE_URL="http://127.0.0.1:8000/v1"
export VLLM_QWEN_API_KEY="EMPTY"
```

配置 PlugMem 数据目录：

```bash
export DIR_PATH="$HOME/PlugMemV3/data"
mkdir -p "$DIR_PATH/episodic_memory" \
         "$DIR_PATH/semantic_memory" \
         "$DIR_PATH/procedural_memory" \
         "$DIR_PATH/tag" \
         "$DIR_PATH/subgoal"
```

## 4. 使用本地 Sentence-Transformers Embedding

如果不想启动较大的 `nvidia/NV-Embed-v2` embedding 服务，可以用较小的本地
sentence-transformers 模型。

安装：

```bash
pip install sentence-transformers
```

推荐的小模型：

```bash
export EMBEDDING_MODEL_NAME="BAAI/bge-small-en-v1.5"
# 或
export EMBEDDING_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
```

取消远程 embedding 变量，让 `src/utils.py` fallback 到本地
sentence-transformers：

```bash
unset EMBEDDING_BASE_URL
unset EMBEDDING_API_BASE_URL
```

使用本地 embedding 时，避免把 `OPENAI_BASE_URL` 设置成 Qwen chat 服务地址。
当前 `src/utils.py` 会把 `OPENAI_BASE_URL` 也当作 embedding API fallback。

如果需要使用 `EMBEDDING_MODEL_NAME`，需要修改 `src/utils.py` 的本地 fallback：

```python
return _get_embedding_local(
    text,
    model_name=embedding_model
    or os.environ.get("EMBEDDING_MODEL_NAME")
    or "nvidia/NV-Embed-v2"
)
```

测试本地 embedding：

```bash
cd ~/PlugMemV3
python - <<'PY'
import sys
sys.path.append("src")
from utils import get_embedding

emb = get_embedding("test local embedding")
print(type(emb), len(emb), emb[:5])
PY
```

如果已经用某个 embedding 模型写入过 memory 文件，后续更换模型时建议清空并
重建 memory graph。不同 embedding 模型的向量维度和空间不同，混用会影响检索，
甚至导致维度不一致报错。

## 5. 创建单任务测试配置

仓库自带的配置文件通常包含很多任务。测试时建议复制一个配置，只保留一个任务：

```bash
cd ~/PlugMemV3/src/eval/webarena
cp configs/AgentOccam_shopping_online.yml configs/test_one.yml
```

编辑 `configs/test_one.yml`：

```yaml
env:
  task_ids: [22]
```

也可以把 `max_steps` 调小，用来做快速 smoke test。

## 6. 先运行无 PlugMem 的 WebArena Smoke Test

先跑这一步。它用于验证 WebArena、AgentOccam、浏览器自动化和 LLM 访问是否正常，
暂时不引入 memory graph。

```bash
cd ~/PlugMemV3/src/eval/webarena
python eval_agentoccam.py \
  --config configs/test_one.yml \
  --disable-memory-graph
```

如果这一步失败，先排查基础 WebArena / AgentOccam 环境，再测试 PlugMem。

常见原因：

- 缺少 `src/AgentOccam` 或 `src/webarena`；
- 依赖没有安装完整；
- 没有执行 `playwright install chromium`；
- WebArena task / site 配置不可用；
- `OPENAI_API_KEY` 或 `AZURE_ENDPOINT` 配置错误。

## 7. 修复当前 PlugMem-WebArena 兼容问题

当前代码中，带 PlugMem 运行前需要处理两个函数签名不匹配的问题。

### 问题 1：MemoryGraph 构造函数参数不匹配

`eval_agentoccam.py` 当前传入：

```python
load_from_disk=args.load_memory_graph,
refresh_embeddings=args.refresh_embeddings
```

但 `src/memory_retrieving/memory_graph.py::MemoryGraph.__init__()` 不接受这两个
参数。

修复方式：创建 `MemoryGraph` 时不要传这两个参数，需要加载已有记忆时手动调用
加载函数：

```python
mg = MemoryGraph(
    tag_equal=TagEqual(),
    tag_relevant=TagRelevant(),
    semantic_equal=SemanticEqual(),
    semantic_relevant=SemanticRelevant(),
    subgoal_equal=SubgoalEqual(),
    subgoal_relevant=SubgoalRelevant(),
    procedural_equal=ProceduralEqual(),
    procedural_relevant=ProceduralRelevant(),
)

if args.load_memory_graph:
    mg.build_mem_from_disk_webarena_ver(
        os.environ["DIR_PATH"],
        refresh_embeddings=args.refresh_embeddings,
    )
```

### 问题 2：retrieve_memory 多传了参数

`plugmem_agent.py` 当前调用：

```python
mg.retrieve_memory(
    ...,
    task_id=self.current_task_id,
    step_idx=self.get_step()
)
```

但 `src/memory_retrieving/memory_graph.py::retrieve_memory()` 不接受 `task_id` 和
`step_idx`。

修复方式：删除这两个参数，或者把 `retrieve_memory()` 改成接受未使用的
`**kwargs`。

## 8. 运行带 PlugMem 的测试

修复兼容问题后执行：

```bash
cd ~/PlugMemV3/src/eval/webarena
python eval_agentoccam.py \
  --config configs/test_one.yml
```

带 PlugMem 的在线评测流程是：

```text
current observation
  -> retrieve_memory()
  -> retrieved memory 注入 AgentOccam actor prompt
  -> agent 预测 action
  -> environment 执行 action
  -> Memory.append(...)
  -> 任务结束后 Memory.close()
  -> MemoryGraph.insert(...)
```

## 9. 测试加载已有 Memory

从 `$DIR_PATH` 读取已有 memory graph：

```bash
python eval_agentoccam.py \
  --config configs/test_one.yml \
  --load_memory_graph
```

只读取已有 memory，不把当前任务轨迹写回：

```bash
python eval_agentoccam.py \
  --config configs/test_one.yml \
  --load_memory_graph \
  --read-only-memory
```

## 10. 测试 Trajectory Replay

如果已有保存的 AgentOccam trajectory，可以先 replay 轨迹来填充 PlugMem，再进行
在线评测：

```bash
python eval_agentoccam.py \
  --config configs/test_one.yml \
  --replay-trajectory \
  --trajectory-dir ./AgentOccam-Trajectories-demo/AgentOccam-debug
```

Replay 模式会从历史 step 构建 memory，然后插入 memory graph。

## 推荐测试顺序

```text
1. 在 src/ 下安装 AgentOccam 和 WebArena。
2. 应用 WebArena patch。
3. 配置 LLM、DIR_PATH 和本地 embedding。
4. 创建只包含一个 task 的 configs/test_one.yml。
5. 先运行 --disable-memory-graph。
6. 修复两个 PlugMem 兼容问题。
7. 运行带 PlugMem 的测试。
8. 运行 --load_memory_graph --read-only-memory，测试已有记忆复用。
```
