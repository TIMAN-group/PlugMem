# HiAgent 与 PlugMem 集成分阶段执行计划

## 总体目标

基于以下两份设计文档，分阶段实现 HiAgent 与 PlugMem 的跨任务记忆集成：

- `docs/plugmem_cross_task_memory_flow.md`
- `docs/plugmem_trajectory_api_initial_observation.md`

总体技术目标不变：

1. PlugMem trajectory API 支持 `initial_observation`。
2. HiAgent 启动任务时从 PlugMem recall 跨任务经验，并注入 prompt。
3. HiAgent 任务结束后把 trajectory 上传到 PlugMem。
4. recall/upload 失败不能中断 HiAgent 评测流程。

## Codex 执行原则

- 每次只执行一个阶段。
- 不要一次性完成全部集成。
- 阶段测试通过并 commit 后，再进入下一阶段。
- 不做无关重构。
- 不格式化整个文件。
- 不修改当前阶段未列出的文件，除非先说明原因。
- 每阶段结束后输出：修改文件、diff 摘要、测试命令、风险点和未完成项。

## 项目拆分与执行位置

本计划分成两个独立项目执行：

- PlugMem 项目：只执行阶段 1，在 PlugMem 独立仓库根目录操作和提交。
- HiAgent 项目：执行阶段 2、阶段 3、阶段 4，在 HiAgent 仓库根目录操作和提交。

不要在 HiAgent 仓库中提交复制版 `PlugMem/` 目录。阶段 2 之后的 HiAgent 联调依赖阶段 1 已经在 PlugMem 独立项目中完成并可启动服务。

## PlugMem 项目：阶段 1 API 修复与测试

执行位置：PlugMem 独立仓库根目录。

### 阶段目标

修复 PlugMem HTTP trajectory API 的初始 observation 语义，使其与 PlugMem 原生逻辑一致：

```python
Memory(goal=goal, observation=obs_0)
append(action_t0=action_1, observation_t1=obs_1)
append(action_t0=action_2, observation_t1=obs_2)
```

同时保持旧 payload 兼容。

### 允许修改的文件

- `plugmem/api/schemas.py`
- `plugmem/api/routes/memories.py`
- `tests/test_api_memories.py`

### 禁止修改的内容

- 不修改 HiAgent 侧任何文件。
- 不修改 PlugMem retrieval/reasoning 逻辑。
- 不修改 Chroma storage、MemoryGraph、Memory 的核心结构化逻辑。
- 不引入新依赖。
- 不改变旧 trajectory payload 的兼容行为。

### 核心实现要求

在 `MemoryInsertRequest` 中新增可选字段：

```python
initial_observation: Optional[str] = None
```

在 `_insert_trajectory()` 中使用：

```python
initial_observation = body.initial_observation or body.steps[0].observation
```

并用它构造 `Memory`：

```python
mem = Memory(
    goal=body.goal,
    observation=initial_observation,
    llm=llm,
    embedder=embedder,
    time=graph.semantic_time,
    session_id=body.session_id,
)
```

`steps` 的空值校验保持不变。即使提供 `initial_observation`，没有 `steps` 也应该返回错误。

测试至少覆盖：

- 新 payload 带 `initial_observation` 时可正常写入。
- 旧 payload 不带 `initial_observation` 时仍可正常写入。
- 空 `steps` 仍然失败。

### 验收标准

- `MemoryInsertRequest` 支持 `initial_observation`。
- 新旧 payload 都能通过 `/graphs/{graph_id}/memories` 写入。
- 新测试能证明新字段没有破坏旧格式。
- PlugMem memories API 相关测试通过。

### 建议测试命令

```powershell
pytest tests/test_api_memories.py
```

如环境无法运行 pytest，需要在阶段结束说明阻塞原因和未验证风险。

### 建议 commit message

```text
fix(plugmem): support initial observation in trajectory API
```

## HiAgent 项目：阶段 2 PlugMem 接入

执行位置：HiAgent 仓库根目录。

前置条件：阶段 1 已在 PlugMem 独立项目中完成，PlugMem 服务端支持 `initial_observation`。

### 阶段目标

在 HiAgent agent 层接入 PlugMem recall 和 upload 能力，但暂不修改 PDDL 任务循环和 yaml 配置。

本阶段只完成 agent 能力：

- 新增 `PlugMemClient`
- 新增 `PlugMemContextEfficientAgent`
- `reset()` 时 recall
- `make_prompt()` 时注入 `Past task hints`
- `remember_current_task()` 时整理 trajectory payload
- 注册新 agent

### 允许修改的文件

- `agentboard/agents/plugmem_client.py`
- `agentboard/agents/plugmem_agent.py`
- `agentboard/agents/__init__.py`

如确实需要读取公共工具或 logger，可说明原因后最小修改相关文件。

### 禁止修改的内容

- 不修改 `agentboard/tasks/pddl.py`。
- 不修改任何 `eval_configs/` 配置文件。
- 不修改 PlugMem 侧文件。
- 不改变 `ContextEfficientAgentV2` 原有行为。
- 不改变现有 agent 的注册名和默认加载方式。

### 核心实现要求

新增轻量 HTTP client：

- `recall(observation, goal, task_type="", session_id=None)`
  - 调用 `POST /api/v1/graphs/{graph_id}/reason`
  - 返回响应中的 `reasoning`

- `upload_trajectory(goal, initial_observation, steps, session_id=None)`
  - 调用 `POST /api/v1/graphs/{graph_id}/memories`
  - payload 使用 `initial_observation`

新增 `PlugMemContextEfficientAgent`，继承 `ContextEfficientAgentV2`。

`reset()` 要求：

- 先调用父类 `reset()`。
- 如果 `plugmem.enabled` 为 true 且 `recall_on_reset` 为 true，则调用 recall。
- recall 成功后保存到 `self.plugmem_context`。
- recall 失败只记录 warning，并把 `self.plugmem_context` 置空。

`make_prompt()` 要求：

- 保留原始 HiAgent prompt 结构。
- 在当前任务 history 前注入：

```text
Past task hints:
{plugmem_context}

These hints come from previous tasks and may help with the current task.
Use relevant strategies and patterns when choosing the next Subgoal or Action.
```

- 如果 `plugmem_context` 为空，不插入该区域。

`remember_current_task()` 要求：

- 方法内部负责判断 PlugMem 上传开关，而不是让 PDDL 任务层判断。
- 以下情况应直接 no-op 返回，不发起 `/memories` 请求：
  - `plugmem.enabled=false`
  - `plugmem.upload_on_finish=false`
  - PlugMem client 未初始化
  - 当前 trajectory 为空或无法构造有效 payload
- 从 `self.memory` 提取 observations 和 actions。
- 生成：

```python
initial_observation = observations[0]
steps = [
    {"action": actions[i], "observation": observations[i + 1]}
    for i in range(min(len(actions), len(observations) - 1))
]
```

- 不上传 HiAgent 自身生成的 subgoal。
- 不使用 synthetic no-op step。
- upload 失败只记录 warning，不抛出到评测流程。

注册新 agent：

```python
from .plugmem_agent import PlugMemContextEfficientAgent
```

确保 `--agent PlugMemContextEfficientAgent` 可以被 registry 加载。

### 验收标准

- 新 agent 可以通过 registry 加载。
- `plugmem.enabled=false` 或缺失 `plugmem` 配置时，新 agent 行为应接近原 `ContextEfficientAgentV2`。
- `reset()` recall 失败不会抛出异常。
- `make_prompt()` 在有 `plugmem_context` 时包含 `Past task hints:`。
- `remember_current_task()` 能从 HiAgent `self.memory` 构造新格式 payload。

### 建议测试命令

优先做轻量导入和构造检查：

```powershell
python - <<'PY'
import sys
sys.path.insert(0, "agentboard")
from agents import load_agent
from common.registry import registry
assert registry.get_agent_class("PlugMemContextEfficientAgent") is not None
print("PlugMemContextEfficientAgent registered")
PY
```

如果 PowerShell 不支持 heredoc，可改用等价的 `python -c`。

### 建议 commit message

```text
feat(hiagent): add plugmem-aware context agent
```

## HiAgent 项目：阶段 3 PDDL 任务上传与配置

执行位置：HiAgent 仓库根目录。

### 阶段目标

把阶段 2 的 `remember_current_task()` 接入 PDDL 任务结束路径，并补充可关闭的 PlugMem 配置。

默认配置必须是 `enabled=false`，避免无意中改变 baseline 行为。

### 允许修改的文件

- `agentboard/tasks/pddl.py`
- `eval_configs/hiagent/blocksworld.yaml`
- 如需要，也可同步修改：
  - `eval_configs/hiagent/barman.yaml`
  - `eval_configs/hiagent/gripper.yaml`
  - `eval_configs/hiagent/tyreworld.yaml`

### 禁止修改的内容

- 不修改 PlugMem 侧文件。
- 不修改 agent prompt 文本。
- 不改动 PDDL 环境逻辑。
- 不改变 `log_example()` 的字段结构。
- 不改变 success/progress/grounding accuracy 的计算方式。

### 核心实现要求

在 `EvalPddl.evaluate_env()` 的结束路径中调用：

```python
if hasattr(self.agent, "remember_current_task"):
    self.agent.remember_current_task(task_type=game_name)
```

PDDL 侧只负责弱耦合触发任务结束回调，不直接读取 `plugmem.upload_on_finish`。
`upload_on_finish` 的判断必须由 `remember_current_task()` 内部完成。

需要覆盖：

- 任务成功 `done` 后返回前
- 达到最大步数或失败返回前

调用位置要保证：

- 不影响 `self.agentboard.log_example(...)`
- 不影响原本 return 的 success/progress/steps/grounding accuracy
- upload 失败不能中断评测

配置中补充：

```yaml
plugmem:
  enabled: False
  base_url: http://localhost:8080
  api_key: dev-key-change-me
  graph_id: hiagent-cross-task
  recall_on_reset: True
  upload_on_finish: True
```

注意：默认 `enabled: False`。

### 验收标准

- 使用普通 agent 或 `plugmem.enabled=false` 时，PDDL 评测流程不依赖 PlugMem。
- 使用 `PlugMemContextEfficientAgent` 且 `plugmem.enabled=false` 时，不发起 recall，也不 upload。
- 使用 `PlugMemContextEfficientAgent` 且 `plugmem.enabled=true`、`upload_on_finish=false` 时，可以 recall，但任务结束不 upload。
- 使用 `PlugMemContextEfficientAgent` 且 `plugmem.enabled=true`、`upload_on_finish=true` 时，任务结束会尝试 upload。
- PDDL 侧不直接读取 `plugmem.upload_on_finish`。
- upload 失败不影响原评测返回。

### 建议测试命令

先跑禁用 PlugMem 的最小 smoke test：

```powershell
$env:EVALTASK="blocksworld"
python agentboard/eval_main.py `
  --cfg-path eval_configs/hiagent/blocksworld.yaml `
  --tasks pddl `
  --model qwen3_5_4b_vllm_server `
  --agent PlugMemContextEfficientAgent `
  --max_num_steps 2 `
  --memory_size 100 `
  --log_path ./logs/hiagent/smoke_plugmem_disabled
```

如本地 LLM 服务不可用，可只运行导入测试，并说明未完成端到端验证。

### 建议 commit message

```text
feat(hiagent): upload pddl trajectories to plugmem
```

## HiAgent 项目：阶段 4 Smoke test 与最小 ablation

执行位置：HiAgent 仓库根目录。

前置条件：启动的 PlugMem 服务必须来自阶段 1 修复后的 PlugMem 项目版本。

### 阶段目标

验证完整链路：

- PlugMem 未启动时可降级运行。
- PlugMem 启动后可以 recall 和 upload。
- prompt 中出现 `Past task hints`。
- PlugMem stats 节点数量增加。
- 建立最小 ablation 运行方案。

### 允许修改的文件

- 默认不修改代码文件。
- 如需要记录 smoke test 和 ablation 命令，可新增：
  - `docs/hiagent_plugmem_smoke_test_notes.md`

如果 smoke test 暴露出前面阶段的 bug，应回到对应阶段做最小修复，并在阶段总结里说明原因。

### 禁止修改的内容

- 不做新功能。
- 不修改 prompt 文本，除非 smoke test 证明它无法工作。
- 不修改 PlugMem API 语义。
- 不修改 `docs/hiagent_plugmem_integration_execution_plan.md`。
- 不扩大到非 PDDL 任务。
- 不做大规模实验脚本重构。

### 核心实现要求

执行以下 smoke test：

1. PlugMem 未启动时运行 HiAgent。
   - 预期 recall/upload 失败只记录 warning。
   - HiAgent 评测流程继续。

2. PlugMem 启动后创建 graph：

```text
hiagent-cross-task
```

3. 运行小规模 PDDL 任务。
   - 检查 prompt 日志里有 `Past task hints:`。
   - 检查任务结束后调用 `/memories`。
   - 检查 `/api/v1/graphs/{graph_id}/stats` 中节点数量增加。

最小 ablation：

- baseline：原 `ContextEfficientAgentV2`。
- agent 但禁用 PlugMem：`PlugMemContextEfficientAgent` + `plugmem.enabled=false`。
- recall only：`plugmem.enabled=true` + `recall_on_reset=true` + `upload_on_finish=false`。
- recall+upload：`plugmem.enabled=true` + `recall_on_reset=true` + `upload_on_finish=true`。

### 验收标准

- 降级场景不崩溃。
- PlugMem 启动后能完成 recall 和 upload。
- prompt 中能看到 `Past task hints:`。
- PlugMem graph stats 在 upload 后增加。
- 四种 ablation 的运行命令可复现并记录。

### 建议测试命令

PlugMem 健康检查：

```powershell
curl http://localhost:8080/api/v1/health
```

创建 graph：

```powershell
curl -X POST http://localhost:8080/api/v1/graphs `
  -H "X-API-Key: dev-key-change-me" `
  -H "Content-Type: application/json" `
  -d "{\"graph_id\":\"hiagent-cross-task\"}"
```

检查 stats：

```powershell
curl http://localhost:8080/api/v1/graphs/hiagent-cross-task/stats `
  -H "X-API-Key: dev-key-change-me"
```

HiAgent smoke test 命令根据本地可用模型调整。

### 建议 commit message

```text
test(hiagent): document plugmem smoke tests and ablations
```

## 给 Codex 的阶段执行模板

后续执行时，可以复制下面模板，并把 `<阶段编号>` 替换为目标阶段。

```text
请只执行 `docs/hiagent_plugmem_integration_execution_plan.md` 中的阶段 <阶段编号>。

执行要求：
- 只修改该阶段“允许修改的文件”。
- 不修改该阶段“禁止修改的内容”。
- 不做无关重构。
- 不格式化整个文件。
- 如果必须修改阶段外文件，先说明原因并等待确认。
- 完成后运行该阶段建议测试命令；如果无法运行，说明原因。
- 测试通过后给出建议 commit message，但不要自动 commit，除非我明确要求。

最终输出：
- 修改文件列表
- diff 摘要
- 实际运行的测试命令和结果
- 风险点
- 未完成项
```
