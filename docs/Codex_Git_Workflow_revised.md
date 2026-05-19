# Codex + Git 修改代码全流程（修正版）

适用于：本地 PyCharm + Codex 修改代码，推送到自己的 fork，服务器拉取指定分支测试，通过后向原仓库提交 PR。

核心原则：

- Codex 永远只在实验分支改代码。
- 没有原仓库写权限时，先 push 到自己的 fork，再向原仓库开 PR。
- 服务器只测试实验分支，不直接在 `main` 上测试未合并代码。
- 测试通过后再创建或更新 PR。
- 合并后如果出问题，用 `revert` 回退，不要重写已经 push 的 `main` 历史。

## 0. 总体流程

本地创建实验分支 -> Codex 修改代码 -> 本地提交 -> push 到 fork -> GitHub 创建 PR -> 服务器从 fork 拉分支 -> 服务器测试 -> PR 合并到原仓库 `main` -> 如有问题用 `revert` 回退。

## 1. 本地创建 Codex 实验分支

在本地项目目录执行：

```bash
git checkout main
git pull origin main
git checkout -b codex-test
```

确认当前分支：

```bash
git branch --show-current
```

输出应该是：

```text
codex-test
```

只有确认当前分支是实验分支后，才让 Codex 修改代码。不要让 Codex 直接在 `main` 分支上改代码。

## 2. 使用 Codex 修改代码

任务要小，例如：

- 只修复某个 API。
- 只添加某个测试。
- 只修改某个配置项。

不要给 Codex 太泛的任务，例如“帮我修一下整个项目”。

推荐给 Codex 的提示词：

```text
请只完成下面这个小任务：[写清楚任务]

限制：
- 只修改指定文件
- 不做无关重构
- 不删除已有逻辑
- 修改前先说明计划
- 修改后总结修改文件、diff 摘要、测试命令和结果
```

## 3. 本地检查、提交

Codex 改完后，先看 diff：

```bash
git diff
```

如果不满意，并且确认这些改动都不需要，可以撤销未提交修改：

```bash
git reset --hard HEAD
```

如果满意，提交：

```bash
git add .
git commit -m "codex: describe the change"
```

## 4. 推送到远程

### 情况 A：你有原仓库写权限

可以直接推到 `origin`：

```bash
git push origin codex-test
```

### 情况 B：你没有原仓库写权限

如果出现类似错误：

```text
remote: Permission to OWNER/REPO.git denied to USER.
fatal: unable to access 'https://github.com/OWNER/REPO.git/': The requested URL returned error: 403
```

说明当前 GitHub 账号没有向原仓库 push 的权限。此时应推送到自己的 fork。

先确认远端：

```bash
git remote -v
```

如果还没有 fork remote，添加它：

```bash
git remote add fork https://github.com/<your-user>/<repo>.git
```

如果 fork remote 已存在但地址不对，修改它：

```bash
git remote set-url fork https://github.com/<your-user>/<repo>.git
```

然后推送实验分支：

```bash
git push fork codex-test
```

推送成功时会看到类似：

```text
* [new branch] codex-test -> codex-test
```

如果出现：

```text
remote: Repository not found.
```

通常表示 fork 仓库不存在、仓库名写错，或当前登录账号无权访问该 fork。需要先在 GitHub 上 fork 原仓库，或修正 fork remote 地址。

## 5. 在 GitHub 创建 PR

从 fork 分支向原仓库开 PR。

PR 页面应类似：

```text
base repository: 原仓库
base: main
head repository: 你的 fork
compare: codex-test
```

如果页面显示：

```text
Able to merge
```

说明当前分支可以自动合并。

PR 标题建议使用清晰的 commit 风格：

```text
fix(module): describe the change
```

PR 描述建议写：

```markdown
## Summary
- 修改点 1
- 修改点 2
- 修改点 3

## Tests
- `pytest tests/test_xxx.py` passed
```

如果本地没有跑通测试，要明确说明原因，例如：

```markdown
## Tests
- Not run: local Python environment does not have pytest installed
```

## 6. 服务器拉取实验分支

在服务器上进入项目目录：

```bash
cd ~/项目目录
```

查看远端：

```bash
git remote -v
```

如果服务器只有原仓库 `origin`，但实验分支推到了 fork，需要添加 fork：

```bash
git remote add fork https://github.com/<your-user>/<repo>.git
git fetch fork
```

从 fork 创建本地测试分支：

```bash
git checkout -b codex-test fork/codex-test
```

如果本地分支已经存在，不要再用 `-b`，直接切换：

```bash
git checkout codex-test
```

如果需要把本地分支绑定到 fork 分支：

```bash
git branch --set-upstream-to=fork/codex-test codex-test
git pull
```

确认当前分支：

```bash
git branch --show-current
```

输出应该是：

```text
codex-test
```

## 7. 处理服务器 checkout 阻塞

如果切换分支时报错：

```text
error: The following untracked working tree files would be overwritten by checkout:
        some_file.py
Please move or remove them before you switch branches.
Aborting
```

说明服务器当前工作区有未跟踪文件，目标分支也有同名文件。Git 为了避免覆盖本地文件而拒绝切换。

先查看：

```bash
git status --short
```

保守做法是先备份未跟踪文件：

```bash
mkdir -p ../backup-untracked-before-checkout
mv some_file.py ../backup-untracked-before-checkout/
```

如果有未跟踪目录，也可以一起移动：

```bash
mv src/some_untracked_dir ../backup-untracked-before-checkout/
```

然后重新切换分支：

```bash
git checkout codex-test
```

只有在确认这些未跟踪文件完全不需要时，才删除它们：

```bash
rm some_file.py
```

## 8. 服务器测试

先确认 Python 环境：

```bash
python --version
which python
```

如果项目使用虚拟环境或 conda 环境，先激活对应环境。

如果测试依赖还没安装：

```bash
python -m pip install -e ".[dev]"
```

运行指定测试：

```bash
pytest tests/test_api_memories.py
```

如果 `pytest` 命令找不到，用：

```bash
python -m pytest tests/test_api_memories.py
```

测试成功的标志类似：

```text
10 passed
```

warnings 不等于失败。只要最终结果是 `passed`，该测试命令就是通过的。

也可以先做语法检查：

```bash
python -m py_compile path/to/file1.py path/to/file2.py
```

或检查整个项目：

```bash
python -m compileall .
```

但语法检查不能替代 pytest。它只能证明文件语法能被 Python 解析，不能证明 API 行为正确。

## 9. 测试失败时怎么办

不要合并 PR。

服务器可以先保留测试日志：

```bash
pytest tests/test_api_memories.py 2>&1 | tee test.log
```

本地继续在同一个实验分支修改：

```bash
git checkout codex-test
# 继续让 Codex 修改
git add .
git commit -m "codex: fix previous issue"
git push fork codex-test
```

服务器拉最新分支继续测试：

```bash
git checkout codex-test
git pull
pytest tests/test_api_memories.py
```

## 10. 测试成功后更新 PR

把服务器测试结果写到 PR 描述或评论里，例如：

```markdown
## Tests
- `pytest tests/test_api_memories.py` passed: 10 passed, 47 warnings
```

如果 PR 已经打开，后续 push 到同一个 fork 分支会自动更新 PR。

## 11. PR 合并后同步 main

PR 被合并后，本地和服务器都可以同步原仓库 `main`：

```bash
git checkout main
git pull origin main
```

如果还需要清理实验分支：

```bash
git branch -d codex-test
```

远端 fork 分支也可以在 GitHub 页面删除，或执行：

```bash
git push fork --delete codex-test
```

## 12. 合并后发现问题，如何回退

如果 PR 已经合并到原仓库 `main`，用 `revert`，不要对已经 push 的 `main` 使用 `reset --hard`。

```bash
git checkout main
git pull origin main
git log --oneline
```

找到需要回退的 merge commit 或 squash commit。

如果是 merge commit：

```bash
git revert -m 1 <merge_commit_id>
git push origin main
```

如果是普通 commit 或 squash commit：

```bash
git revert <commit_id>
git push origin main
```

`revert` 会生成一个反向提交，历史记录清楚，也更安全。

## 13. 常用命令速查

创建实验分支：

```bash
git checkout main
git pull origin main
git checkout -b codex-test
```

提交修改：

```bash
git diff
git add .
git commit -m "codex: describe the change"
```

推到 fork：

```bash
git remote add fork https://github.com/<your-user>/<repo>.git
git push fork codex-test
```

服务器从 fork 拉分支：

```bash
git remote add fork https://github.com/<your-user>/<repo>.git
git fetch fork
git checkout -b codex-test fork/codex-test
```

本地分支已存在时切换：

```bash
git checkout codex-test
```

测试：

```bash
pytest tests/test_api_memories.py
python -m pytest tests/test_api_memories.py
```

处理未跟踪文件阻塞 checkout：

```bash
git status --short
mkdir -p ../backup-untracked-before-checkout
mv <file-or-dir> ../backup-untracked-before-checkout/
git checkout codex-test
```

同步 main：

```bash
git checkout main
git pull origin main
```

回退已合并的 PR：

```bash
git checkout main
git pull origin main
git log --oneline
git revert -m 1 <merge_commit_id>
git push origin main
```

最终记忆版：

```text
本地 Codex 分支改；
没有原仓库权限就 push 到 fork；
GitHub 从 fork 向原仓库开 PR；
服务器 fetch fork 并 checkout 实验分支测试；
测试成功再合并；
合并后出问题用 revert。
```
