# AppWorld GiGPO 修复实施与验收交接

日期：2026-10-14。状态：修复代码与本地验证已完成；HPC 原生任务验收、四卡 7B 恢复验收和正式训练尚未运行。不能将下面的本地测试理解为已解决 HPC 的全部 OOM。

## 1. 本次最终配置

用户最后更正为 **Qwen2.5-7B-Instruct**，覆盖修复方案中原先的 1.5B。复用已有目录：

```text
/scratch/prjsnnn0000269/Students/shuming/cjk/data/models/Qwen2.5-7B-Instruct-a09a35458c70
```

模型和 tokenizer 固定 revision：`a09a35458c702b33eeacc393d103063234e8bc28`。复用此前 7B 分支的官方 Git blob/LFS 校验清单，仅读取已有文件，不下载或改写模型目录。

| 项目 | 本次设置 |
|---|---|
| 硬件 | 单节点、4 GPU、32 CPU、`--mem=320G`，没有继续增加 RAM |
| 算法 | 标准 GiGPO；gamma .95、step advantage weight 1、mean_norm、exact anchors |
| 采样 | train batch 16 × group 8；val batch 16；共 144 个服务 |
| 奖励/优化 | 原生成功奖励与 invalid penalty .1；KL loss .01；LR 1e-6；PPO minibatch 64 |
| 7B 显存设置 | 沿用之前 7B：actor/ref/rollout logprob microbatch 每卡 1，vLLM utilization .45，TP=1 |
| 日程 | 150 epochs；正式作业 17:50:00；不是 150 次更新 |
| 保存 | 每 10 次更新、以及原 trainer 的训练结束边界；保留本 run 最近两份完整 checkpoint |
| 日志 | SwanLab 和既有本地降级日志；诊断与真实轨迹保留 |

按用户最后要求，修复直接延续已有 `codex/appworld-qwen25-7b-train` 分支，从其 `9d7b424f2e28ab1a10de84f723a984e1de9a341f` 追加普通新提交，保留原 7B 的模型身份、离线校验、历史说明和日志路径。没有新建远程分支或强推。HPC 通过 `git worktree add --detach` 在固定 commit 上建立新工作目录；该命令不新建分支，也不修改原作业工作目录。

主要参考已核对为 `examples/gigpo_trainer/run_appworld.sh`，固定上游 commit `cd71379130155b2d75c9100753d1454d3cde6696`；上游默认 `save_freq=-1` 不构成完整保存恢复的验证。本次继续保存完整训练状态。

## 2. 针对历史问题实施了什么

最强的历史 OOM 证据仍是 161511：worker 2、任务 `ccb4494_3`、serial 248 持续生成非空模拟歌曲页并向列表追加新对象；进入 `shell_run_cell` 后没有结束记录，之后断连，Slurm 确认 320G 作业 OOM。前一请求 247 约 108 秒返回，可能留下异常引用。没有当时逐 PID/cgroup 曲线，不能把整个峰值精确归到请求 248。

### 2.1 原生错误完成之后清理框架异常引用

文件：`appworld_exception_cleanup.py`、`appworld_server.py`。

- 在原生 execute 已返回、错误文本/interaction/状态和日志保存已经完成之后，释放新产生的 `InteractiveTB.tb`、`last_execution_result`、SyntaxTB 及对应 `sys.last_*` 引用，再 GC。
- 保留 Agent 的变量、函数、容器和显式保存的异常/栈对象；没有 `user_ns.clear()`，没有 `traceback.clear_frames()`。
- 原生保存异常没有被吞掉；不会把未完成持久化伪装为一次健康执行。
- 清理只能处理框架引用。Agent 主动保存在全局变量中的大列表仍可能增长，必须由资源预算限制。

### 2.2 单服务资源约束与进程外监督

文件：`appworld_resource.py`、`appworld_service_launcher.py`、`appworld_resource_preflight.py`、`appworld_service_probe.py`、`services.py`、`appworld_execution_trace.py`。

- 使用现有独立服务进程。新解释器在导入 AppWorld 前设置硬 `RLIMIT_AS`；没有在多线程启动器里使用 `preexec_fn`，也没有限制 Ray/trainer 的地址空间。
- 在新 allocation 上用两个 train、两个 dev 任务的合法交互测量 RSS/VMS/VmPeak。选定 AS = 实测 VMS 峰值 × 1.30，按页取整，再用该限额复测。
- 环境总预算上界是本次 320G 的 60%，分摊给 144 个服务。**60% 是声明的预算策略，不是已测得 trainer 只需剩余 40%。**合法基线放不下时直接拒绝启动；不自动加 RAM、不减 group。
- RSS watchdog 为选定 AS 的 90%，要求高于合法基线 RSS 并留 15% 余量。`RLIMIT_AS` 是地址空间上限，不是 RSS 硬上限。
- 原生软 deadline 100 秒，外部 execute 最终 deadline 140 秒，HTTP 仍 160 秒。即使模型用裸 `except` 吞掉软异常，服务外监督器仍可终止该进程。
- 每 5 秒记录逐 PID RSS/VMS、request identity 和可读 cgroup v2 祖先链。若能识别有限额的 Slurm job cgroup，超过其 `memory.max` 的 90% 时中止整批并停止本 run 自有服务，按未知基础设施失败处理。不能识别时记录不可用；没有改集群 cgroup 或 OOMKillStep 设置。
- 只管理本进程启动的子服务。没有全局 kill、没有停止其他作业。

服务正常基线仅抽样四个任务，不能证明覆盖全部合法任务。实际限额、集群 cgroup 可用性及 7B+144 服务的总峰值，仍由 HPC 验收/正式运行提供证据。

### 2.3 明确 `bounded_runtime_v1` 故障语义

文件：`appworld_client.py`、AppWorld `envs.py`、collector `rollout_loop.py`、`diagnostics.py`。

| 情况 | 行为 |
|---|---|
| 原生语法/API 错误，或已健康返回的 MemoryError/100s 超时 | 保留原生反馈与状态，继续该 episode |
| 已证明该请求触发外部资源预算 | 本 episode 终止，reward 0；`resource_terminal=true`，`native_evaluator_success=null`，`budgeted_task_success=false` |
| 服务已资源终止后仍收到 inactive step | 不再 RPC、不再计数、不重复奖励；保持 inactive mask |
| 后续新 episode reset | 等待新 PID/generation，创建新 namespace/experiment；不把旧轨迹接起来 |
| HTTP 响应丢失但有身份完全一致的完成 ack | 返回已记录的原始反馈，不重新执行代码 |
| 其他断连、未知退出、状态/保存无法确认 | 整个 rollout batch 在优势计算前失败，不改成普通任务失败，不自动补样 |

request identity 包含 task、experiment、执行前 interaction 数和代码哈希。保留 raw action、实际执行代码、group/trajectory identity、active mask 和已有反馈；没有自动重试有副作用的行动。

这是显式资源终止协议，会改变资源超限轨迹的结束语义，必须与无限预算的原始环境区分。GiGPO 核心优势公式、正常奖励、任务分组和策略优化保持不变。

### 2.4 FSDP 保存/恢复与完整 checkpoint 事务

文件：`fsdp_checkpoint_manager.py`、`checkpoint_memory.py`、`checkpoint_transaction.py`、`ray_trainer.py`。

- model state → 保存/释放临时引用 → optimizer state → 保存/释放 → extra state；load 同样分阶段。保留各 rank collective 顺序。
- 每个阶段记录 RSS/VMS/cgroup、state tensor storage 所在设备与大小。GC 不等同于操作系统必然归还 RSS，因此不宣称峰值已降到某个数值。
- 完整 checkpoint 包含各 rank model/optimizer/scheduler/RNG、driver RNG、global step/epoch、StatefulDataLoader 状态与 train/val 环境采样器。
- 全部写到同文件系统的 staging 目录，确认各 rank 分片及元信息后原子发布，再更新 latest，最后清理同一 run 中较旧的完整 checkpoint，保留两份。保存失败不先删除上一份。
- 恢复后跳过重复初始验证、还原采样和 dataloader 游标，从下一更新继续。checkpoint 在更新边界，下一 rollout 重置 episode；不承诺恢复半个 Python episode。
- 目前支持同 world size 的四卡完整恢复；拒绝旧的不完整/两卡 checkpoint。文件清单检查缺失/截断，不是对任意同长度数据损坏的校验。
- 新 run 恢复时旧 run 的 checkpoint 不删除。未来随机 vLLM 生成不承诺逐位一致。

## 3. 已实际执行的本地验证

共 **72 项不同测试通过**，使用 Windows 本地 Python 和 WSL Linux；其中边界如下：

| 测试 | 数量 | 实际覆盖 |
|---|---:|---|
| `test_appworld_exception_cleanup` + `test_appworld_timeout` | 16 | 已安装的 AppWorld 0.2.0.dev0/IPython 9.17.1 原生 execute/shell/SafetyGuard；状态保存调用顺序与小对象引用；DB 持久化由 fixture 替代，Windows alarm 使用替身 |
| `test_appworld_failure_protocol` | 14 | CPU fixture、生产 worker/collector 方法；组内终止、active mask、unknown abort、身份与 sampler 恢复；不是实跑 Ray 集群 |
| `test_checkpoint_transaction` + `test_checkpoint_cpu_roundtrip` | 9 | 发布/失败/保留、driver 恢复；实际 CPU Adam 更新后保存/加载/再更新一致；FSDP/distributed context 为 fixture，非四卡验收 |
| `tests/test_multi_episode_rollout.py` | 4 | 既有 collector 回归 |
| `test_appworld_resource` | 6 | WSL Linux 真实受限子进程触发 MemoryError、进程外 deadline kill、健康 peer 保留、整作业压力归类；未加载 AppWorld task DB |
| `test_appworld_execution_trace` | 6 | 包含原生 SafetyGuard 中预开控制 FD 的写入和 ack；未重放历史失控任务 |
| `test_appworld_client` + `test_appworld_server` | 12 | 原生客户端/LibCST/FastAPI 与短时间 loopback HTTP 回归；不是 HPC 160 秒现场复现 |
| `test_verify_existing_model` | 5 | 小型合成文件验证只读校验、版本/分片/大小/同长度损坏；没有声称已在本机读取 HPC 的 7B 权重 |

Python AST 检查及 shell Bash 语法检查通过，`git diff --check` 通过。模型改为 7B 后重新运行了上述 5 项模型完整性校验测试。

主要复核命令（在相应已具备依赖的 Python 环境内）：

```bash
python -m unittest experiments.gigpo_diag.test_appworld_exception_cleanup experiments.gigpo_diag.test_appworld_timeout -v
python -m unittest experiments.gigpo_diag.test_appworld_failure_protocol experiments.gigpo_diag.test_checkpoint_transaction experiments.gigpo_diag.test_checkpoint_cpu_roundtrip -v
python -m unittest experiments.gigpo_diag.test_appworld_resource experiments.gigpo_diag.test_verify_existing_model -v
python -m unittest experiments.gigpo_diag.test_appworld_execution_trace experiments.gigpo_diag.test_appworld_client experiments.gigpo_diag.test_appworld_server -v
python -m pytest tests/test_multi_episode_rollout.py
```

## 4. HPC 执行入口与通过条件

用户已授权通过 GitHub 同步：拉取已有 `codex/appworld-qwen25-7b-train` 分支，在回复给出的固定 commit 上创建 detached worktree，再运行下列入口。无需上传或解压此前的 Git bundle，也不运行 prepare。不要在现有作业使用的目录直接 pull/切换版本。

新工作树中入口为：

```bash
bash experiments/gigpo_diag/submit_bounded.sh
```

该入口同时提交两份**新的** 4 卡/320G 作业，两者均保留发往 `1909674001@qq.com` 的 BEGIN/END/FAIL/TIME_LIMIT_90 Slurm 通知：

1. **验收作业，最长 1:15:00**：离线校验已有 7B；有硬 AS 与外部超时的原生四任务基线/危险测试；八个真实 HTTP 服务组成一组，验证一个资源终止、七个 peer 健康、sticky inactive、reset 新 PID、不重放和未知退出失败；最后执行真实四卡 7B Adam/FSDP 更新→保存→加载→下一次更新对比。
2. **完整训练作业，17:50:00**：`afterok` 依赖验收作业。验收不成功则不开始训练；开始时核对 commit/模型路径/依赖版本/卡数/RAM 相符，重新校准所在节点的服务限额，然后运行原完整 GiGPO 日程。

四卡测试读取本地 7B 全模型；仅短 token batch，不改变正式训练数据或超参数。它测试完整状态恢复，不是任务能力评测；测试时未同时启动 144 个环境，不能替代正式作业的总峰值监测。

通过标记必须实际存在且内容有效：

```text
<验收 run>/bounded_runtime_profile.json
<验收 run>/live-protocol/LIVE_PROTOCOL_PASSED.json
<验收 run>/checkpoint-roundtrip/CHECKPOINT_ROUNDTRIP_PASSED.json
<验收 run>/acceptance.json
```

日志根目录：`/scratch/prjsnnn0000269/Students/shuming/cjk/logs/gigpo-reproduction/appworld/`。入口输出真实 `PREFLIGHT_JOB`、`TRAIN_JOB` 与完整 run 路径；当前尚无这两份作业的 job ID。

## 5. 后续观察与恢复

- 先看验收失败在哪个阶段，不因失败就继续加 RAM。若合法基线在限额内不能初始化，保留 `resource_preflight/*/result.json` 和 stdout；下一步才根据真实测量评估物理并发方式。
- 正式 run 查看 `services/resource_usage.jsonl`、`terminal-PID.json`、`execute-PID.jsonl`、checkpoint 分阶段内存记录及 Slurm 状态。RSS 求和包含共享映射，不能当成精确 job 总内存。
- SwanLab 中 `bounded_runtime/train/*`、`bounded_runtime/eval/*` 与基础设施内存指标单独显示；原 AE/AS/anchor 等研究指标继续记录。native evaluator 不可知时留空，不能把被杀服务算成已评测失败。
- 未来跨作业恢复须指向新格式四卡完整 `global_step_N` 目录，设置 `GIGPO_RESUME_MODE=resume_path` 和 `GIGPO_RESUME_FROM=<该目录>`，在新的 run 保存后续 checkpoint。不要复用不完整 staging 目录。
- 320G 仍可能不足，尤其大量合法全局状态或 checkpoint/rollout 总峰值。当前改动建立可观测、可拒绝的边界，尚无 HPC 结果可证明完整训练成功或学习曲线改善。

代码通过用户授权的 GitHub 分支同步；没有连接 HPC 执行新任务，也没有产生新的学习曲线。实际 job ID 与运行状态应在用户执行一体化拉取/提交命令后补充。
