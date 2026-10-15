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

日志根目录：`/scratch/prjsnnn0000269/Students/shuming/cjk/logs/gigpo-reproduction/appworld/`。入口输出真实 `PREFLIGHT_JOB`、`TRAIN_JOB` 与完整 run 路径。首次验收作业 162144 的初始化失败见第 6 节；后续作业 162401 的通过阶段与测试环境隔离问题见第 7 节。

## 5. 后续观察与恢复

- 先看验收失败在哪个阶段，不因失败就继续加 RAM。若合法基线在限额内不能初始化，保留 `resource_preflight/*/result.json` 和 stdout；下一步才根据真实测量评估物理并发方式。
- 正式 run 查看 `services/resource_usage.jsonl`、`terminal-PID.json`、`execute-PID.jsonl`、checkpoint 分阶段内存记录及 Slurm 状态。RSS 求和包含共享映射，不能当成精确 job 总内存。
- SwanLab 中 `bounded_runtime/train/*`、`bounded_runtime/eval/*` 与基础设施内存指标单独显示；原 AE/AS/anchor 等研究指标继续记录。native evaluator 不可知时留空，不能把被杀服务算成已评测失败。
- 未来跨作业恢复须指向新格式四卡完整 `global_step_N` 目录，设置 `GIGPO_RESUME_MODE=resume_path` 和 `GIGPO_RESUME_FROM=<该目录>`，在新的 run 保存后续 checkpoint。不要复用不完整 staging 目录。
- 320G 仍可能不足，尤其大量合法全局状态或 checkpoint/rollout 总峰值。当前改动建立可观测、可拒绝的边界，尚无 HPC 结果可证明完整训练成功或学习曲线改善。

代码通过用户授权的 GitHub 分支同步；本机没有连接 HPC 执行新任务。已收到的 HPC 证据记录如下，修复后的真实任务/GPU 验收与学习曲线仍待验证。

## 6. 作业 162144：测试目录名误触发 AppWorld 数据库路径校验

用户提供的 run 为 `appworld-7b-4gpu-preflight-20261014T124359Z-1383635`，代码为 `c8a9a233e2c55c753dd293e01cd5d569266bf4bd`。`resource_preflight/catch-memory/result.json` 显示 `returncode=1`、`external_kill=false`，约 2.72 秒退出；采样/事件峰值 RSS 143405056 字节、VMS 170663936 字节，AS 限额 1092112384 字节。事件停在 `world_begin` 后的 `failed`，未进入 `danger_started`；训练更新数与 episode 数均为 0。

实际异常是 `Invalid changes_file_path. It cannot contain in-memory connection strings.`。AppWorld 0.2.0.dev0 的 `apps/lib/models/db.py:346` 对 `changes_file_path` 使用 `"memory" in changes_file_path`，并非仅判断内存连接前缀。我们的磁盘目录 `catch-memory` 因而导致原生任务数据库加载失败。`get_db_args` 也使用同类字符串判断，所以不应只绕过该异常检查。

这次失败发生在危险代码执行前，现有日志不支持将其归因为再次 OOM、HTTP 超时或 7B 能力问题。依照顺序执行的控制流，前面的四个基线、四个限额健康检查及 `allocate` 已返回通过；尚未读取这些检查的全部明细，也不能据此声称后续全部验收成功。

修复仅将该模式的磁盘子目录改为 `catch-allocation`，CLI/event 中仍保留 `catch-memory` 模式和原有危险代码、限额、超时及通过条件。不修改 AppWorld 安装包，不跳过测试，不调整 7B/4 卡/320G、GiGPO 参数或每 10 轮保存配置。

新增 `test_appworld_resource_preflight`：调用 preflight 主入口生成的全部 11 个检查目录，使用本机实际安装的 AppWorld 数据库加载函数和小型 SQLite/JSONL fixture 验证路径。旧代码在相同 `apply_db_changes` 检查处复现了同一异常；改名后全部路径成功加载，并确认八个健康检查和三个危险模式均保留。该测试替代了任务执行与资源测量，不能代替 HPC 原生任务、危险循环或四卡训练验收。它已加入现有验收作业的本地回归测试列表。

## 7. 作业 162401：资源预检通过，协议单测继承了作业端口路径（2026-10-15 复核）

用户提供的 run 为 `appworld-7b-4gpu-preflight-20261014T154946Z-2065874`，代码为 `bbef2333dabfcb6bcb3ace9aeb2203ad637a7039`。日志已打印 `BOUNDED_RUNTIME_PREFLIGHT_PASSED`，表明两 train/两 dev 任务的基线与限额健康检查，以及 `allocate`、`catch-memory`、`swallow-deadline` 三个危险检查均通过。选出的单服务 AS 限额为 1092112384 字节，RSS watchdog 为 982901145 字节；这是抽样校准，不代表 144 服务并发或完整训练已通过。随后第一组 31 项回归测试为 `OK`。

作业停在第二组 CPU/fixture 回归测试：15 项测试中报告 13 个 error（包含同一测试对八个 slot 的子测试），均为协议测试读取不存在的正式 `appworld_ports.ports`。`train.sbatch` 已导出 `APPWORLD_PORTS_FILE`；生产 `load_available_ports()` 按设计优先使用该环境变量。协议测试虽给 `AppWorldEnvs` 传入自己的临时端口文件，却漏掉了对这个环境变量的隔离。此时正式服务尚未启动，正式端口文件不存在是正常的阶段状态。

修复在协议测试的 `setUp` 中将 `APPWORLD_PORTS_FILE` 指向该测试独有的临时 `ports`，并通过已有 cleanup 恢复父环境。保持生产环境变量优先级，不制造正式端口文件、不提前启动服务、不跳过检查，也不修改模型、RAM、GPU、训练超参数或检查点频率。

本地验证使用外层注入的作业环境变量，包括指向不存在文件的 `APPWORLD_PORTS_FILE`：旧测试复现同一 `FileNotFoundError`；修复后协议与 CPU checkpoint 两个测试模块共 15 项全部通过，并核对父环境恢复、正式端口文件未被创建。Windows 本地验证将 TEMP/TMP 指向工作区内的可写临时目录。CPU checkpoint 使用实际小型 Torch/Adam 模型，FSDP 上下文为 fixture，不是四卡 7B 训练。

本次日志中 `updates=0`、`episodes=0`。八服务真实 HTTP 协议验收与四卡 7B 检查点验收尚未执行。资源 profile 的整作业 cgroup v2 读取显示 `unavailable`，checkpoint 诊断另有 cgroup v1 记录；不能将前者解释为作业内存为零，也没有本次再次 OOM 的证据。

## 8. 作业 163002：首次 actor 更新编译异常；新增单次真实更新验证

用户提供的正式作业代码为 `58825a74e7de378070970581ae4d8d7b344c5235`，运行于 `8FS9B74`，申请 320G，31 分 12 秒后状态为 `FAILED`。Slurm 的 MaxRSS 为 0，不能据此判断峰值；本次没有 `OUT_OF_MEMORY` 结论。主日志确认已有资源/HTTP/四卡检查点验收通过，正式节点资源校准也通过，随后完成初始评估和第一批 rollout，累计 144 episodes，完成更新数为 0。

失败发生在 `actor_rollout_wg.update_actor(batch)`。Ray 还原异常时出现 `BackendCompilerFailed.__init__() missing ... inner_exception`，进而报 `Failed to unpickle serialized exception`。这证明存在先前的 PyTorch 编译后端异常，但原始底层错误未出现在主日志中。不能将缓存竞争、GPU/CPU OOM、四卡并行或某个编译器错误当成已确认根因。旧作业 161511 使用 `9d7b424`、7B/2 卡/320G，完成 4 次更新后 OOM；两版 actor/FSDP worker 更新代码、编译开关和缓存设置原本相同。已有检查点验收未覆盖实际 PPO 编译熵路径，这是验收范围缺口。

本次按用户要求增加独立入口 `experiments/gigpo_diag/verify_actor.sbatch`，只提交一个 7B/4 卡/320G/32 CPU、最长 75 分钟的诊断作业，不指定计算节点，不接续完整训练，不下载模型或修改依赖，不重新运行四卡检查点往返。它仍执行同一资源校准和服务保护，再走正式初始评估、16×8 真实 rollout、GiGPO 优势计算及一次原生 actor 更新。保留编译、熵系数、微批大小和缓存设置；不以 eager 回退来制造通过结果。诊断作业将 `total_training_steps=1`，关闭本次结束时的额外评估/检查点和云端 tracker。正式入口的 150 epochs、每 10 次更新保存及 SwanLab 配置未变。单次更新成功也不等于完整训练或后续检查点稳定。

`GIGPO_ACTOR_VERIFY=1` 使原生 `update_actor` 外层记录 rank/PID、输入形状、编译配置及返回的梯度范数；异常与显式 `inner_exception` 堆栈在 Ray 传输前直接写入共享 run 的 `actor-update/*.traceback.txt`，随后重新抛出原异常，不重试或改变计算结果。退出时尽力归档本作业的节点本地 Ray 日志到共享 run 的 `ray-logs.tar.gz`（最多 60 秒，归档退出码单独打印，不掩盖训练退出码）。进程被 SIGKILL 时退出归档不保证执行；已同步写入的 worker 堆栈独立保留。

通过条件为原生训练进程正常结束，四个独立 rank/PID 都返回一次 update 1、梯度范数均有限、每个 worker 有真实 optimizer step 记录，且 driver 恰好确认一次 policy update。只有全部满足才生成 `ACTOR_UPDATE_VERIFIED.json`。此文件是单次更新验证证据，不发布或替代原 `acceptance.json`。原始错误、编译包版本、节点 GPU/驱动及路径设置均保存在本次共享 run，无须再为取日志申请同一节点。

本地已通过 11 项检查：7 项标准库证据测试，包括模拟无法 pickle 还原的异常、嵌套原始异常保存、记录写入失败时仍保留原异常，以及拒绝缺 rank/优化器记录/有限梯度等情况；4 项 Linux shell fixture 检查覆盖成功、资源预检失败、训练管道失败和最终证据不全，确认 `tee` 与退出归档不隐藏失败状态。它们没有在本机运行 7B/GPU；HPC 单次真实更新结果待提交后确认。

## 9. 两卡验证调整（2026-10-15）

用户认为四卡等待过长，本次将单次更新验证入口改为 **7B/2 卡/320G/32 CPU**，最长 75 分钟。Slurm 申请、训练实际 GPU 数和 run 名同时改为两卡；真实 16×8 分组、144 服务、编译设置、初始评估及一次更新保持原样。正式四卡训练入口及每十次更新保存配置不变。

验证器按显式期望卡数核对所有 worker，结果写入 `world_size`，两卡结果不能通过四卡检查。两卡仍报同类编译错误，有助于取得原始异常；两卡通过只证明该次两卡配置运行成功，不能排除四卡并行、节点或缓存状态对旧故障的影响。新增两项证据测试并扩展 shell 检查验证实际传入两卡配置，相关 13 项本地测试通过；HPC 结果仍待本次作业确认。

## 10. 作业 163052：真实优化器更新进行中，验证总时限到期

本次两卡验证使用 `d7a3ef30e27fedaf2e7d206625089a5c8abbacc7`，run 为 `appworld-7b-2gpu-actor-verify-163052-20261015T022730Z`，节点为 `8GS9B74`。资源保护预检通过，完成初始评估和首批 rollout，共记录 144 episodes。训练子进程在原先的 55 分钟限时处收到 SIGTERM，作业入口返回 124，Ray 日志归档成功。最终未生成单轮验证通过标记，driver 的完整 policy update 数仍为 0。

两个 worker 的 update 1 于北京时间 2026-10-15 11:08:13 左右开始，约 11:24:15 在 `loss.backward()` 中经 Ray 的 SIGTERM handler 抛出 `SystemExit: 1`，实际 actor 更新运行约 16 分钟。两份 optimizer 记录各有 51 行，最后的 `optimizer_step_local=51`，且 `parameter_sample_changed=true`。这些记录在原生 optimizer.step 返回后写出，证明已完成多次真实优化器步骤；两卡对应同一分布式训练，不能相加成 102 步。本次记录未复现此前的 BackendCompilerFailed，也不能由此认定四卡或长程训练已稳定。

本批每卡输入为 2537 行，`ppo_epochs=1`，实际配置的 actor rollout n 为 1（环境组大小仍为 8），mini-batch 64 经两卡归一化为 32。因此每卡一轮遍历 80 个 mini-batch，已有 51 次 optimizer.step，占计划步数约 64%。以观测平均速度线性估算，完整 actor 更新约 25 分钟，连同此前约 39 分钟的初始化、评估、采集等阶段，约需 64 分钟；这只是估计，不保证剩余 mini-batch 等速。

按用户同意的调整，将两卡验证 Slurm 时限从 75 分钟改为 120 分钟，内部训练阶段从 55 分钟改为 95 分钟。资源预检仍最多 15 分钟，余量留给环境准备、终止宽限和退出证据。模型仍为现有 7B、2 GPU、320G、32 CPU，保留初始评估、真实 16×8 rollout、编译和一次完整更新的通过条件。该验证仍不保存训练检查点、不接续正式训练；独立新 run 从原模型重新开始。此前已取消的四卡完整训练改动未纳入这次提交。

调整后，已有的 4 项 Linux 入口控制流测试与 9 项 worker 证据测试全部通过。它们使用本地 fixture 检查成功条件和失败传播，不替代新的 HPC 单轮运行结果。
