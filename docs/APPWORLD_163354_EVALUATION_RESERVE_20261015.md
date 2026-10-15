# AppWorld 163354：评估内存故障与修复边界

## 已确认的现场事实

运行 `appworld-qwen25-7b-gigpo-163354-20261015T131851Z`，提交
`c1fb4827301393c2f24f56a604a3fcb741e48f2f`，7B / 4 GPU / 320G。
完成 3 次 policy 更新，在第 4 批 rollout 的 worker 38（PID 269952、generation 0）失败。
任务为 `6ea6792_1`，experiment 后缀 `w38-r4`。

服务端 `/evaluate` HTTP 500 的底层异常：

```text
AppWorld.evaluate -> evaluate_task -> task.model_collection
 -> ModelCollection.load -> SQLModel.set_db -> get_db_args
 -> from_db_connection.backup(to_db_connection)
MemoryError
```

worker 38 的 limit 文件记录 `RLIMIT_AS=(1092116480,1092116480)`。
最后连续采样的 VmSize 为 `1091608576` 字节，只剩 **507904 字节（496 KiB）**；
RSS 为 `756531200` 字节（约 721.5 MiB）。
这强烈支持单服务地址空间限额挡住评估时的数据库复制，不是整份 320G 作业 OOM 的证明。
这些读数也不能倒推出失败的那次 SQLite 分配精确请求了多少字节。
不能仅凭这些数据判断占用是模型保留对象、数据库缓存、分配器保留，或它们的组合。

原校准只覆盖两条 train、两条 dev 的初始化、执行、异常与日志保存，
没有调用原生 `world.evaluate()`。原生 evaluator 会另行加载任务起始数据库。
把生成代码与评估共用一个已接近耗尽的硬上限，缺少框架收尾余量。
此前验证通过不能覆盖这条未测路径，这是原校准设计的遗漏。

`COMPLETE_CHECKPOINTS=[]`。保持每 10 次更新保存时，这 3 次更新无法续训。
旧 `resume_chain.py` 明确绑定 c1fb482 和原训练目录；它不会自动切换到本补丁。
不能用旧链命令给新训练续训，亦不能把新训练偷偷接到没有检查点的旧作业。

## 本次修复

1. 单服务保留固定硬上限，模型 cell 使用较低的固定软上限。
   新 wrapper 只包裹 `_shell_run_cell`，无论正常返回、异常或 deadline，
   都在 `finally` 中恢复框架额度，然后让原生 execute 保存、记录，以及原生 evaluate 评估。
   不清空用户命名空间，不重放动作，不重写奖励，不把评估异常当任务失败。
2. 预算仍为 144 个服务合计最多 192 GiB（320G 的 60%）。每服务硬上限不超过
   `floor(192 GiB / 144 / 4096)*4096`，约 1.333 GiB。
   cell 额度保持从实测评估前 VmPeak 的 1.30 倍按页向上取整；剩余额度留给框架。
   要求保留额度至少覆盖实测评估额外 VmPeak 的 1.30 倍再加 64 MiB。
   超预算就停止，不能自动增加作业 RAM 或放宽到无限制。
3. 五个样本全部执行原生评估：原先四条样本，加上这次失败的 `6ea6792_1`。
   在选定限额下再次测正常流程，然后用真实 cell 在 `user_ns` 保留大对象直到 MemoryError，
   验证日志保存、原生评估、对象未被删除，以及原生 close 后同进程下一 episode 的执行与评估。
   这只增加校准样本，不改变训练/dev 数据集或任务分布。
4. 原来的无限分配、捕获异常、吞 deadline、HTTP 分组失效与 checkpoint 验证继续保留。
   新 profile 必须包含评估和保留对象压力测试的通过证据，旧 profile 不可直接使用。
5. 新增 `GIGPO_PREFLIGHT_AND_TRAIN=1`：同一个完整训练作业先做本版本的有限校准、
   HTTP 协议和四卡保存恢复检查，通过后直接进入原来的完整训练。任一步失败不启动训练。
   17:50 时限包含这些检查。旧版本验收不被本补丁冒用，亦不需要另排一个训练作业。

Linux 的 AS 限制约束虚拟地址空间；内核执行 soft limit，而 hard limit 是它允许的上界。
非特权进程可在 hard limit 内恢复 soft limit，不能提高既定 hard limit。
参考：[Linux getrlimit 手册](https://man7.org/linux/man-pages/man2/getrlimit.2.html)、
[Python resource 文档](https://docs.python.org/3/library/resource.html)。
这是一项资源约束，不宣称是对恶意 Python 的安全沙箱。

## SwanLab 与未改变的配置

用户要求不修 SwanLab，故 tracker、依赖和上传行为均未修改。
该作业使用 0.7.0，创建项目接口返回 422，已回退到本地记录。
此前 CONFIG_DIFF.md 已记录新版本项目需要 >=0.9 SDK，以及升级训练环境的 protobuf 冲突。
可在结束后用现有独立新版解释器及 `backfill_swanlab.py` 补传本地 scalar；本次未上传。

模型保持已下载的 Qwen2.5-7B-Instruct 固定 revision，4 GPU、32 CPU、320G、17:50；
GiGPO、16×8 train / 16 dev、150 epochs、每 10 次更新保存、保留两份完整检查点均不改。
没有修改或取消 HPC 上的作业、现有 checkout、共享环境或历史日志。

## 本地验证与限制

- Linux 真实 RLIMIT_AS 子进程：保留对象耗尽 cell 额度后，12 MiB SQLite 数据库复制成功；
  对象仍在，随后 cell 可主动释放对象并继续执行。外部 20 秒上限，不使用 GPU/任务数据。
- 原生 AppWorld/IPython 集成：测试用限制替身核实 cell 内额度、错误文本、原生保存前恢复、
  一次动作一次计数、持久用户状态。没有把限制替身称为内核验证。
- 校准、正常/异常限制恢复、未知限制变动拒绝、固定总预算、原有 timeout/异常清理、
  检查点与续训边界，以及同作业前置检查失败不训练的回归均在本地测试。
  最终结果为 Linux 47 项、Windows 原生 AppWorld/IPython 18 项通过，Bash 语法检查通过。
- **尚未在 HPC 的真实任务数据库上运行本修复，也未提交新作业。**
  必须以新作业的原生评估/保留对象压力记录为准；有限样本不保证所有任务与长时间训练稳定。
  未实现新的 cgroup 支持；旧现场的 cgroup 读数 unavailable 仍需与服务 AS 上限区分。

## 部署

推送到原分支 `codex/appworld-qwen25-7b-train` 的新提交。使用新 detached worktree
固定该提交，避免修改旧训练或依赖作业使用的目录。它不是新建分支。
提交 `train.sbatch` 时显式设置：

```text
GIGPO_EXPECTED_COMMIT=<本补丁的完整提交 SHA>
GIGPO_RUN_ID=
GIGPO_PREFLIGHT_ONLY=0
GIGPO_PREFLIGHT_AND_TRAIN=1
GIGPO_RESUME_MODE=disable
GIGPO_RESUME_FROM=
GIGPO_ACCEPTANCE_DIR=
GIGPO_ACCEPTANCE_REUSE_FROM=
```

原作业无完整检查点，本次从固定初始 7B 权重开始。新训练的后续依赖链必须绑定新提交、
新首作业 ID 与其检查点，不能继续使用绑定旧 c1fb482 的 helper。
