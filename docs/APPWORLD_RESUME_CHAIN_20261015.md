# AppWorld 7B：给 163354 追加三个续训作业

2026-10-15。用户已提交的首个正式训练作业是 **163354**。本次仅追加三个串行作业，不取消、不重复提交首个作业。

## 配置与代码

- 每个续训作业：`testqueue`、账户 `prjsnnn0000269`、单节点、4 GPU、32 CPU、320G RAM、17:50:00。
- 模型继续使用已有 Qwen2.5-7B-Instruct，固定 revision `a09a35458c702b33eeacc393d103063234e8bc28`。
- 正式训练保持 150 epochs、16×8 分组、每 10 次完整策略更新保存一次、每个 run 保留最近两个完整检查点。
- 训练本体继续使用 `c1fb4827301393c2f24f56a604a3fcb741e48f2f`，目录固定为 `/scratch/prjsnnn0000269/Students/shuming/cjk/code/gigpo-reproduction/appworld-7b-4gpu-train-c1fb482`。
- 新 checkout 只运行 `resume_chain.py` / `resume_train.sbatch` 这些提交与恢复控制代码。启动模型之前会切回上述固定训练 checkout，并核对版本、已跟踪文件是否被修改。
- 不要在新控制脚本 checkout 里直接运行 `train.sbatch`；新控制脚本没有更新旧训练入口的版本验收白名单。调度器会自动进入固定的已核验训练 checkout。
- 沿用现有资源/协议/四卡检查点验收目录 `appworld-7b-4gpu-preflight-20261015T003620Z-3834802`，每次正式训练仍执行原有节点资源校准。不重新下载模型或修改环境依赖。

## 依赖与检查点选择

链为 `163354 → resume 1 → resume 2 → resume 3`。每条依赖用 `afterany`：前一作业终止后解除依赖，包括达到时限或失败的情况；解除依赖后仍需等待资源。参考 [Slurm sbatch dependency 文档](https://slurm.schedmd.com/sbatch.html#OPT_dependency)。

每个续训作业真正获得资源时才选检查点，提交时不固定 step。只扫描本链已记录的先前作业和当前作业的输出目录，不扫描其他实验、模型或验证作业。只使用已发布的 `global_step_N`，执行原生完整性校验：

1. 顶层 `CHECKPOINT_COMPLETE.json`，模型、优化器和额外状态的四个 rank 分片及文件大小；
2. 数据加载器、driver RNG/epoch、环境采样状态均存在；
3. 原始 run 的训练 commit、模型、任务环境、seed、完整训练入口和 schedule 匹配；
4. 根据最高的全局更新编号选取最新完整检查点，跳过残缺分片和 `.incomplete-*`，不依赖可能过时的 `latest_checkpointed_iteration.txt`。

恢复通过原生 `trainer.resume_mode=resume_path`、`trainer.resume_from_path=<选中的完整目录>`，继续全局步数与原有训练目标，不为每个作业重新追加 150 epochs。原训练器恢复模型、optimizer/scheduler、各 rank RNG、driver RNG、数据加载器和环境 sampler；续训不重复初始验证。

没有完整检查点时明确失败，绝不静默从初始模型重新开始。若某次作业失败而未写出新检查点，后面的作业仍可使用更早作业的最新完整检查点。达到时限或中途失败后，最后一个完整检查点以后的未保存更新需要重跑。

若最新完整检查点已经达到训练日志中记录的原生总更新数，后续作业启动后直接退出，不再启动模型或环境。三个续训作业是额外预算，不保证四个作业一定完成整个 schedule。

## SwanLab

采用用户允许的备选方式：**每个续训作业创建独立 run**，保留真实的绝对全局步数。

SwanLab 本身支持 `id` / `resume`（[官方说明](https://docs.swanlab.cn/guide_cloud/experiment_track/resume-experiment.html)）。但本任务每十次更新才保存检查点，而指标每次更新都会上传：例如旧 run 上传到 17，检查点在 10，恢复后会重跑 11–17。核对过的 SDK 0.7.0 源码中，`data/porter/utils.py::filter_metric` 会过滤不大于云端已有最新 step 的同名指标；`data/run/key.py::mock_from_remote` 也保护历史步数。不能可靠地把被丢弃的旧训练尾部当成恢复后轨迹，也不通过人为平移 step 掩盖回退。本次不升级 SDK、不修改既有 run、不尝试覆盖云端历史。

- 清除继承的 `SWANLAB_RUN_ID`、自定义运行目录和并行共享设置，明确 `SWANLAB_RESUME=never`。
- 三个续训 run 的 group 为 `appworld-chain-163354`，tags 记录 chain、resume 序号和起始检查点 step。
- run 名称中包含当前 Slurm job ID、`chain163354-rN-fromK`；描述包含来源完整检查点路径。
- 原来的首个 run 保持原样；三个续训 run 的描述与名称关联到它。没有声称已把原 run 也改入 group。
- 如果从 `global_step_10` 恢复，第一个完成的新策略更新仍记为 step 11，而非 1。SwanLab 配置中的 `trainer.resume_from_path` 也保留来源。
- SwanLab 上传失败时继续使用原有本地诊断保留机制；实际云端可见性以作业日志中的 tracker 状态为准。

跨 run 分析时，前一 run 在被选恢复点之后的旧尾部与续训数据属于不同尝试，不能直接平均或重复计数。

## 提交及记录

在新控制脚本 checkout 内运行现有独立 Python 3.10（不需要激活训练环境）：

```bash
/scratch/prjsnnn0000269/Students/shuming/cjk/runtimes/bin/python3.10 experiments/gigpo_diag/resume_chain.py submit --parent-job 163354
```

提交器先核对父作业的用户、目录、资源和完整训练入口，再申请三个 `afterany` 作业。邮件设置继续使用 BEGIN,END,FAIL,TIME_LIMIT_90。

记录保存在 `/scratch/prjsnnn0000269/Students/shuming/cjk/logs/gigpo-reproduction/appworld/resume-chains/from-163354/chain.json`。同一版本、同一路径重跑命令会复用已记下的作业号，不重复提交。文件锁防止同时提交；提交前持久保存意图、返回后记录作业号。如果中途丢失了 `sbatch` 响应而无法确认是否提交成功，停止自动重试，保留 `submitting` 状态供人工核对，不自行创建重复作业。不要删除这个记录以强制重试。

每个作业启动时在同目录的 `starts/` 写入选择结果、检查点 step、原生目标步数、训练/控制脚本版本和 SwanLab 策略。Slurm 日志为 `/scratch/prjsnnn0000269/Students/shuming/cjk/logs/ggp-appworld-7b-resume-<jobid>.out`，训练详细输出继续保存在每个新 run 的目录下。

使用 `afterany` 时，单独取消上游作业也会解除下游依赖。若要停止整条链，应使用 `chain.json` 中的准确作业号一起取消；提交脚本不会自动取消任何作业。

## 本地验证范围

以真实检查点事务代码生成的小型文件夹和模拟调度器，验证最新检查点选择、残缺分片回退、世界大小、run 身份和 schedule 校验、无检查点拒绝、达到目标退出、串行依赖、重复提交保护、模糊提交结果处理，以及独立 SwanLab run 的配置。另运行已有检查点和完整训练入口测试。均不启动 HPC、GPU 模型或云端实验；完整 GPU 恢复继续由已有四卡检查点验收覆盖，本次 HPC 续训尚待实际运行确认。
