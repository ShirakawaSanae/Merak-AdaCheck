# Checkpoint Grouping 阶段进展汇报（09-09）

## 一句话概括

AdaCheck 解决“训练状态中哪些内容可以通过冗余消除而少传”，本工作进一步研究“哪些 rank 应当
互为 checkpoint peer，才能把可利用冗余转化为最少的跨故障域传输，同时保持每步可恢复”。
当前已经完成两机 16 卡的检查点原型、严格 one-checkpoint-per-step（1C1S）短期与 20-step
验证，以及 fresh-job 数值恢复；但在当前对称并行布局中，AdaCheck-style 固定分组已经等于
最优分组，因此尚未得到真实大模型端到端的 grouping gain。这是当前最重要、也必须诚实说明的
研究边界。

## 截至目前的工作量

截至目前，我们已完成 AdaCheck 论文与公开代码 breakdown、全局两两冗余代价和故障域约束的数学建模、exact matching 与 frozen catalog 工具链、异步增量检查点原型，以及覆盖静态重建、fresh-job 数值续训和故障注入的测试接口。
实验上已形成 LLaMA-like-1B/3B、BERT 与 GPT-2 的双机 16-rank 初步矩阵，并在真实 Merak 1B 上完成 20-step 严格 1C1S 与独立恢复验证。论文尺度实验已推进到 Qwen2-MoE 14.32B 的 20-step NoCheckpoint 对照，以及 GPT dense 6.33B 的 200-step 对照/检查点配对：NoCheckpoint 侧完整通过，但检查点侧在 epoch 16 因后台服务率不足触发有界背压，因而尚未形成可报告的 200-step overhead。

## 1. 动机与研究问题

AdaCheck 利用 worker 间及相邻迭代间的训练状态冗余，通过梯度增量和远端 CPU 更新降低频繁
checkpoint 的成本。其论文在 LLaMA-30B、DeepSeek-V2-Lite、GPT-7B 和 GPT-MoE 的选定配置上
报告了每个 iteration 保存一次 checkpoint。公开代码能够帮助理解冗余检测和固定 worker group，
但没有覆盖论文系统的完整远端更新、提交和恢复路径，因此我们不能把自己的运行结果称为
AdaCheck 端到端复现。

我们的核心问题是：给定与旧分组无关的全局冗余信息，能否联合考虑 rank 的状态重叠、并行角色
和物理故障域，选择更好的 checkpoint peer？需要特别避免循环定义：不能先按 AdaCheck 固定组
得到组内“100 MB 冗余”，再拿这个数优化分组；必须先构造任意合法 rank 对的两两代价矩阵，
再求全局分组。

暂定论文主线不是“一个更好的配对 heuristic”，而是：

> 将逻辑状态冗余转化为满足恢复约束的物理保护布局。

## 2. 数学形式化

设 rank `i` 的编码后恢复对象集合为 `E_i`，对象 `o` 的编码字节数为 `w_o`。若 `i` 与 `j`
互为 peer，双向需要补传的对象代价为

```text
C_ij = sum(w_o, o in E_i - E_j) + sum(w_o, o in E_j - E_i).
```

令 `x_ij` 表示是否选择跨主机边 `(i,j)`。两机各 8 个 rank 时，在以下约束下求最小权完美匹配：

```text
minimize    sum(i,j) x_ij C_ij
subject to  every rank has exactly one peer
            host(i) != host(j)
            parallel roles and recovery objects are compatible
            service capacity and recovery budget are feasible.
```

当前离线 8x8 问题用精确位掩码动态规划求解，并用 `8!` 穷举校验。精确性只针对冻结代价矩阵上的
加性字节目标，不等价于端到端时延最优；网络争用、后台服务长尾和训练通信仍需实测。
当前 solver 硬约束每个 rank 唯一配对和跨主机，并由对象差集定义恢复代价；service capacity
仍是求解后的 runtime feasibility gate，尚未作为优化器内部的容量约束。上式是后续联合优化的
目标形式化，不能反过来描述为当前代码已实现的全部约束。

静态恢复正确性的依据是：若 `j` 所在故障域保存 `E_j` 和差集 `E_i-E_j`，则能够重建 `E_i`。
在线系统还要求对象属于同一 epoch，且 catalog、peer、编码和 checksum 一致。只有 16 个 rank
全部满足这些条件，才可发布全局恢复点。

## 3. 当前分组和系统架构

当前在线部署为 node17/node8 两台各 8 张 A40、`PP=4, TP=2, DP=2`、DP-first rank mesh：

```text
node17: rank 0  1  2  3  4  5  6  7   (DP replica 0)
                |  |  |  |  |  |  |  |
node8 : rank 8  9 10 11 12 13 14 15   (DP replica 1)

groups: (0,8), (1,9), ..., (7,15)
```

每对 rank 的流水线并行（Pipeline Parallelism, PP）和张量并行（Tensor Parallelism, TP）坐标
相同，数据并行（Data Parallelism, DP）坐标不同，并跨越物理主机故障域。这里是八个两-rank
恢复组，不是把同一主机的八个 rank 放进一个组。

当前系统包含彼此分离的两条路径：

```text
offline planner / service replay:
AdaCheck-style redundancy evidence
        -> group-independent frozen object catalog
        -> all-pairs peer-delta matrix
        -> fixed / random / redundancy_max / profile_exact
        -> offline bytes analysis and frozen-catalog service replay

online Merak runtime (r45/r46):
pre-update asynchronous gradient capture
        -> mirror-DP peers (0,8), ..., (7,15)
        -> host-local CPU materialization + cross-host runtime delta
        -> PIN -> atomic global publish -> SEAL
        -> fresh-job load and continued-training oracle
```

训练线程不等待远端确认；后台队列有界且不能丢弃或合并中间 AdamW update。若最新全局 sealed
epoch 相对训练落后超过一步，则该 run 即使最终 drain 完成也不算 1C1S。每个 reciprocal peer
还必须提交相同的 epoch equivalence checksum，避免混合错误材料。

### 3.1 可追溯的接口与代码布局

全局两两冗余分组从不可变 catalog 建立矩阵，短签名如下（源码原样节选）：

```python
def build_peer_delta_matrix_from_catalog(
    objects: Sequence[Mapping[str, Any]],
    *,
    world_size: int,
    ranks_per_host: int,
    catalog_hash: str,
    profile: PeerResourceProfile | None = None,
) -> PeerDeltaMatrix:
```

策略入口同样位于 `peer_delta.py`：

```python
def peer_grouping_for_policy(
    matrix: PeerDeltaMatrix,
    policy: str,
    *,
    seed: int = 20260901,
) -> RankGrouping:
    if policy == "fixed":
        return fixed_peer_grouping(matrix)
    if policy == "host_global":
        return RankGrouping("host_global", fixed_peer_grouping(matrix).groups)
    if policy == "random":
        return random_peer_grouping(matrix, seed)
    if policy in {"redundancy_max", "profile_exact"}:
        return exact_peer_grouping(matrix, policy=policy)
    raise ValueError(
        "peer policy must be fixed, random, redundancy_max, profile_exact, or host_global"
    )
```

`redundancy_max` 和 `profile_exact` 使用 exact bitmask solver，但当前只用于 frozen catalog 的
离线分析和 service replay。r45/r46 的 online Merak 路径只支持 mirror-DP；在该对称场景中其
输出与 fixed 相同，尚未把任意 exact pairing 接入 online recovery。模型实验同样由统一注册表
和一个入口分派，保证早期多模型和后续大模型使用可追溯的本地配置（以下为源码原样节选）：

```python
SUPPORTED_WORKLOADS = (
    "bert",
    "gpt2",
    "opt",
    "llama_1b",
    "llama_3b",
    "llama_3_8b",
    "gpt_6b",
    "qwen2_moe_14b",
    "deepseek_v2_lite",
)
```

```python
def get_workload(name: str) -> Workload:
    if name == "bert":
        config = BertConfig(**load_bert_config("bert-large-uncased"))
        return Workload(name, "BERT-4L-1024", config, BertForMaskedLM)
    if name == "gpt2":
        config = GPT2Config(**load_gpt2_config("gpt2"))
        return Workload(name, "GPT-2", config, GPT2LMHeadModel)
    if name == "opt":
        config = OPTConfig(**load_opt_config("opt-350m"))
        return Workload(
            name,
            "OPT-350M",
            config,
            OPTForCausalLM,
            leaf_modules=(OPTAttention,),
            split_method="layer_split",
        )
```

函数后续分支以相同接口覆盖两种 LLaMA-like、小规模到论文尺度的 LLaMA/GPT、Qwen2-MoE 与
DeepSeek-V2-Lite；报告中的模型名和参数量应以该注册表及其输出 metadata 为准，而不是口头估算。

当前对本课题最有用的代码按职责组织如下，文件树只列出会议讨论和复现实验的主路径：

```text
Merak/core/checkpoint/
  peer_delta.py               # frozen catalog 矩阵与 fixed/random/redundancy_max/profile_exact
  peer_delta_runtime.py       # online mirror-DP runtime delta
  matched.py                  # 早期双机 service replay 的保护、commit 与 reduced restore
  incremental_*.py            # 异步捕获、1C1S 后台流水线
  global_commit.py            # PIN / atomic publish / SEAL
  replica_service.py          # checkpoint service 与传输度量
  recovery.py, state_oracle.py # fresh-job 恢复与数值 continuation 合同
examples/checkpoint_grouping/
  workloads.py, run_workload.py       # 可复现实验模型注册、并行配置和训练入口
  build_reduced_capture.py             # detector catalog 冻结
  matched_service_harness.py           # 早期 fixed/random/greedy/exact service 回放
  incremental_1c1s_harness.py          # 两机严格 1C1S gate
  peer_delta_service_harness.py        # peer-delta 与恢复协议实验
  distributed_preflight.py             # 分布式 NCCL/rank 拓扑连通检查
tools/
  run_two_node_*.sh, analyze_*.py      # 双机执行、汇总和机会分析
test/core/checkpoint/
  test_peer_delta.py, test_matched.py  # catalog pairing、exact solver、保护与恢复
  test_incremental_*.py, test_global_commit.py, test_state_oracle.py
                                      # 1C1S、全局提交和数值恢复合同
docs/
  report-09-09.md                      # 本汇报
  two_node_v2_experiment_matrix.md     # 两机实验台账和证据边界
  figures/                             # 已生成的对比图
```

## 4. 与 AdaCheck-style baseline 的当前结果

下表中的 `AdaCheckStyleFixed` 是在我们的同一实现中保留固定 mirror-DP pairing 的可控 baseline，
不是对 AdaCheck 论文系统绝对性能的复现。论文报告、公开代码行为和我们的实测始终分开。

### 4.1 早期多模型实验如何形成当前 hypothesis

早期 capture 实验使用 node17/node1、每机 8 张 GPU、共 16 ranks、`PP4/TP2/DP2`，模型均随机初始化
并使用本地 synthetic data。它们的历史命令没有显式传入 mesh order，按入口默认使用 `pp-dp`：
一台主机放置两个 PP stage；这与当前在线实验显式使用的 `dp-pp` 不同。并行度相同不代表 rank
placement 或状态重叠矩阵相同。这些结果不是最终端到端训练对比，但构成了后续研究设计的直接依据：

#### 早期 capture 的实际步数

下表只统计本节早期 detector catalog 的真实训练/采样步；均以 artifact 的 `command.txt` 中
`--max_steps`/`--check_steps` 和 node-17 launcher log 的 `steps:` 交叉核对，而不是按目录名推断。

| 模型 | 用于本节的成功 capture | 实际 training/capture steps | 原始字段与边界 |
|---|---|---:|---|
| BERT-4L-1024 | `bert-seed-20260831-v7`；`bert-world-seed-20260901-v1` | `1`；`1` | 两者均为 `CKPT_CAPTURE_ONLY=1`、`max_steps=1`、`check_steps=1`，仅冻结 epoch-1 catalog |
| GPT-2 | `gpt2-seed-20260831-v1`；`gpt2-world-seed-20260901-v1` | `1`；`1` | 同上；world artifact 的 rank-0 log 明确记录 `steps: 1` |
| OPT-350M | `opt-seed-20260831-v4`；`opt-world-seed-20260901-v1` | `1`；`1` | 同样是 capture-only 的 epoch-1 catalog；没有 GPT-2/BERT 那样的 formal runtime |
| LLaMA-like-1B | `llama_1b-seed-20260831-v1`；报告另引的 DP-first catalog、r51/r55 | `1`；`2` | 旧 data-scope artifact 为 `max_steps=1`；后面三项是 `max_steps=2`、`check_steps=1,2` 的 two-snapshot static capture |
| LLaMA-like-3B | `llama-3b-world-seed-20260901-v1` | `1` | `CKPT_CAPTURE_ONLY=1`、`max_steps=1`、`check_steps=1`；只支持 offline opportunity |

这些都是小步训练后采集状态、构建冻结 catalog 的实验，不启动训练路径中的远端提交/materialization
闭环，也不验证 fresh-job recovery。离线 fixed/exact pairing 只是对同一 catalog 的后处理，不增加
任何训练 step；BERT/GPT-2 后来的 `5 x 8` 是 frozen-catalog service-replay epoch，亦不是模型训练。
因此表中不混入后来的两机 stable-compute 20-step、LLaMA-1B r46/r47，或尚未执行的 200-step
正式目标；nnScaler 的静态 model-slice 与失败 preflight 也没有可计入的 training/capture step。

| 模型 | AdaCheck detector / grouping 观测 | 由此得到的 insight |
|---|---|---|
| LLaMA-like-1B，0.941B 参数 | 16-rank full state 为 34.297 GB，detector reduced catalog 为 10.117 GB，逻辑缩减 70.502%；但早期旧 `data`-scope catalog 的 fixed 与 exact pairing 均无额外收益 | “状态可压缩”不等于“任意物理分组都能利用该冗余”；旧 scope 也未覆盖论文所述全部 relevant communication groups |
| BERT-4L-1024，122M 参数 | 单次 world-scope frozen catalog 上，fixed-to-exact 逻辑 pair bytes 降低 23.022% | 首次给出与旧 group 无关的全局两两 catalog 非零机会；仍是 one-observation 机会证据 |
| GPT-2，163M 参数 | 单次 world-scope frozen catalog 上，fixed-to-exact 逻辑 pair bytes 降低 71.054% | 模型间机会差异很大，不能用单一模型决定课题价值；仍是 one-observation 机会证据 |
| LLaMA-like-3B，3.015B 参数 | 单次 frozen catalog 的 fixed/exact pair bytes 为 12.658/9.513 GB，降低 24.851%，16/16 rank 静态重建通过 | 机会可扩展到更大模型，但当时只完成离线 catalog/reconstruction，未测在线训练 |

BERT 和 GPT-2 随后完成了双机 frozen-catalog service-replay matrix：每个模型包含 5 个独立
paired trials、
每个策略 8 个 epoch，并轮换策略顺序。相对 fixed grouping，BERT 的 application protocol bytes、
InfiniBand 性能管理代理（Performance Management Agent, PMA）端口发送量和有限样本
completion-P99 分别下降
`23.016%`、`23.018% +/- 0.044%` 和 `11.039% +/- 5.771%`；GPT-2 分别下降
`71.025%`、`70.975% +/- 0.023%` 和 `31.135% +/- 3.191%`。每个 trial 只有 6 个计时样本，
这里的 P99 实际接近样本最大值，不能称 production-grade tail latency。

BERT 还完成 5 次 paired recovery matrix，共 20 次 ordinary catalog restore。优化分组相对 fixed
的最慢 service 恢复时间变化为 `-8.596% +/- 4.541%`，95% 上界 `-4.055%`，通过预注册的
“不比 fixed 慢 5%”门槛。这证明早期字节收益没有以更慢的目录重建为代价，但它仍不是完整
fresh-job continued-training recovery 或物理主机故障。

这些观察依次促成了当前设计：从 group-local 数值转为 world-scope catalog，从离线逻辑字节
转为 application/PMA 实测，再补全 epoch 原子提交、1C1S 和 continued-training oracle。因此
后续两机 16-rank 端到端实验不是突发构造，而是在早期多模型结果上逐级收紧证据合同。

三种模型的差异来自状态重叠矩阵而非并行度：fixed 把 `PP0`/`PP1` 分别与 `PP2`/`PP3` 配对，
exact 对三者都选择 `PP0<->PP3`、`PP1<->PP2`。BERT 的配对 overlap 从 84.10 MB 增至
584.20 MB，GPT-2 从 263.46 MB 增至 1,977.79 MB，LLaMA-like-3B 从 22.096 GB 增至
23.669 GB。完美匹配下 rank 总需求不变，双向 pair bytes 等于该常数减去两倍 overlap，因此
分别得到 23.022%、71.054% 和 24.851% 的降幅；GPT-2 的 PP 切分后矩阵最偏斜。

#### GPT-2 约 70% 字节差的来源

这里的“早期双机 GPT-2”先是一次真实训练的 **单 step capture**，而不是端到端 checkpoint
试验：`runs/matched-two-node-capture/gpt2-world-seed-20260901-v1/command.txt` 固定早期
node17/node1、每机 8 rank、`PP4/TP2/DP2`、默认 `pp-dp`、FP16、seq128，并传入
`--max_steps 1 --check_steps 1` 与 `CKPT_CAPTURE_ONLY=1`；rank-0 日志也只记录 `steps: 1`。
它在 epoch 1 采集 AdaCheck detector 的 world-scope catalog，未启动远端 service、传输、原子
commit 或 fresh-job restore。因此这一训练 step 只提供状态样本；不能把随后算出的字节差称为
训练时间、online AdaCheck gain 或端到端恢复结果。

约 `71.054%` 也不是该 capture 内的 `grouping-exact.json` 所得：该文件与 fixed 都是
`(0,8)..(7,15)`。数字来自同一冻结 catalog 的后续 8x8 cross-host planner
`runs/peer-delta-opportunity/gpt2-world-v1/`：fixed 仍为 `(0,8)..(7,15)`，而
`profile_exact`/`redundancy_max` 都选择 `(0,12)..(3,15),(4,8)..(7,11)`。其 pair-local
logical bytes（两向、对端缺失对象的逻辑 tensor-byte 和）从 `4,825,431,552` 降至
`1,396,784,640`；aggregate pair peak（每对两向 payload 的较大者再求和）从
`4,164,787,968` 降至 `839,950,080`。两者都是冻结 catalog 的成本函数，既不是 serialized
wire bytes，也不是 NIC 计数。

机制是可观测的 PP state 异质性，不是任意重排都会获益。在这一早期 `pp-dp` 布局中，node17 承载
`PP0/PP1`、node1 承载 `PP2/PP3`；planner matrix 中四个 stage 代表 rank 的冻结 requirement
分别为 `524,040,164`（rank 0）、`121,411,556`（rank 4）、`109,615,076`（rank 8）和
`583,022,564` bytes（rank 12）。按本地序号 mirror 的 fixed 因而配 `PP0->PP2`、`PP1->PP3`，没有利用该 catalog
中更强的跨 stage 重叠；exact 改为 `PP0->PP3`、`PP1->PP2`，观测到的配对 overlap 由
`263.46 MB` 增至 `1,977.79 MB`。rank 总需求不变时，双向 pair-local bytes 随 overlap 的两倍
减少，故得到大幅差距；它是该 snapshot、该 PP/TP/DP rank layout 的机会证据，不是普遍的
“固定分组必然差”结论。

同一 bundle 后来进入 `runs/peer-delta-formal/gpt2-world-v1/` 的 frozen-catalog service replay：
5 个 paired trials、每 policy 8 个 service epoch、丢弃前 2 个。它把 TCP application protocol
bytes/epoch 从 `4,827,295,349` 降至 `1,398,716,824`（`71.025%`），并把独立 PMA 物理 NIC
TX bytes/epoch 从 `5,127,598,151` 降至 `1,488,263,626`（`70.975% +/- 0.023%`）；前者含请求和
响应，后者才是实际网卡发送计数，均不同于 aggregate peak。这个 replay 证明固定 catalog 的
分组差异到达了两机数据面，但不是 GPT-2 end-to-end：它重放 bundle，未在训练中运行在线 encoder/
1C1S，也没有 fresh-job continued-training；其 two-service ACK 还是顺序可见、非 distributed
atomic commit，且每 trial 仅有 6 个有效 completion 样本。故应报告为“单 step 机会 +
frozen-catalog runtime-traffic”证据，而非完整在线训练收益。

### 4.2 当前端到端与负对照

| 实验 | Baseline/对照 | 主要结果 | 目前能说明什么 |
|---|---|---|---|
| 对称 LLaMA-like-1B catalog | AdaCheckStyleFixed | fixed 与 exact 都选择 `(0,8)..(7,15)`，改善 0%；随机错配显著更差 | 正确的零机会负对照：优化器不会凭空制造冗余 |
| 两个 nnScaler LLaMA-3-8B 布局 | fixed physical placement | 64 条跨 host 边的 model-slice overlap 均为 0，fixed 与 exact 改善 0% | 仅 static model-slice 负对照；无 AdaCheck detector evidence、optimizer/runtime placement 或 checkpoint wire 实测 |
| DDP toy mechanism end-to-end，stable-compute 20-step | full remote update（弱协议消融） | 两层 width-512 MLP 真实执行 forward/backward/AdamW，但人为重复计算 512 次；peer-delta 的跨 host logical update payload 由 2.692 GB 降到 4.316 MB，约 -99.840% | 仅证明异步队列、delta 和 fresh-job chain-replay 机制；不能作为 grouping 收益或 AdaCheck 对比 |
| 真实 Merak 0.941B，r46 | NoCheckpoint r47 | 20/20 global sealed epochs，所有 boundary lag=1；稳态 3.979 vs 3.777 s/step，单次约 +5.34% | 当前源码下 20-step 严格 1C1S gate 通过；性能仍是单次开发 A/B，无置信区间 |
| 数值恢复 r45 | uninterrupted r41 | 16 ranks 从 epoch 3 fresh restore，下一步 loss 都是 9.6475；全部状态通过预注册数值合同 | 从“可加载”推进到“可数值继续训练”；不是逐位恢复或物理 host loss |
| LLaMA-3-8B r31/r48/r50 | strict `lag<=1` | GA64 与 GA96 均完成 5/5 sealed，但最大 lag 都为 2；r50 将 shared stage concurrency 限为 2 后 mean/max commit 反而升至 20.240/41.088 s | 增大计算窗口和限制 staging 并发均未消除 tail，尚不能进入 200 steps；该旋钮已删除 |

r45 的 BF16 model 最大每张量相对 Linf 误差为 `0.0025907`，低于一个 BF16 epsilon
`0.0078125`；FP32 optimizer 最大绝对误差为 `1.19e-7`，且无非有限值。CPU 与 CUDA AdamW
不保证逐位相同，因此我们同时保留 `numerical_continuation_valid=true` 和
`bitwise_continuation_valid=false`，不把数值合同描述成 bitwise equivalence。

截至 r72--r75 的 GPT dense 6.33B 状态须与 r45 分开：r72 的 20-step sealed epoch 可由 fresh job
加载并完成续训，但训练 epoch-20 与 restored-boundary 的 model/optimizer checksum 在 `16/16` rank
不一致，且 `CKPT_VALIDATE_MATERIALIZED_STATE=0`，未启用 materialized numerical oracle。r73/r74 的
21-step `off` reference 彼此可重复，但其 AdamW kernel 与 r72 checkpoint/materialized 路径不匹配；
故不能据此比较宣称 r72 数值恢复成功或失败。r75 以 `MERAK_FORCE_SCALAR_ADAMW=1` 对齐后，21-step
`off` 为 `COMPLETE`、launcher `0/0`、16 digests，step-21 loss `6.8438` 与 r72 restore 相同，定位此前
loss 差异为 optimizer-kernel baseline confound；它仍不替代 r72 未启用的 boundary numerical oracle，
不能宣称 20-step 数值恢复通过。目前 r72 只证明 loadability/liveness。这不改变 r45 在其预注册数值
合同下的独立 continuation 结论（证据：各 `runs/two-node-v2/*r72*`、`*r73*`、`*r74*`、`*r75*` artifact）。

**今晚最新门禁。** r76 的同 GPT dense 6.33B、20-step scalar sealed checkpoint 已完成全部训练 step，
但最终 16-rank full materialized oracle 并发请求使 node8 service 被 OOM killer 杀死（约
`301,945,960 kB` anon RSS）；launcher 为 `143/1`、checkpoint metrics 为 0，故没有数值 verdict。
已在 service 边界串行化 full materialized response（普通请求仍可并发），14 个 focused tests 通过；
r77 已 `COMPLETE`：16/16 rank valid，epoch-20 materialized oracle 均 numerically close，
但 0/16 bitwise；最大 lag 为 2，训练 phase `1680.702 s`，remote-delta payload/wire 分别为
`4,416,152 B`/`5,172,337 B`。它通过的是 20-step materialization 数值门禁；该 run
`RUN_RESTORE=0`，没有 fresh restore。恢复活性仍仅由 r72 支持，完整恢复数值闭环仍待完成
（证据：`runs/two-node-v2/merak-gpt6b-bf16-fulltp-seq1024-ga64-peer-sealed-lag2-oracle-20step-r76-20260908T200000Z/`、
`runs/two-node-v2/merak-gpt6b-bf16-fulltp-seq1024-ga64-peer-sealed-lag2-oracle-20step-r77-20260908T202000Z/`）。

r78 的 post-PIN-fix 5-step GPT dense 6.33B 中，前台约 10 s 的周期性 stall 已消失（与 PIN
不再等待 CPU materializer 相符），但 global lag 为 `[1,1,2,1,2]`、max=2，违反 configured lag=1。
五个 epoch 都已 global commit、16 份 metrics 连续；`FAILED` 仅来自该门禁，因而无 digest/summary。
80 条 local-full 的 commit mean/max 仍为 `9.637/21.626 s`、enqueue-to-commit mean/max 为
`11.179/21.634 s`，说明后台服务率仍不足。随后已修复 `get_materialized` 的 staged-first race，
12 项 materializer 与 14 项 service tests 通过。

r79 已 `COMPLETE`，train/restore launcher 均为 `0/0`：16/16 active materialized oracle
numerically close、0/16 bitwise，max lag=2，并产生 16 份 restored-boundary 和 16 份 restore
digest。same-source scalar `off` 6-step r80 亦为 `COMPLETE`、launcher `0/0`；二者 step-6 loss
均为 `8.3750`，r79 train/restore phase 为 `1506.285/654.971 s`，r80 phase 为 `124.882 s`。
`r79/recovery-oracle.json` 的 `valid=true`、`numerical_continuation_valid=true`、
`logged_losses_equal=true`；但 boundary/final 的 model+optimizer 均在 16/16 rank bitwise mismatch，
scheduler 匹配。故口径为 lag-two、healthy-service worker-restart 的数值 continuation，不是
strict lag-one、physical host failure 或 durable recovery
（证据：`runs/two-node-v2/merak-gpt6b-bf16-fulltp-seq1024-ga64-peer-sealed-lag1-tailfix-5step-r78-20260908T205632Z/`、
`runs/two-node-v2/merak-gpt6b-bf16-fulltp-seq1024-ga64-peer-sealed-lag2-oracle-restore-5plus1-r79-20260908T211216Z/`、
`runs/two-node-v2/merak-gpt6b-bf16-fulltp-seq1024-ga64-off-scalar-6step-r80-20260908T214942Z/`）。

r81 的同 fixed-mirror scalar GPT dense 6.33B 200-step sustained-rate 尝试为 `FAILED`：launcher
`143/1`，rank-0 只到 step 15；node8 rank8 enqueue epoch 16 时触发 bounded-inflight
`CheckpointBackpressureError`，global control files 仅为 0--11。它没有 final rank metrics、summary/analysis
或 digests，故没有可报告的 final max-lag、remote payload/wire、cumulative shared handoff 或 queue/commit tail；
失败前 train phase `671.269 s` 含 setup/abort，不是稳态时间。它是 200-step 持续服务率负结果，不构成
200-step 连续性、strict 1C1S、P99、恢复、端到端、overhead 或 grouping 证据
（证据：`runs/two-node-v2/merak-gpt6b-bf16-fulltp-seq1024-ga64-peer-sealed-lag2-200step-r81-20260908T215756Z/`）。

r82 是与 r81 同 source snapshot、tracked diff、模型、并行布局、序列长度、GA、seed、constant LR 和
scalar AdamW 的 200-step NoCheckpoint 对照。它为 `COMPLETE`、launcher `0/0`，完成 200/200 steps
并生成 16 份 iteration-200 terminal digest；完整 phase wall 为 `2126.841 s`。预注册的 steps 2--200
窗口含 199 个 rank-0 step-wall 样本，mean/P50/P95/P99/max 分别为
`10.305/10.306/10.330/10.349/10.353 s`（nearest-rank percentile），step-200 loss 为 `6.5625`。
因此训练本身可持续运行 200 steps，r81 的失败可归因于当前 checkpoint service/materialization
供给率而非 GPT 训练不稳定；但 treatment 没有完成，所以 r81/r82 仍不能计算 checkpoint overhead
或声称 200-step 1C1S（证据：`runs/two-node-v2/merak-gpt6b-bf16-fulltp-seq1024-ga64-off-scalar-200step-r82-20260908T221200Z/`）。

### 4.3 r50--r82 实验归纳

下表覆盖 r50--r82 的全部编号，并按科研用途合并连续诊断。`COMPLETE` 只表示该 run 的预设命令
正常结束；mode-off、metadata-only 或 relaxed-lag run 即使完成，也不能据此称为 checkpoint、
strict 1C1S 或 grouping 正结果。

| Runs | 终态 | 模型与目的 | 关键观测 | 进入论文/汇报的价值 |
|---|---|---|---|---|
| r50 | `FAILED` | LLaMA-3-8B，GA96、限制两路 shared staging 的 5-step strict gate | 5/5 sealed，但 lag=`1,1,2,2,2`；local-full commit mean/max=`20.240/41.088 s`，比 r48 更差 | 有价值的尾延迟负结果：单纯限并发不能恢复 strict 1C1S；该旋钮已删除 |
| r51 | `COMPLETE` | LLaMA-like-1B，PP-first FP32 两快照 world catalog | 可重建；full-state fixed/exact=`7.911/5.813 GB`，静态上界改善 `26.5109%`；真正 changed-object activity gap 为 `0%` | 证明“full-state 分组机会”可能被活跃更新集合消除，支持必须测 AdaCheck-faithful payload 的动机 |
| r52--r54 | `FAILED/EXCLUDED` | r51 的 BF16 capture 兼容性修复 | 分别止于 BF16 NumPy、tensor hash、0-D Adam step 编码，均无完整 catalog | 仅保留为实现审计/exclusion，不进入模型或性能结果 |
| r55--r56 | `COMPLETE/FAILED` | LLaMA-like-1B BF16 catalog 与 pre-optimizer update catalog | r55 两 epoch 可重建，full/reduced=`34.297/17.148 GB`，fixed=exact；r56 两 epoch catalog 完整但同步 hash 导致 lag=2，且 payload fidelity 为 false | r55 是 BF16 零机会负对照；r56 证明捕获可行并暴露同步全量 hash 不能放在关键路径 |
| r57--r60 | `EXCLUDED/COMPLETE` | Qwen2-MoE 14.32B 兼容性、catalog 和 baseline | r57 no-ZeRO OOM；r58 ZeRO-1 one step 通过；r59 两步 metadata catalog，fixed=exact、random 更差但无 payload；r60 20/20 NoCheckpoint，steps 2--20 mean=`4.660 s` | 建立稀疏大模型 no-EP 可运行边界与正式对照；不能声称 expert placement、checkpoint 或 grouping gain |
| r61--r65 | `EXCLUDED/COMPLETE` | GPT dense 6.33B tracing/TP2 映射审计 | r61/r62 分别暴露 FX control-flow 和 TP2 shape 问题；r63/r64 的 MLP mapping 不完整，排除；修复后 r65 full-TP2 one step、16 digests 完成 | 形成可信的 GPT 论文尺度执行配置；r61--r64 不进入性能表 |
| r66--r67 | `COMPLETE/FAILED` | GPT seq128 短 NoCheckpoint 与 sealed-path rate gate | r66 GA1 五步通过，steady mean=`0.4775 s`；r67 GA64 三步 sealed，但 lag=`1,2,2`，strict fail；二者 GA 不同 | r67 定位 local-full/service tail；不可把 r66/r67 当 overhead 配对 |
| r68--r69 | `COMPLETE/FAILED` | GPT seq1024、GA64 首个同 workload ordered pair | r68 off steady mean=`10.292 s`；r69 fixed peer mean=`12.356 s`，观察差 `+20.05%`，但 global lag=`1,1,2` 且仅 `n=2` | 仅 development observation；证明比较配置已对齐，也证明 strict tail 尚未过门禁 |
| r70--r72 | `FAILED/COMPLETE/COMPLETE` | GPT 20-step relaxed-lag 数据面、对照与首次 restart | r70 20 个 global epoch 但后处理脚本污染；r71 off steady mean=`10.308 s`；r72 lag-two checkpoint/restart 完成，单次 step-wall 比 r71 `+31.28%`，仅证明 loadability/liveness | 建立 20-step 数据面和性能量级；不能称 strict 1C1S、数值恢复或正式 overhead |
| r73--r75 | `COMPLETE` | GPT uninterrupted references 与 optimizer 公平性 | r73/r74 non-scalar 16/16 bitwise 可重复；r75 scalar step-21 loss=`6.8438` 与 r72 restore 相同 | 观测与 optimizer-kernel baseline confound 的解释一致，但不替代 r72 缺失的 boundary numerical oracle；后续公平配置冻结为 scalar AdamW |
| r76--r77 | `FAILED/COMPLETE` | GPT 20-step materialized numerical oracle | r76 的 16 路并发 full GET 使 service OOM；串行 response 后 r77 完成，16/16 numerical-close、0/16 bitwise，max lag=2 | r77 证明 CPU materialization 数值正确；仍无 fresh restore、strict lag-one 或 grouping treatment |
| r78 | `FAILED` | GPT post-PIN strict lag-one 5-step gate | 前台约 10 s 周期 stall 消失，但 lag=`1,1,2,1,2`；local-full commit mean/max=`9.637/21.626 s` | 结果与同步 PIN stall 已移除、后台 steady-state service rate 仍不足的诊断一致 |
| r79--r80 | `COMPLETE/COMPLETE` | GPT lag-two healthy-service fresh restart + same-source reference | recovery oracle `valid=true`，step-6 loss 均 `8.3750`，numerical continuation 有效但非 bitwise | 当前最强 GPT 恢复证据；边界明确为 lag-two、健康 service、worker restart，不是 host failure/durability |
| r81--r82 | `FAILED/COMPLETE` | GPT 200-step fixed-peer treatment / NoCheckpoint 对照 | r81 epoch 16 有界背压失败、global control 到 11；r82 完成 200/200，steady mean/P99=`10.305/10.349 s` | 关键系统结论：训练 baseline 可持续，当前 checkpoint consumer rate 不足；不能计算 overhead |

对汇报最重要的五项证据是 r51 的“静态机会与真实 activity 不一致”、r59 的大 MoE 对称零机会、
r77 的 16-rank materialization 数值正确性、r79/r80 的 fresh-job 数值 continuation，以及 r81/r82
共同给出的 200-step service-rate 反例。其余 run 的主要用途是证明配置如何被校正或为何被排除，
避免用失败/不公平配置支撑论文结论。

**表外最新进展（r83）。** 在同一 GPT dense 6.33B、`PP4/TP2/DP2`、seq1024 配置中，将 GA64
提高到预注册的第一个候选 GA80 后，20-step fixed-mirror sealed run 已 `COMPLETE`：16/16 rank
有效且连续，20/20 global committed，每一步 rank/global boundary lag 都等于 1。steps 2--20
step-wall mean 为 `13.694 s`；320 次 `materializer.apply` 的 mean/P95/P99/max 为
`8.852/10.456/10.891/11.144 s`，两侧 service 最终均无 backlog/poison。它说明当前物化路径在
GA80 计算窗口下通过 20-step strict lag-one cadence gate；由于尚未执行本配置的 numerical oracle/
fresh restore，不能称完整 strict 1C1S。GA80 也改变了每个 optimizer step 的有效 batch；此外，
r83 加入 apply instrumentation 后的 source snapshot 为 `3296a969...`，不同于 r81/r82 的
`a47d2658...`。因此它只是 cadence sensitivity 结果，不是 grouping 收益、训练加速或跨 revision/
GA64 的正式 overhead 对比。

**表外最新进展（r84）。** r84 沿用 r83 的 GA80 source/configuration，完成 5-step strict lag-one
训练及独立 fresh-worker restore：train/restore launcher 均为 `0/0`，16/16 rank metrics
valid/contiguous，global controls 为 0--5，所有 boundary lag 均为 1，训练与恢复后的 digests
分别覆盖 iteration 5 和 6。active `materialized_state_oracle` 在 16/16 ranks 数值接近、0/16
joint model+optimizer bitwise equal，最大 BF16 relative Linf error=`0.004545454545454545`、
optimizer max abs error=`1.1920928955078125e-7`，且无 nonfinite/structure error；这是训练边界
materializer agreement，不是恢复 continuation oracle。恢复 step-6 loss=`8.2500`、step wall=`411.318 s`；
两侧 service 保持健康，epoch 5 的 peer restore 成功。train/restore phase wall=`1496.489824706/630.197100218 s`；
80 次 apply 的 mean/P95/P99/max=`8861.533/11904.912/13772.067/13772.067 ms`，两侧无 pending/staged
bytes 或 poisoned source。artifact 没有 `recovery-oracle.json` 或独立 uninterrupted GA80 reference，
因此只能称 healthy-service worker-restart 下的 fresh-worker loadability/liveness，不能称 numerical
continuation，也不能把它升级为完整 strict 1C1S recovery contract。其 failure model 为
`healthy-services-worker-restart`（无真实 host/service/network/durable fault）；source snapshot 与 r83
相同、与 r81/r82 不同，故不与 r83/r82 形成 paired performance/overhead 结论。

### 4.4 AdaCheck breakdown 分析

我们将论文机制拆为六段：冗余检测、gradient/direct-state encoding、固定 worker/rank grouping、
远端 CPU materialization、原子 commit 和 recovery。公开 commit `cf910be` 可直接核对 detector、
固定 group 与本地 differential `torch.save`；远端 CPU updater 不可达、remote-buffer constructor
无 caller、loader 为空。因此我们的 `AdaCheckStyleFixed` 仅是在共同底座上的可控 grouping
baseline，不是公开论文系统的端到端复现。

这里的“不完整”不表示 AdaCheck 没有 checkpoint 保存或冗余核对。公开 artifact 已包含
`RcharaCheckpointSaver` 的本地/差分保存、状态哈希与跨 rank 冗余比较，以及 remote group、CPU
buffer 和发送索引的准备代码；论文也描述并评估了逐步 gradient update 与远端 CPU optimizer。
缺少的是从公开训练入口可达的完整闭环：buffer constructor 没有 caller，未找到持续接收并应用
每步更新的可运行 remote CPU updater、跨 16 rank 的一致 commit/manifest，以及与 reduced state
对应的 fresh-job loader。因此我们能够复用和核对其 detector 思路，但不能仅靠公开 artifact
复现论文的“在线远端更新 -> 全局提交 -> 故障后续训”绝对性能。

breakdown 按可归因变量推进：`data`-scope 到 world-scope catalog、full-state/reduced catalog 与
16-rank reconstruction、fixed/random/exact matching、冻结 catalog 的 service replay/PMA NIC bytes/
有限样本 tail，以及真实 Merak 的 1C1S queue/materialization、fresh-job numerical continuation 和
service-process fault smoke。静态逻辑字节、encoder application payload、protocol/NIC bytes 与时间
序列 staleness 分开记账，避免用一个 proxy 代替在线系统行为。

这一区分已排除两个错误外推。r51 的 PP-first LLaMA-like-1B full-state static pair-byte upper
bound 为 `26.5109%`，但未测 AdaCheck online detector、gradient record、encoder、NIC、queue 或
迁移状态，不能称 AdaCheck online gain；当前 DP-first 对称在线 catalog 则是 fixed 已等于 exact 的
零机会负对照。未来的 grouping update 必须共享 AdaCheck-style encoder 和冻结 catalog，并在每个
trial 内冻结 grouping；snapshot 时序、ordered log、service limit 和 fault model 保持相同，唯一
treatment 才能是 pairing。

#### 当前 live grouping 边界

r81 仍采用 fixed mirror。`CKPT_INCREMENTAL_GROUPS` 虽可传入 rank pair，但当前 live `peer_delta`
恢复要求 pair 跨 host、PP/TP 坐标相同、DP 坐标不同且 optimizer topology 一致；在两机
`DP2/PP4/TP2` 下，每个 rank 只有唯一合法 DP peer，whole-rank matching 必然退化为
`(0,8)..(7,15)`。offline exact/hierarchical/matched solver 尚未连接 trainer，不能写成 online
grouping。真正非退化的方向有两种：增加同 `(PP,TP)` 的多副本拓扑以提供多个合法 peer，或升级为
per-object holder/requirement recovery，使不同 rank 的部分冗余可参与恢复；两者均尚未实现
（证据：r81 command artifact、`Merak/core/checkpoint/merak_incremental.py:143-164,770-783`）。

理论上，whole-rank 协议的合法边仍须同 PP/TP、异 DP。若每个 `(PP,TP)` shard 在两 host 各有
`D/2` 个 DP replica，则每个 shard 的跨 host perfect matching 有 `(D/2)!` 种，总数为
`N=((D/2)!)^(PP*TP)`：当前两机 16-rank 的 `DP2` 只有唯一解，`DP4` 才有多解。但同步 DP 的
可替换 full state 与成本理论上等价，故 DP4 只应作为小型可证伪边界实验，不能预设收益；要利用
不同 rank 的部分冗余，仍必须实现 per-object holder/requirement recovery，尚未完成。

#### 端到端长尾：从字节到可靠性

端到端须分开计前台训练 step、后台 local-full operation/queue、`PIN -> publish -> SEAL` 的提交
完成尾部和边界 global-seal lag；capture 不等于 commit。r6 的 byte-stream operation 均值为
`10.230 s`，其中序列化/PUT receive+checksum/COMMIT CPU apply 为 `1.099/3.643/4.989 s`
（`runs/two-node-v2/merak-llama1b-peer-materialized-breakdown-r6-20260907T212000Z/analysis.json`）。

共享内存将 r9/r10/r11 operation 降至 `2.419/2.176/1.947 s`（r11 shared copy `0.384 s`），但三者
strict lag 仍为 3；单线程 r12 更慢，r15 即使允许 lag 4 仍在 epoch 6 撞满有界队列。它们只能缩小
开销或容忍积压，不能让服务率超过生产率。GA64 小模型 r30 是正点：20/20 sealed、每个边界 lag=1。

8B 的 r31/r48/r50 都完成五个 sealed epoch，却有 max lag=2；local-full completion 的样本最大值为
`17.724/29.134/41.088 s`，r50 限两并发反而更差。每 run 只有 80 条 rank-local 样本，不能称严格
P99。故论文目标从纯 bytes 变为在 atomic seal、可恢复性与 `lag<=1` 约束下优化 makespan，并在足够
独立 trial 后估计 P99（台账 `TWO_NODE_RUN_LOG.md:902-995,1148-1162,1210-1222`）。

#### 故障注入接口和边界

我们的可执行 fault-injection 入口是 `tools/run_two_node_merak_1c1s.sh:41,68-73` 的
`FAILED_SERVICE_NODE_ID`：它只接受两端 service 之一，并要求 sealed restore。训练后脚本在
`407-415` kill 该 checkpoint-service process，随后在 `418` 启动 fresh-job restore；存活 service
承接两个逻辑 endpoint（`261-269`），各节点以 `CKPT_MATERIALIZED_RESTORE_SOURCE=own|peer` 选择
恢复路由（`327-346`）。底层分别由
`Merak/core/checkpoint/incremental_runtime.py:488,562` 的
`restore_peer_materialized`/`restore_own_materialized` 执行，
`Merak/core/checkpoint/merak_incremental.py:607-630,899-901` 消费该选项；
`test/core/checkpoint/test_global_commit.py:69-132,202-229` 另注入 writer/evidence/PIN failure，
覆盖 fail-closed 的控制面单测。r29 因而只验证一个节点的 checkpoint-service process 停止后，
16-rank fresh job 能续训；这是早期 node17/node1 的 service-process 测试，不验证物理 host-stack
故障、网络隔离或 durable recovery。

对于 AdaCheck，在我们审阅的公开 commit `cf910be` 和公开 artifact 中，未找到可直接运行的完整
“remote CPU updater + atomic publish + fresh-job fault-injection harness”。论文给出了故障恢复设计和
结果，但这不能推出作者私有系统没有测试；这里仅界定公开可复核的实现边界。

#### Failure-domain-aware 拓展构想

这是待检验的研究问题，不是已有实现或结果。给定 epoch catalog、物理拓扑和预注册失效集合 `F`，
候选边/组 `g` 必须先满足：任意 `f in F` 后，每个重启 rank 都能从失效域外取得同 epoch 的
model/optimizer/runtime 及 `PIN -> publish -> SEAL` evidence。记 `C_i` 为 rank `i` 的编码恢复
对象字节，`R_ij` 为可验证的 pair redundancy；对称时残余传输 `D_ij=C_i+C_j-2R_ij`，方向非对称时
用 `D_i->j+D_j->i`。最小形式为：

```text
G_safe = {g | for all f in F, cover(g, f) = 1}
minimize over g in G_safe:
  alpha * sum_(i,j in g) D_ij
  + beta * max_(i,j in g)[D_ij/bandwidth(i,j) + T_materialize_tail(i,j)]
  + gamma * correlated_fault_risk(g)
```

第一项是同一 AdaCheck-style encoder 下的 pairwise residual transfer，而不是最小化 `R_ij`：冗余越大，
`D_ij` 越小。第二项按方向带宽计入 CPU materialization 和共享资源的最长完成时间，不能用静态逻辑
bytes 代替。较小 `D_ij` 的组仍可能集中 NIC 或 materializer 尾部，故字节和 makespan/tail 是同一
安全集合内的权衡。`G_safe` 是不可用风险权重抵消的硬可行条件；`correlated_fault_risk` 只在合法组间
排序或施加额外预算。共享 NIC、ToR、供电域或同一 service 的多份副本都不应被误算为独立保护，同时
还须满足 CPU/pinned-memory、队列长度和恢复时间预算。

`cover` 不能只标 rank：必须追踪 rank process、GPU、host、rack/ToR 或共同网络路径、checkpoint
service 和 durable storage；同一相关域内的副本不能重复计作保护。两机只有唯一的另一 host：在单
host 或单 service 故障假设下最多验证这一条 peer-recovery 路径，不能比较 host pairing 或相关故障；
当前 r29 实测还更窄，仅为 service process loss。四机后才有不同 host pairing 的选择，并可分层先选
独立 host/service/storage 域、再选 rank/object group；每个 object 的 source、target、路由和 manifest
都须绑定该选择，避免只因跨 host 传输更少而牺牲故障覆盖。

公平比较以 `AdaCheckStyleFixed` 为 baseline，固定 detector/encoder、冻结 catalog、并行 placement、
服务预算、snapshot/seed/order、commit/recovery 协议和 `F`，只改变 `g`。预注册 fault matrix 至少覆盖 rank process/GPU、
单 host、rack/路径、单 service 与 durable-copy 丢失或损坏；每格均以 fresh-job numerical continuation
判定，并记录 sealed epoch、恢复时间、失败域和数据/数值合同。可证伪假设是：在相同恢复覆盖和预算下，
安全优化组同时不劣于 fixed 的 bytes 与 completion makespan/tail；若 fixed 是唯一合法解、收益消失，
或任一 matrix 格无法恢复，即拒绝该假设（研究边界见 `AGENT.md:25,80,140-201,376-386`）。

尚未完成的是：two-iteration stable detector 与 scope breakdown、能恢复任意合法 asymmetric edge 的
per-object online path、非零 fixed-to-exact gap 上的真实训练 overlap、物理 host-stack/durable
fallback fault matrix，以及 200-step 独立 paired trials。完成这些前，当前证据只能支持机制和
测量合同，不支持“rank matching 优于 AdaCheck”或近零在线开销的结论。

## 5. 当前判断

已经完成的不是一个简单离线分组脚本，而是一条可运行的两机 16-rank 原型：真实 optimizer
更新捕获、有界后台发布、CPU 状态物化、原子全局封存和 fresh-job continuation 都已连通。
它证明研究问题在工程上可实现，也给出了严格 cadence、字节记账和恢复正确性的测量方法。

这里的“端到端已连通”仅指真实 Merak LLaMA-like-1B：r46 完成 20-step 1C1S，r45 独立完成
3-step checkpoint 后的数值 continuation；不是同一个 20-step recovery run。四个论文尺度模型、
200-step paired trials 和非零 grouping treatment 均未完成。

grouping 论文贡献也尚未成立。当前在线对称场景中，固定分组已经最优；非对称 catalog 的
23%/71% 收益尚未连接到真实 Merak training。因此目前最有价值的发现是：冗余检测、rank
placement、故障域和后台服务率必须联合考虑，仅在固定坏 placement 上更换配对可能没有空间。

此外，当前内存服务只能覆盖健康 service 下的 worker restart，以及单侧 checkpoint service
进程停止的 smoke test；它尚未覆盖物理主机永久失效或两个内存故障域同时丢失。因而目前不能称为
完整的 reliable and efficient checkpoint system。

## 6. 下一阶段目标

近期优先顺序如下：

1. 重复 r46/r47 配对实验并报告 trial-level 置信区间，而不是把 step 或 rank 当独立样本。
2. 先让 LLaMA-3-8B、GPT dense、Qwen2-MoE 和 DeepSeek-V2-Lite 完成 bounded-lag 的 20-step
   checkpoint 与恢复闭环，再单独扫描可达 checkpoint cadence；strict 1C1S 是最终工作点，不再是
   大模型进入后续实验的前置淘汰条件。
3. 构造真实可执行的非对称状态布局，冻结 world-scope 两两冗余矩阵，并将 exact matching 接入
   per-object online recovery；只有 fixed-to-exact gap 非零时才运行昂贵的 grouping 主对比。
4. 在通过 20-step、容量和恢复 gate 后，运行 200-step paired trials，报告 wall time、global
   checkpoint 数、staleness、总状态大小、跨机 NIC bytes、后台长尾和恢复时间。
5. 增加持久化 full fallback 与真实 host-stack fault matrix，明确单主机与双故障域的恢复点目标
   （Recovery Point Objective, RPO）和恢复时间目标（Recovery Time Objective, RTO）。

所有频率结论都以本地实测为准。AdaCheck 论文在选定配置上报告每 iteration checkpoint，但其公开
artifact 未提供可在本集群干净复现的完整远端 updater/commit/recovery 闭环；因此
`AdaCheckStyleFixed` 只表示我们在同一 detector/encoder/service 下固定 mirror pairing 的受控 baseline，
不能预设或宣称它在本环境已经达到 strict 1C1S。

## 7. 开会时建议强调的三点

1. **问题定义是正确的。** 优化输入是与旧 group 无关的全局对象 catalog 和任意合法 rank 对代价，
   不是复用旧 group 确定后才得到的冗余数。
2. **系统路径已经连通。** r46 在真实两机 16 卡完成 20/20 sealed 1C1S，但该 run 未做恢复；
   独立的 r45 在 3-step checkpoint 后给出 fresh-job 数值 continuation 证据。两项共同构成后续
   只改变 grouping 的实验底座，不能合并描述为一个 20-step recovery run。
3. **核心 hypothesis 仍待验证。** 对称布局得到 0% grouping gain 是必要负对照。下一步必须找到
   真实非对称且可恢复的配置，并证明逻辑字节收益能转化为端到端收益；否则论文应转向
   redundancy、placement 与 failure-domain 联合规划，而不能声称简单 rank matching 优于 AdaCheck。
