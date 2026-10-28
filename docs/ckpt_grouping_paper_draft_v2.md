# 冗余感知的分布式训练检查点分组：两机研究草稿 V2

## 摘要

大模型训练希望频繁保存检查点，以减少故障造成的重复计算。AdaCheck 利用训练状态在 worker
之间以及相邻迭代之间的冗余，将梯度增量发送到远端内存，并声称可以在每个训练迭代保存一次
检查点。我们研究一个与其互补的问题：在恢复语义和编码方式不变时，哪些 worker 应当互为
检查点 peer？固定配对没有利用不同 worker 之间不均匀的状态重叠，可能产生不必要的跨机传输。

本文首先在 frozen catalog 上从与旧分组无关的全局对象关系构造任意跨主机 rank 对的编码后
传输代价，再在两台各含 8 张 GPU 的主机之间求解精确完美匹配，作为 offline oracle。当前
online whole-rank 路径只支持合法的镜像 DP peer；将任意边接入恢复仍需 per-object
holder/requirement recovery。因此“只改变 rank pairing”的公平对比是未来实现后的预注册设计。
端到端 200-step 结果在实验完成前保留为空，不以历史 frozen replay 代替。

## 1. 动机

检查点大小并不等于远端保护流量。假设 rank `i` 和 rank `j` 具有大量相同的恢复对象，那么
对端已有对象可以作为恢复基础，只需传输集合差。若固定配对把 `i` 分给一个状态结构不同的
rank，相同的逻辑检查点会产生更多传输。AdaCheck 已提供快速识别 tensor redundancy 的基础，
但其公开 artifact 按对应 local rank 固定构造组，没有搜索全局配对。

这里有两个容易混淆的边界。第一，优化输入必须是全局 holder/requirement 关系，不能使用旧
group 确定后才统计出的一个 redundancy 数值。第二，两台物理主机只有一种 host pairing，
本研究优化的是 rank，不是 machine。host 级调度需要至少四台机器。

## 2. 问题形式化

令 `E_i` 表示 rank `i` 每个迭代需要保护的编码后对象集合，`w_o` 表示对象的实际编码字节数。
跨主机 pair `(i,j)` 的双向增量为：

```text
C_ij = sum(w_o, o in E_i - E_j) + sum(w_o, o in E_j - E_i).
```

在 frozen catalog 的 offline oracle 中，设二元变量 `x_ij` 表示二者是否配对。每个 rank 恰有
一个 peer，目标是最小化 `sum(x_ij C_ij)`。两侧各有 8 个 rank，我们使用位掩码动态规划枚举
右侧已占用集合，得到全局最优解，并用 8! 穷举检查实现。该结论只保证编码后 application bytes
最小；实际时延还受到 publisher 负载、共享网卡和计算空隙影响，必须由端到端实验测量。

对于未来的 per-object holder/requirement recovery，若 `j` 所在主机保存 `E_j` 和 `E_i-E_j`，
它能够重建 `E_i`。所有对象还必须属于同一 committed epoch，并通过 catalog、plan、encoding 和
checksum 验证。该集合关系证明单 host 副本的静态可恢复性，不替代 fresh-job continued-training
测试，也不是当前 whole-rank live path 的能力。

## 3. 系统设计

系统在两个相邻 iteration 上运行 redundancy detector，并以完整 checksum 确认稳定等价类。
在训练过程中，梯度完成同步后，checkpoint hook 保存原始 BF16 梯度以及本 step 精确的
unscale/clip 标量；恢复端按与在线 FP32 optimizer 相同的“上转换、乘标量”顺序构造真实更新量。
这避免把已经缩放的 FP32 梯度再次舍入到 BF16，梯度主体仍为每元素两字节。hook 将记录复制到
有界 pinned-memory buffer。后台 publisher 按 epoch 顺序追加增量日志；恢复时从周期性
base checkpoint 重放经过校验的日志。当前实现还可以在独立服务中用中央处理器 AdamW 将更新
折叠为有界的当前状态。只有 16 个 rank 的本地完整更新与跨故障域 runtime delta 均提交，且
peer、epoch、catalog 和 checksum 证据互相一致时，系统才原子发布全局 committed control。
训练线程不等待远端 ACK；队列满或 step boundary 的未提交 epoch 超过 1 时，系统不丢弃或
合并记录，而是将该 run 判为不满足 one-checkpoint-per-step。

AdaCheck 论文描述了远端中央处理器（Central Processing Unit, CPU）optimizer 更新，但公开源码
没有可执行的完整路径。本文不声称复现 AdaCheck 端到端系统，也不把我们实现的绝对性能归因于
AdaCheck；在 per-object recovery 接入后，固定分组与优化分组必须使用完全相同的增量日志、传输、
提交和恢复实现，唯一自变量才是 rank pairing。后台 CPU 物化是恢复延迟与容量机制，不自动带来
足够吞吐。当前本机共享内存路径仍需一次 pinned-to-shared copy、semantic digest 和 CPU optimizer
apply，三者必须独立计时。

通信可观测性是独立模块，而不是 detector、encoder 或 grouping solver 的内部功能。调用者只向
adapter 提交训练计算、capture、poll、drain 和远端边区间；每个区间的首尾时间戳来自同一进程内
monotonic clock domain，跨 rank 聚合仅比较已完成的 duration，不直接比较未同步的跨机绝对时间。模块不读取
GPU 时钟、不执行 CUDA synchronize、不插入 collective，也不影响传输和放置决策。它输出区间
并集、idle/slack 以及 P50/P95/P99/max，用来区分后台传输时间和训练关键路径暴露时间。当前它是
公平评测基础设施；只有未来 slack-aware 调度器在相同编码器和分组下通过独立 ablation 获益时，
才把“profile-guided scheduling”列为额外 system-design contribution。

为了让跨主机重叠具有真实含义，我们显式使用 DP-first rank mesh，使两个 Data Parallelism
副本处在不同物理主机。原 PP-first 布局作为低机会对照。当前 `DP2` whole-rank 协议下，每个
rank 只有一个同 PP/TP、异 DP 的合法 peer；未来可执行的 fixed/exact 对比才须共享其余代码和
资源。拟议系统每 50 步异步写入一次完整持久化检查点，用于两个内存故障域同时丢失的场景；该
durable fallback 尚未实现，因此当前结果不构成完整可靠检查点系统。

## 4. 评测设计

评测使用 LLaMA-3-8B、GPT dense 6.3B、Qwen2-MoE 14.3B 和
DeepSeek-V2-Lite 15.7B 的公开结构，均随机初始化并使用确定性合成 token。每个配置先通过
trace、显存、10+5 数值恢复和 20-step bounded-lag preflight，再寻找可持续频率并逐步收紧到
1C1S，最后才运行 200 optimizer steps。主对比为
NoCheckpoint、AdaCheck-style fixed grouping 和 redundancy-aware exact matching。正式性能
数字来自至少五个独立 paired trials，并轮换执行顺序；随机分组敏感性才使用预注册 permutation
seed，精确匹配本身不使用 seed。

主要结果报告 200-step wall time、200/200 committed checkpoint、每小时 committed 数量、
编码后字节、NIC 字节、queue lag、P95 exposed stall 和恢复时间。统计单位是独立 trial，
不是同一 trial 内的 rank 或 step。故障实验在独立 run 中停止一侧 host stack，再恢复原
16-rank allocation 并继续五步。

## 5. 当前结论与缺口

DP-first 的 0.941B 对称 frozen catalog 中，固定 pairing 已等于精确最优；随机错配会显著增加
保护流量，但这只说明错误分组有代价，不能证明优化优于 AdaCheck 风格固定配对。当前实现已具备
pre-update capture、有界 publisher、CPU materializer 和 typed global commit。r7 验证了健康
service 下 16-rank fresh-worker loadability；r9 将本机 update operation 从平均 10.287 s 降到
2.419 s，但最大提交 lag 仍为 3，严格 1C1S 失败且没有运行 restore。

因此当前最强结论仍是机制与负面 breakdown，而不是可靠高效系统。论文 idea 是否成立取决于
三个尚未闭合的问题：真实 workload 是否存在非零 fixed-to-exact edge gap；该 gap 能否通过在线
per-object recovery 转化为更少跨故障域流量；以及在持久化 fallback、host-fault recovery 和
数值 oracle 都成立时，materializer 服务率能否达到每 step 一个全局 committed checkpoint。

## 6. 当前更新：分组语义与恢复证据

在 `PP4/TP2/DP2` 的 DP-first 部署中，节点 17 承载 ranks 0--7，节点 8 承载 ranks 8--15。
当前 whole-rank 恢复要求 pair 跨 host、保持 PP/TP 坐标且仅改变 DP 坐标，故 `(0,8)..(7,15)`
是唯一合法 matching，而不只是对称 catalog 下的受约束最优。该 frozen catalog 也使 fixed 与
exact 同值，因而它提供必要的零机会对照，却不能单独证明冗余感知分组优于 AdaCheck-style fixed
grouping。

一般地，若每个 `(PP,TP)` shard 在两台主机各有 `D/2` 个 DP 副本，合法 whole-rank matching
数为 `((D/2)!)^(PP*TP)`：当前 `DP2` 为一；`DP4` 虽有多解，同步 DP 使可替换 full state 及成本
理论等价，故只能作为小型可证伪边界实验，而非预期 grouping gain。利用不同 rank 的部分冗余仍
需要尚未实现的 per-object holder/requirement recovery。

最新 mini-test r45 证明该在线路径能够在每个 iteration 提交一个封存恢复点：三个训练步骤的
global commit 均连续，最大 local/global lag 为 1。每个 reciprocal peer 的 epoch checksum 一致后，
系统才固定材料并原子发布全局 control。随后 16 个 fresh workers 恢复 epoch 3，继续一步所得 loss
与 uninterrupted reference 同为 `9.6475`。全部 rank 的 CPU materialized state 通过预注册数值
合同；BF16 model 最大每张量相对 Linf 为 `0.0025907`，低于一个 BF16 epsilon `0.0078125`。

这个结果将当前论证从“可加载”推进到“有数值依据的可继续训练”，但仍不构成完整论文结论。
它只覆盖两个 checkpoint service 均存活的 worker restart，未覆盖物理主机永久失效或持久化
fallback；三步也不足以说明长期 service rate 和训练开销。r81 在 `lag=2` 诊断设置下尝试 GPT
dense 6.33B 的 200-step 持续态，step 15 后 rank 8 对 epoch 16 enqueue 触发
`CheckpointBackpressureError`；launcher 为 local `143`、remote `1`，global controls 仅 `0..11`，
且无终态 metrics 或 digests。它是服务率不足的持续态负结果，不构成 200-step 性能、tail 或
grouping 结论。严格 cadence、持久化 fallback、host-fault recovery 与非零在线优化空间闭合前，
论文不会报告 grouping speedup。

最新 r101 首次闭合 live SemanticFixed 的 GPT-6.33B 10+5 baseline：16-rank
`PP4/TP2/DP2` 下十个 global update 连续提交，16/16 materialized model/optimizer
满足预注册数值合同，新进程恢复 epoch 10 后的 step 11--15 loss 与 uninterrupted
reference 完全一致。十个 epoch 的全 rank logical/payload/wire 为
`269.960/269.964/269.970 GB`，epoch bottleneck edge mean/max 为
`26.640/34.815 s`。但训练窗口只观察到 6 个完成 checkpoint，global lag 最大为 4，
steady step mean 为 `19.075 s`，相对 NoCheckpoint `12.736 s` 仍有明显暴露开销；
final validation/drain 更达到 `3040.828 s`。因此 r101 是可恢复的 bounded-lag fixed
baseline，不是 1C1S、non-blocking 或 grouping gain 结果。
