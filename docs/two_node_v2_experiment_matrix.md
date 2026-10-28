# 两机 V2.1 实验与 AdaCheck 对照矩阵

更新时间：2026-09-08。本文档是持续更新的实验索引，不替代原始 run 中的命令、日志和 JSON。

## 对照边界

| 条目 | 分组 | 每步状态 | 数据面 | 当前可验证结论 |
|---|---|---|---|---|
| AdaCheck 论文 | 按 model parallelism 构造相似计算图 worker group | gradient incremental checkpoint | 远端 CPU optimizer，与训练 overlap | 论文报告 one checkpoint per iteration；公开源码不足以独立复现完整在线路径 |
| AdaCheck 公开 artifact | 对应 local rank 固定配对 | differential/redundancy 相关代码 | 可观察到固定 group 和 detector，但远端 CPU updater 路径不完整 | 只用于机制 breakdown 和固定分组依据，不能冒充端到端复现 |
| `FullRemoteIncremental` | 固定跨机 rank pair | base + 每步完整同步后梯度 | 每个 rank 将完整 update 发送到 peer host service | 验证本研究有序日志和恢复；也是“不利用 peer redundancy”的系统对照 |
| `PeerDeltaIncremental` | 当前仅同 PP/TP 的镜像 DP peer | local full gradient update + remote failed-rank runtime delta | full update 由本机 service 消费，跨 host 只传 runtime；尚不支持任意非对称 `E_i/E_j` live recovery | 有序恢复原型；offline matching 尚未接入在线恢复 |
| `MaterializedPeerIncremental` | 当前仅同 PP/TP 的镜像 DP peer | 远端 CPU AdamW 将 base+update 折叠为当前状态 | bounded current state + remote runtime delta；恢复时按需序列化 | r7 已验证 global manifest 下 fresh-job loadability；service rate 与 host-fault gate 未通过 |

## 已完成 gate

| 日期 | 范围 | 配置 | 连续提交 | 最大远端 lag | 跨域 update | fresh-job 恢复 | 结论 |
|---|---|---|---:|---:|---:|---:|---|
| 2026-09-07 | 同机 2 GPU，full update | width 1024，3 steps | 3/3 | 3 steps | 约 25 MB/rank 逻辑 remote-service payload | 通过，参数/loss 逐位一致 | 恢复正确，但不满足严格 1C1S lag gate；run `incremental-local-smoke-nonblocking-20260907T150638Z` |
| 2026-09-07 | 同机 2 GPU，peer delta | width 1024，3 steps | 3/3 | 2 steps | 约 40 KB/rank 逻辑 remote-service payload | 通过，参数/loss 逐位一致 | 逻辑 payload 约降 99.84%；不是跨 host NIC 结果；run `peer-incremental-local-smoke-20260907T152001Z` |
| 2026-09-07 | 两机 16 GPU，NoCheckpoint | width 1024，warmup 5，20 steps，compute repeats 64 | 不适用 | 不适用 | 0 | 不适用 | 两轮均完成；max-rank loop 为 756.54/746.10 ms，仅作 harness compute control |
| 2026-09-07 | 两机 16 GPU，full update | 同上，bounded inflight=4 | 否，第 6 个 epoch backpressure | 至少 5，未形成完整 summary | 未完成 | 未运行 | 不满足 1C1S；不得用于性能对比 |
| 2026-09-07 | 两机 16 GPU，peer delta | 同上，bounded inflight=4 | 否，第 7 个 epoch backpressure | 至少 5，未形成完整 summary | 未完成 | 未运行 | remote delta 不能消除 local full updater 瓶颈；不得用于性能对比 |
| 2026-09-07 | 两机 16 GPU，full update service-rate gate | stable repeated-forward mean，warmup 5，20 steps，compute repeats 512 | 20/20 | 1 | 2,691,660,020 bytes | 通过；边界 model/AdamW/scheduler bytes 一致 | max-rank loop 5,126.96 ms，较 NoCheckpoint +16.78%；restore max 1,341.83 ms |
| 2026-09-07 | 两机 16 GPU，peer delta service-rate gate | 与上一行相同 | 20/20 | 1 | 4,317,360 bytes | 通过；边界 bytes 一致，继续 5 steps 数值一致 | 跨 host logical payload -99.84%，但 max-rank loop 5,355.21 ms，较 NoCheckpoint +21.98%、较 full +4.45% |
| 2026-09-07 | 两机 16 GPU，peer snapshot A/B | 同一 stable-compute gate；移除第二次 CPU snapshot | 20/20 | 1 | 4,314,680 bytes | 通过；边界 bytes 一致，继续 5 steps 数值一致 | peer mean capture 最大值 2.25 ms；max-rank loop 4,943.19 ms，较同 trial NoCheckpoint +16.44%、较 full -3.80%；单次短 trial，不是论文统计量 |
| 2026-09-07 | 真实 Merak LLaMA-like-1B，peer r1 | PP4/TP2/DP2，seq128，2 steps | 最终完成，未形成完整全-rank summary | 2 | local full + remote runtime | 未进入正式恢复 | 训练完成但 strict gate 失败；run `merak-llama1b-peer-smoke-r1-20260907T193000Z` |
| 2026-09-07 | 真实 Merak LLaMA-like-1B，peer r2 | PP4/TP2/DP2，seq1024，3 steps | 3/3 最终 drain | 3 | 每 rank local update 404--667 MB；remote runtime 约 13.7 KB/step | 未运行 | iteration 约 1.0--2.4 s，local commit 约 3--14 s，服务率不足；run `merak-llama1b-peer-seq1024-r2-20260907T195000Z` |
| 2026-09-07 | 真实 Merak LLaMA-like-1B，peer recv-into r3 | 与 r2 相同，service 预分配 socket receive buffer | 3/3 最终 drain | 3 | 与 r2 同量级 | 未运行 | transport 仍约 2.2--10.2 s；单独消除 receive copy 不足；run `merak-llama1b-peer-recvinto-r3-20260907T201000Z` |
| 2026-09-07 | 真实 Merak LLaMA-like-1B，CPU materializer r4 | PP4/TP2/DP2，seq1024，3 steps，默认 52 CPU threads | 3/3 rank ACK，非 global manifest | 3 | 25.723 GB local-full update；0.657 MB remote runtime；current state 15.005 GB/host | 未运行 | mean/max local commit 13.291/25.096 s，max queue 40.242 s；有界状态成功、strict service-rate 失败；run `merak-llama1b-peer-materialized-r4-20260907T203000Z` |
| 2026-09-07 | 真实 Merak LLaMA-like-1B，CPU materializer r5 | 与 r4 相同，service intra-op threads=2，含原子/幂等协议修正 | 3/3 rank ACK，非 global manifest | 3 | 与 r4 相同；恢复容量按 2x logical reservation | 未运行 | mean commit 10.003 s（r4 比值 -24.7%），max queue 34.608 s（-14.0%），但 max commit 28.714 s；strict gate 仍失败 |
| 2026-09-07 | 真实 Merak LLaMA-like-1B，materializer breakdown r6 | 与 r5 相同，只新增 RPC phase timing | 3/3 rank ACK，非 global manifest | 3 | 与 r5 相同 | 未运行 | mean operation 10.230 s：serialize 1.099 s、PUT server 3.643 s、COMMIT server 4.989 s；max queue 29.718 s；用于 data-plane 决策，不是独立性能 trial |
| 2026-09-07 | 真实 Merak LLaMA-like-1B，global restore r7 | 与 r6 相同；schema-v1 global control；诊断性 `STRICT_GATE=0` | 3/3，global epochs 0--3 | local/global 均为 3 | 25.723 GB local update；0.662 MB remote delta | 16 fresh ranks 从 epoch 3 加载并完成 step 4 | launcher 均成功但最终 strict validity=false；只证明健康 service 下 loadability，不证明 M2/M4 |
| 2026-09-07 | 真实 Merak LLaMA-like-1B，NoCheckpoint r8 | 与 r7 同 seed/并行/seq1024/GA8，3 steps | 不适用 | 不适用 | 0 | 不适用 | 三步 loss 与 r7 一致；rank-0 稳态 0.671 s vs r7 1.024 s（单次 diagnostic 约 +52.5%） |
| 2026-09-07 | 真实 Merak LLaMA-like-1B，shared-tensor r9 | 与 r7 相同；schema-v2 global control；strict；本机命名共享内存 | 3/3，global epochs 0--3 | local/global 均为 3 | local update TCP payload/wire 均为 0；累计 shared handoff 25.723 GB；remote delta 0.661 MB | strict 失败，未运行 | mean local operation 2.419 s、queue 1.378 s，较 r7 分别 -76.5%/-87.4%，仍慢于约 1.0 s steady step；data-plane breakdown 负结果，不是 zero-network/1C1S 结果 |
| 2026-09-08 | 真实 Merak LLaMA-like-1B，no-grad-clone r10 | 与 r9 相同；CPU AdamW 借用已校验 gradient | 3/3，global epochs 0--3 | local/global 均为 3 | shared handoff 25.723 GB；remote delta 0.662 MB | strict 失败，未运行 | mean operation/apply/queue 2.176/0.731/1.087 s；较 r9 分别 -10.05%/-24.29%/-21.11%，仍未过 rate gate |
| 2026-09-08 | 真实 Merak LLaMA-like-1B，shared-slot reuse r11 | 与 r10 相同；service ACK 后复用共享 storage | 3/3，global epochs 0--3 | local/global 均为 3 | shared handoff 25.723 GB；remote delta 0.661 MB | strict 失败，未运行 | mean shared copy 0.384 s、operation 1.947 s、queue 0.872 s；优化有效但仍不是 1C1S |
| 2026-09-08 | 真实 Merak LLaMA-like-1B，service threads=1 r12 | 与 r11 相同；仅 2 threads 改为 1 | 3/3，global epochs 0--3 | local/global 均为 3 | shared handoff 25.723 GB；remote delta 0.661 MB | strict 失败，未运行 | mean operation/apply 2.622/1.427 s、max queue 7.357 s；明显劣于 r11，保留 2 threads |
| 2026-09-08 | LLaMA-3-8B 架构 base preflight r13 | PP4/TP2/DP2，seq1024，GA8，3 steps，400,000 MiB/service，默认 60 s RPC timeout | 未进入 update | 不适用 | base rank payload 约 12--19 GB | 未运行 | 八 rank 并发上传初始 base 时 timeout；是尺度 rejection，不是 cadence 结果 |
| 2026-09-08 | LLaMA-3-8B 架构 strict preflight r14 | 与 r13 相同；显式 RPC timeout 600 s | 3/3，global epochs 0--3 | local/global 均为 3 | base 127.137 GB/host；final update 36.325 GB/host；shared handoff 217.949 GB | strict 失败，未运行 | steady step 约 5.22 s；mean operation/copy/digest/apply 18.107/2.135/6.774/9.123 s；更长 iteration 仍未隐藏 materializer |
| 2026-09-08 | LLaMA-like-1B bounded-staleness r15 | r11 path，20 steps，strict max-lag=4，measurement only | 训练 5 steps 后 epoch 6 backpressure | 无有效 final summary | 不适用 | 明确关闭 | 放宽 staleness 不提高 service rate；负结果 |
| 2026-09-08 | LLaMA-like-1B GA32 r17 / NoCheckpoint r18 | PP4/TP2/DP2，seq1024，GA32，seed 20260908，20 steps | 20/20，global epochs 0--20 | local/global 2/3；lag-three gate 通过 | shared handoff 171.485 GB；remote delta 4.404 MB | 明确关闭 | steady steps 6--20：2.1897 vs 2.0095 s，单次 +8.96%；不是 strict lag-one |
| 2026-09-08 | LLaMA-like-1B sync-control r19 | 与 r17 相同；仅 tiny global JSON 改为同步发布；strict lag-one | 20/20，global epochs 0--20 | local/global 2/2，strict 失败 | shared handoff 171.485 GB；remote delta 4.407 MB | 明确关闭 | 去除 control-plane 假滞后，暴露真实 slowest-rank rate |
| 2026-09-08 | LLaMA-like-1B GA64 r20 / NoCheckpoint r21 | 与 r19 相同；GA64；20 steps | 20/20，global epochs 0--20 | 每个 boundary local/global 均为 1，strict 通过 | shared handoff 171.485 GB；remote delta 4.407 MB | 尚未运行，等待 sealed protocol | steady steps 6--20：3.9363 vs 3.8245 s，单次 +2.92%；首个真实 Merak strict cadence 正点 |
| 2026-09-08 | LLaMA-3-8B GA64 strict r22 / NoCheckpoint r23 | r14 topology/service；GA64；5 steps | 两者均完成 5/5；r22 global epochs 0--5 | r22 local/global 为 1,2,2,2,3，strict 失败 | shared handoff 363.248 GB；remote delta 1.102 MB | 明确关闭 | step-2--5 16.8238 vs 14.6223 s，单次 +15.06%；mean/max operation 15.878/32.774 s |
| 2026-09-08 | LLaMA-3-8B CPU4 strict r24 | 与 r22 相同，仅 service threads 2->4 | 5/5，global epochs 0--5 | local/global `1,2,2,2,2`，strict 失败 | shared handoff 363.248 GB；remote delta 1.102 MB | 明确关闭 | mean operation/queue 14.681/2.035 s，max operation 33.084 s；部分改善但 tail 仍不满足 1C1S |
| 2026-09-08 | LLaMA-like-1B sealed healthy restore r27 | `PP4/TP2/DP2`、seq1024、GA64、3 steps、PIN/publish/SEAL | 3/3，sealed global epochs 0--3 | 每个 boundary local/global 均为 1 | shared handoff 25.723 GB；remote delta 0.661 MB | 16 fresh ranks 从健康 services 恢复 epoch 3 并完成 step 4 | 首个 sealed live-path 正点；仅 liveness，尚非 uninterrupted numerical oracle |
| 2026-09-08 | LLaMA-like-1B node17 service-loss r29 | 与 r27 相同；训练后终止 node17 checkpoint service | 3/3，sealed global epochs 0--3 | 每个 boundary local/global 均为 1 | 恢复仅访问 node1；失效侧 peer、存活侧 own | 16 fresh ranks 恢复并完成 step 4 | 单节点 checkpoint-service loss 通过；不等于永久 host loss/M3 durability |
| 2026-09-08 | LLaMA-like-1B sealed 20-step r30 | 与 r27 相同；20 steps；明确关闭 restore | 20/20，sealed global epochs 0--20 | 20 个 boundary local/global 均为 1 | shared handoff 171.486 GB；remote delta 4.407 MB | 明确关闭 | bounded slots 超过 queue depth 后仍持续前进；相对旧 r21 steady +5.99%，须补当前代码 paired control |
| 2026-09-08 | LLaMA-3-8B sealed r31 | 与 r22 相同；per-epoch peer equivalence；5 steps | 5/5，sealed epochs 0--5 | `1,1,1,1,2`，strict 失败 | shared handoff 363.248 GB；remote payload 1.102 MB | 明确关闭 | mean/max local operation 10.793/17.724 s；接近但尚未达到严格 1C1S |
| 2026-09-08 | LLaMA-like-1B ULP calibration r38 | GA64，3 steps，sealed，CPU numerical replay | 3/3 | strict 未作为接受门槛 | 与 r45 同量级 | fresh restore 完成 | 一 ULP 门槛失败但仅 2,000/2.144B BF16 元素不同；保留为数值合同负面对照 |
| 2026-09-08 | LLaMA-like-1B GPU replay r42 / NoCheckpoint r41 | constant LR，专用 DataLoader RNG；3+1 对 4 steps | 3/3 | full-remote 最大 lag 2 | full remote update | 自身 checkpoint boundary 逐位一致；step-4 loss 均为 9.6475 | 定位并修复 fresh iterator 消耗全局 RNG；full-remote 不是目标数据面 |
| 2026-09-08 | LLaMA-like-1B sealed numerical oracle r45 | `PP4/TP2/DP2`、GA64、3+1 steps，pairs `(0,8)..(7,15)` | 3/3，sealed epochs 0--3 | 每个 boundary local/global 均为 1 | shared handoff 25.723 GB；remote wire 0.774 MB | 16-rank checkpoint state 全部数值通过；step-4 loss 9.6475 与 r41 一致 | 当前最强 mini 正点；BF16 relative-Linf max 0.002591 < 0.0078125，非 bitwise、非物理 host-loss |
| 2026-09-08 | LLaMA-like-1B current-source r46/r47 | `PP4/TP2/DP2`、GA64、20 steps，sealed vs NoCheckpoint | r46 20/20，sealed epochs 0--20；r47 完成 | r46 每个 boundary local/global 均为 1 | shared handoff 171.486 GB；remote wire 5.164 MB | 本配对明确关闭恢复 | rank-0 steps 6--20：3.9791 vs 3.7775 s，单次 overhead +5.34%；需独立重复与 CI |
| 2026-09-08 | LLaMA-3-8B GA96 r48 | `PP4/TP2/DP2`、GA96、5 steps、sealed | 5/5，sealed epochs 0--5 | `1,1,1,1,2`，strict 失败 | shared handoff 363.248 GB；remote wire 1.292 MB | 明确关闭 | local-full mean/max commit 13.013/29.134 s；更长计算窗口仍未消除 tail |
| 2026-09-08 | LLaMA-3-8B r49 | 与 r48 相同，拟测 shared stage cap=2 | 未进入首步 | 不适用 | 无 checkpoint metrics | 未运行 | node1 外来进程占用约 36+8 GiB 后 optimizer init OOM；明确 `EXCLUDED` |
| 2026-09-08 | LLaMA-3-8B stage-cap r50 | 与 r48 相同，仅 shared `SAVE_UPDATE` concurrency cap=2 | 5/5，sealed epochs 0--5 | `1,1,2,2,2`，strict 失败 | shared handoff 363.248 GB；remote wire 1.291 MB | 明确关闭 | mean/max commit 20.240/41.088 s，mean queue/transport 2.347/16.677 s；旋钮无效并已删除 |
| 2026-09-08 | LLaMA-like-1B r51 full-state static catalog | node17/node8，16 rank，PP4/TP2/DP2，PP-first，world-scope，epoch 1/2 stable detector capture | 16/16 catalog reconstruction | 不适用 | fixed/exact pair-local logical tensor bytes 为 7.911/5.813 GB | 未运行 | 26.51% 仅是 frozen full-state static opportunity upper bound；非 AdaCheck online gain、非 wire/P99/训练时间或恢复结果 |
| 2026-09-08 | LLaMA-like-1B r51 adjacent-epoch activity diagnostic | 同一 r51 的 epoch 1->2 checksum filter | 不适用 | 不适用 | 仅 8/3,576 changed requirements、40,448 B；fixed=exact=40,448 B，0% gap | 未运行 | 仅 `torch_rng_state` 变化；非 runtime gate、非 AdaCheck gradient/wire cost，也不证明完整 checkpoint 恢复 |
| 2026-09-08 | LLaMA-like-1B BF16 r52--r54 | r51 后的 BF16 capture diagnostics | 三者均在 catalog/payload 前失败 | 不适用 | 无 catalog、payload 或 opportunity observation | 未运行 | r52 name/value BF16 NumPy，r53 independent hashing BF16 NumPy，r54 Adam 0-D step tensor byte-view；均为 `EXCLUDED` 诊断，不进入科学结果 |
| 2026-09-08 | LLaMA-like-1B BF16 r55 world catalog | node17/node8，16 rank，PP4/TP2/DP2，PP-first `pp-dp`，2 steps | 两 epoch 均 16/16 frozen reconstruction；每 epoch full/reduced 为 34.297/17.148 GB | 不适用 | static 与 temporal 均 fixed=profile-exact，0% gap | 未运行 | detector/frozen catalog 正点；仍是 full-state/activity diagnostic，非 AdaCheck online payload、wire/P99、1C1S 或恢复结果；raw source bins 已按 `RAW_TENSOR_CLEANUP.md` 删除，frozen artifacts 保留 |
| 2026-09-08 | LLaMA-like-1B BF16 r56 update catalog | PP-first `PP4/TP2/DP2`，`full_remote`，2 steps，`UPDATE_CATALOG=1` | 两个 train launcher 均为 0；epochs 1/2 各有 16 份 pre-optimizer catalog | local/global max lag=2，配置为 1，root `FAILED` | fixed/frozen/hindsight 各为 8,574,238,720 B，overlap 0，0% | 未运行 | 仅 previous-epoch offline diagnostic；同步 hash 扰动 cadence，`adacheck_payload_faithful=false`，unrestricted plan 不兼容 current `peer_delta`，不得计入 1C1S/P99/runtime/recovery |
| 2026-09-08 | Qwen2-MoE r57 no-ZeRO preflight | node17/node8，BF16，16 rank，`PP8/TP1/DP2`、`dp-pp`、seq64、1 step、mode off；public config 14.316B/constructed metadata 13.825B | FX trace/layer split 成功；AdamW `exp_avg` 初始化时约 44.54 GiB OOM，step 1 前失败 | 不适用 | 无 checkpoint/catalog/payload | 未运行 | `EXCLUDED`；仅定位 no-ZeRO 显存边界，不计 runtime、1C1S、P99、checkpoint、grouping 或 EP |
| 2026-09-08 | Qwen2-MoE r58 ZeRO-1 preflight | 与 r57 相同，`ZERO_STAGE=1`，mode off，1 step | `COMPLETE`；launcher `0/0`；loss 12.5369，step 8.169 s | 不适用 | 无 checkpoint/catalog/payload | 未运行 | 仅 sparse architecture + ZeRO-1 兼容性；无 EP process group/expert placement，不能计入 checkpoint、grouping、1C1S、recovery 或性能结论 |
| 2026-09-08 | Qwen2-MoE r59 world metadata capture | node17/node8，BF16，`PP8/TP1/DP2`、`dp-pp`、ZeRO-1、seq64；2 actual train/capture steps | launcher `0/0`；两 epoch 各 16 rank catalog；32 个 per-rank object-catalog records，约 97 MB，0 `.bin` | 不适用 | full/reduced logical metadata 295,639,187,232/270,156,295,968 B；fixed=profile-exact=255,954,378,528 B，0%；random 284,985,777,952 B | 未运行 | AdaCheck world/model-role stable metadata evidence；`recoverable_payload_present=false`、无 bundle；大 MoE fixed-mirror static negative control，非 runtime/1C1S/P99/recovery/EP/e2e checkpoint |
| 2026-09-08 | Qwen2-MoE r60 NoCheckpoint baseline | node17/node8，BF16，`PP8/TP1/DP2`、`dp-pp`、ZeRO-1、seq64、GA1、constant LR、seed20260908、mode off，20 steps | `COMPLETE`；launcher `0/0`；20/20 | 不适用 | node-0 step 1 warmup 8.100 s；steps 2--20 n=19 arithmetic mean 4.660157895 s，range 4.600--4.697 s | 未运行 | 同配置 future checkpoint treatment 的 NoCheckpoint baseline；无 checkpoint/service/catalog/detector/grouping/recovery/EP，非 1C1S/AdaCheck/e2e 结果 |
| 2026-09-08 | GPT dense 6.33B r61 FX preflight | BF16，`PP4/TP2/DP2`、`dp-pp`、seq128、mode off | GPT2Attention symbolic `q_len` control-flow `TraceError`；0 step | 不适用 | 无 checkpoint/catalog/payload | 未运行 | `EXCLUDED`；之后仅注册 GPT2Attention FX leaf，focused workload suite 12 passed；不是 GPT runtime 或模型结果 |
| 2026-09-08 | GPT dense 6.33B r62 leaf/split preflight | 与 r61 相同，含 GPT2Attention FX leaf | trace/layer split 成功，PP4 stage params 1.263--1.359B；stage-0 forward shape `[1,128,4096]`/262144 失败，0 step | 不适用 | 无 checkpoint/catalog/payload | 未运行 | `EXCLUDED`；疑似 TP2 attention metadata/mapping，尚在审计；不是 GPT runtime、模型结果或任何 checkpoint/grouping 证据 |
| 2026-09-08 | GPT dense 6.33B r63 eager one-step preflight | BF16，`PP4/TP2/DP2`、`dp-pp`、seq128、mode off；eager-only | `COMPLETE`；launcher `0/0`，16 terminal digests；loss 11.7812，1.792 s | 不适用 | 无 checkpoint/catalog/payload | 未运行 | `EXCLUDED` 为完整 TP2 性能结果：审计发现 `n_inner=14336` MLP 未 TP shard；仅确认 attention root cause |
| 2026-09-08 | GPT dense 6.33B r64 incomplete-TP2 five-step diagnostic | 与 r63 相同，5 requested steps | 5 个 train steps 已完成；因发现 MLP mapping 不完整，在 terminal digest 期间主动终止，root `FAILED` | 不适用 | 无 checkpoint/catalog/payload；无 completed final-digest set | 未运行 | `EXCLUDED`；不能将五个 logged step 写为完整 TP2、吞吐或模型结果 |
| 2026-09-08 | GPT dense 6.33B r65 full-TP2 one-step preflight | BF16，`PP4/TP2/DP2`、`dp-pp`、seq128、mode off；MLP `(2,7,col)/(7,2,row)` mapping | `COMPLETE`；launcher `0/0`，16 terminal digests；loss 11.5938，1.739 s；PP4 stage params 0.831--0.861B | 不适用 | 无 checkpoint/catalog/payload | 未运行 | 验证 MLP TP shard 和 one-step forward/backward；仅 correctness preflight，非 NoCheckpoint baseline、1C1S、AdaCheck、grouping 或 recovery 结果 |
| 2026-09-08 | GPT dense 6.33B r66 full-TP2 NoCheckpoint baseline | node17/node8，BF16，`PP4/TP2/DP2`、`dp-pp`、seq128、GA1、constant LR、seed20260908、mode off，5 steps | `COMPLETE`；launcher `0/0`，5/5、16 terminal digests | 不适用 | step 1 warmup 1.723 s；steps 2--5 n=4 mean 0.4775 s，range 0.467--0.505 s | 未运行 | 完整 TP2 的短 NoCheckpoint reference；无 checkpoint/service/catalog/detector/grouping/recovery，非 1C1S/AdaCheck/e2e 结果 |
| 2026-09-08 | GPT dense 6.33B r67 fixed sealed tail diagnostic | node17/node8，BF16，`PP4/TP2/DP2`、`dp-pp`、seq128、GA64、`peer_delta`、fixed mirror、`MATERIALIZE=SEALED=1`、shared-tensor、3 steps | global epochs 0--3 sealed、all-rank contiguous；但 launcher `1/143`，root `FAILED` | boundary lag `1,2,2`，max global/rank 2 > configured 1 | local-full 48 updates max commit/queue/transport/serialize 23.561/18.715/20.505/3.767 s；remote 48 deltas payload/wire 661,456/774,830 B，max commit 534.165 ms | 未运行 | strict 1C1S fail；无 restore/oracle；不是 pairing A/B、P99、recovery 或 e2e 结果，也不能与 GA1 r66 作 overhead 比较 |
| 2026-09-08 | GPT dense 6.33B r68 seq1024 GA64 NoCheckpoint | node17/node8，BF16，`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64、constant LR、seed20260908、mode off，3 steps；固定执行顺序中先运行 | `COMPLETE`；launcher `0/0`，3/3、16 terminal digests | 不适用 | rank-0 11.496/10.294/10.290 s；steps 2--3 n=2 mean 10.292 s | 未运行 | r69 的 ordered NoCheckpoint counterpart；无 checkpoint/service/detector/grouping/recovery，非 1C1S 结果 |
| 2026-09-08 | GPT dense 6.33B r69 seq1024 GA64 fixed sealed | 相同训练负载，固定执行顺序中后运行；`peer_delta` fixed mirror、`MATERIALIZE=SEALED=1`、shared-tensor、strict lag=1，3 steps | all-rank contiguous、global control epochs 0--3、3 epochs sealed；launcher `1/143`，root `FAILED`，无 final digest | per-rank lag `1,1,1`；global lag `1,1,2`，max global 2 > 1 | local-full 48 updates: shared 161,975,745,024 B; max capture/commit/queue/transport/serialize 2.049/16.499/4.223/13.102/3.570 s; remote 48: payload/wire 661,840/775,212 B, max commit 690.948 ms | 未运行 | strict 1C1S fail；无 restore/oracle；不进入 P99/recovery/e2e/grouping A/B；r68 vs r69 仅 n=2、固定顺序，不能声称 overhead |
| 2026-09-08 | GPT dense 6.33B r70 seq1024 GA64 relaxed-lag two diagnostic | node17/node8，BF16，`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64、constant LR、seed20260908；fixed mirror `peer_delta`、`MATERIALIZE=SEALED=1`、shared-tensor、`MAX_LAG_STEPS=2`、20 steps | train launcher `0/0`；analysis 有 16 valid ranks、global control epochs 0--20 sealed/contiguous、20/20 capture；但 root post-analysis `FAILED` exit 2，无 summary/phase timing | max rank/global lag 2（steps 3/8/12/15）；只满足 relaxed-lag two，不是 strict 1C1S | local-full 320: shared 1,079,838,300,160 B, max capture/commit/queue/transport/serialize 2.013/20.260/6.001/16.408/3.847 s; remote 320: payload/wire 4,413,912/5,170,091 B, max 0.285/662.986/2.447/465.755/530.477 ms | 未运行 | active-shell script edit caused post-analysis `unexpected EOF`; infrastructure-contaminated development evidence，不能计正式 20-step、paired overhead、P99、recovery、e2e 或 grouping；`RUN_RESTORE=0` |
| 2026-09-08 | GPT dense 6.33B r71 seq1024 GA64 20-step NoCheckpoint | node17/node8，BF16，`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64、constant LR、seed20260908、mode off，20 steps | `COMPLETE`；launcher `0/0`；20 rank-0 iter/step-wall records；16 final digests at iteration 20 | 不适用 | driver phase wall 270.875787752 s；steps 2--20 n=19 iteration mean/min/max 10.306947368/10.277/10.333 s，step-wall 10.308315789/10.279/10.335 s | 未运行 | clean same-workload NoCheckpoint reference；无 checkpoint/service/detector/grouping/recovery，非 1C1S/AdaCheck/checkpoint-overhead/e2e 结果 |
| 2026-09-08 | GPT dense 6.33B r72 seq1024 GA64 peer restore | node17/node8，BF16，`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64、fixed mirror `peer_delta`、`MATERIALIZE=SEALED=1`、shared-tensor、`MAX_LAG_STEPS=2`、20 steps，然后 peer epoch-20 worker restart | `COMPLETE`；train/restore launcher `0/0`；16 valid ranks，global epochs 0--20 committed；recovery oracle fail-closed `valid=false` | max rank/global lag 2（steps 3/7/11/19）；relaxed-lag two，不是 strict 1C1S | local 320: shared 1,079,838,300,160 B；remote 320 payload/wire 4,416,472/5,172,660 B；local mean/max commit 8684.780/14485.531 ms，remote 78.474/689.328 ms；train/restore phase 881.574255546/425.665780597 s | 仅 loadability/liveness | epoch-20 boundary model/optimizer bitwise 16/16 mismatch，scheduler match；`VALIDATE_MATERIALIZED_STATE=0`，无 materialized numerical oracle，不能计 recovery、1C1S、P99、e2e 或 grouping |
| 2026-09-08 | GPT dense 6.33B r73 uninterrupted 21-step reference | 与 r72 同 seed/configuration，mode off，21 steps | `COMPLETE`；launcher `0/0`；16 terminal digests | 不适用 | train phase 279.848533485 s；rank-0 step 21 loss 6.8906，iter/wall 10.338/10.340 s | 未运行 | r72 numerical recovery oracle 的 uninterrupted reference；不是 checkpoint/1C1S 结果 |
| 2026-09-08 | GPT dense 6.33B r74 21-step replicate | r73 同 non-scalar seed/configuration，mode off，21 steps | `COMPLETE`；launcher `0/0`；r73/r74 16/16 terminal state/model/optimizer/scheduler digest exact | 不适用 | step 21 loss 6.8906，iter/wall 10.292/10.293 s | 未运行 | 终端 schema 无独立 runtime checksum；iteration/loss 一致，排除 non-scalar run-to-run nondeterminism |
| 2026-09-08 | GPT dense 6.33B r75 scalar 21-step reference | r73 配置，`MERAK_FORCE_SCALAR_ADAMW=1`，mode off，21 steps | `COMPLETE`；launcher `0/0`；16 terminal digests | 不适用 | step 21 loss 6.8438，iter/wall 10.290/10.292 s，与 r72 restored loss 相同 | 未运行 | 定位 r73/r72 loss 为 optimizer-kernel confound；未来所有 runner 统一 scalar；不是 recovery 结果 |
| 2026-09-08 | GPT dense 6.33B r76 scalar materialized-oracle 20-step | GPT 6.33B scalar sealed `peer_delta`，`VALIDATE_MATERIALIZED_STATE=1`，20 steps | 20-step loss trajectory logged；final concurrent 16-rank materialized GET OOM；`FAILED` exit 1，launcher `143/1` | 无有效最终 lag/oracle verdict | step 4/8 poll wall 23.265/22.677 s；node8 service approx 301,945,960 kB anon RSS；0 metrics/0 digests | 未运行 | Linux OOM killer 后 connection closed mid-frame；oracle scalability negative，非数值失败/P99/overhead/recovery/grouping 结果；后续 `materialized_response_lock` 覆盖 serialize+send，14 service tests pass |
| 2026-09-08 | GPT dense 6.33B r77 serialized materialized-oracle 20-step | node17/node8，BF16，scalar AdamW，`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64、constant LR、seed20260908；fixed mirror `peer_delta`、`MATERIALIZE=SEALED=1`、shared-tensor、`VALIDATE_MATERIALIZED_STATE=1`、`MAX_LAG_STEPS=2` | `COMPLETE`；launcher `0/0`；16/16 valid/contiguous，global epoch 20 committed；r76 后 serialize+send response lock 下无 service OOM | max rank/global lag 2（steps 6/9/12/15）；仅 relaxed-lag two，非 strict 1C1S | driver train phase 1680.702231021 s（含 setup/finalization）；local 320 shared handoff 1,079,838,300,160 B；remote 320 payload/wire 4,416,152/5,172,337 B | epoch-20 active-job materialized-state oracle：16/16 numerical-close，0/16 model+optimizer bitwise equal；max BF16 tensor relative Linf 0.004464285714285714 < 0.0078125，optimizer max abs 1.1920928955078125e-7 | `RUN_RESTORE=0`/`restore_attempted=false`：不是 fresh-job recovery、continuation、host failure、durable recovery、P99、paired overhead 或 grouping 结果；证明 r76 OOM 修复后的单次 oracle scalability/numerical-path 成功 |
| 2026-09-08 | GPT dense 6.33B r78 strict-lag-one tail-fix | node17/node8，scalar GPT 6.33B、BF16、`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64；fixed mirror `peer_delta`、strict `MAX_LAG_STEPS=1`、5 steps | global controls 0--5 全部 committed；16 metrics contiguous，但 0/16 valid；`FAILED`，launcher `local=1`/`remote=143`，无 summary/digest | lag `1,1,2,1,2`，max 2 > configured 1，strict 1C1S fail | local-full 80：mean/max commit 9.637/21.626 s，enqueue-to-commit 11.179/21.634 s，publisher queue max 10.872 s；remote 80 payload/wire 1,104,032/1,292,997 B | 未运行 | 前台日志不再有此前约10 s extra poll stall，说明 PIN 解锁消除同步 stall；后台服务率仍不足 strict lag-one。随后 `get_materialized` staged-first evidence race fix 12/12 tests pass，但无 post-fix recovery，非 recovery/P99/overhead/e2e/grouping 结果 |
| 2026-09-08 | GPT dense 6.33B r79 lag-two oracle + healthy-service restart | node17/node8，scalar GPT 6.33B、BF16、`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64；fixed mirror `peer_delta`、`MATERIALIZE=SEALED=1`、shared-tensor、`VALIDATE_MATERIALIZED_STATE=1`、`MAX_LAG_STEPS=2`，5 train + 1 resumed step | `COMPLETE`；train/restore launcher `0/0`；global controls 0--5；16/16 valid/contiguous | max rank/global lag 2，relaxed-lag two，非 strict 1C1S | train/restore phase 1506.285/654.971 s（含 oracle/restart）；local shared handoff 269,959,575,040 B；remote payload/wire 1,104,672/1,293,629 B | epoch-5 active-job materialized oracle：16/16 numerical-close、0/16 model+optimizer bitwise equal；16 train + 16 restored-boundary + 16 restore digests；resumed step-6 loss 8.3750 | 对 r80 的 recovery oracle `valid=true`：boundary numerical-close、logged loss equal、numerical continuation true；boundary/final model+optimizer 仍16/16 bitwise mismatch、scheduler match、bitwise continuation false。仅 lag2 healthy-service restart，非 lag1/host failure/durable/P99/paired-overhead/e2e/grouping |
| 2026-09-08 | GPT dense 6.33B r80 scalar NoCheckpoint 6-step reference | 与 r79 同 source/scalar training configuration，mode off、`byte_stream`、6 steps | `COMPLETE`；launcher `0/0`；16 terminal digests | 不适用 | phase 124.882 s；rank-0 step-6 loss 8.3750 | 为 r79 continuation oracle reference | 首次同标识命令因 `mode=off` 携带 `shared_tensor` 被参数校验拒绝、未启动实验；随即以 `byte_stream` 正确重启，只有完成的 invocation 计入 r80；非 checkpoint/1C1S/grouping 结果 |
| 2026-09-08 | GPT dense 6.33B r81 200-step sustained-rate | node17/node8，scalar、BF16、`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64；fixed mirror sealed `peer_delta`、configured lag=2、目标 200 steps | `FAILED`；launcher `local=143`/`remote=1`；rank-0 仅至 step 15，node8 rank8 enqueue epoch16 时 bounded inflight capacity 溢出；global control files 仅 0--11，0 final metrics/summary/analysis/digest | 无有效 final max-lag；不把 epoch16 与 control11 的未闭合差距报告为正式 lag | aborted train phase 671.269 s；无最终 remote payload/wire、cumulative shared handoff 或 queue/commit-tail 指标 | 未运行 | 200-step sustained-rate/backpressure 负结果；非 strict 1C1S、P99、recovery、end-to-end、overhead 或 grouping 证据 |
| 2026-09-08 | GPT dense 6.33B r82 200-step NoCheckpoint | 与 r81 同 source/config，scalar、BF16、`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64、constant LR、seed20260908，mode off | `COMPLETE`；launcher `0/0`；200/200 rank-zero steps，16 terminal digests at iteration 200 | 不适用 | phase 2126.841 s；steps 2--200 (`n=199`) wall mean/P50/P95/P99/max `10.305/10.306/10.330/10.349/10.353 s` | 未运行 | 证明训练 baseline 可持续并将 r81 失败定位到 checkpoint 路径；因 treatment 未完成，不能计算 overhead，也不是 checkpoint/1C1S/grouping/recovery 结果 |
| 2026-09-09 | GPT dense 6.33B r83 GA80 strict cadence | node17/node8，scalar、BF16、`PP4/TP2/DP2`、`dp-pp`、seq1024、GA80；fixed mirror sealed `peer_delta`、shared tensor、strict lag=1、20 steps | `COMPLETE`；launcher `0/0`；16/16 valid/contiguous，global epochs 0--20，16 terminal digests | 所有 20 个 rank/global boundary lag 均为 1；首次 GPT 6.33B 20-step strict cadence pass | steps 2--20 wall mean/P99 `13.694/15.549 s`；local 320 commit mean/max `9.640/11.467 s`；remote payload/wire `4,413,912/5,170,108 B`；apply mean/P99/max `8.852/10.891/11.144 s` | 未运行 | 两侧均物化到 epoch20、无 backlog/poison；未启用 oracle/restore，GA80 改变 effective batch，故只作 cadence sensitivity，不作 grouping/overhead/恢复主张 |
| 2026-09-09 | GPT dense 6.33B r84 GA80 strict cadence + restore diagnostic | node17/node8，scalar、BF16、`PP4/TP2/DP2`、`dp-pp`、seq1024、GA80；fixed mirror sealed `peer_delta`、shared tensor、strict lag=1；5 train + 1 fresh restore step | `COMPLETE`；train/restore launcher `0/0`；16/16 valid/contiguous；train controls 0--5；train digests iteration 5，post-restore digests iteration 6 | train boundary lags all 1；fresh worker loads peer epoch 5 and runs step 6, loss `8.2500`/wall `411.318 s` | train/restore phase `1496.489824706/630.197100218 s`；80 local handoffs `269,959,575,040 B`；remote payload/wire `1,104,672/1,293,625 B`；80 apply mean/P95/P99/max `8.862/11.905/13.772/13.772 s` | active materialized oracle 16/16 numerical-close, 0/16 jointly bitwise; max BF16 rel Linf `0.004545454545454545`，optimizer max abs `1.1920928955078125e-7`；无 recovery-oracle | `healthy-services-worker-restart`，无真实 host/service fault；仅 fresh-worker loadability/liveness，非 numerical continuation、非完整 strict 1C1S recovery contract；source 与 r83 相同、与 r81/r82 不同，不能作 paired overhead/performance、grouping 或 durability 结论 |

上述两机失败来自
`incremental-1c1s-compute64-20260907T154215Z` 和
`incremental-1c1s-fastservice-20260907T154743Z`。更早的 compute-repeats=4 gate 在
full-update 第 5 个 epoch 已触发 backpressure，且旧 driver 未继续 peer-delta；它只用于定位
容量与生命周期问题。服务端移除重复 object-catalog 反序列化后，失败边界仅从第 5 个 epoch
推迟至 full-update 第 6、peer-delta 第 7 个 epoch，因此当前瓶颈不只是服务端入站校验。
这些 full-update/peer-delta 行是 data-plane 与恢复协议 control，不是 pairing-only 的
`AdaCheckStyleFixed`/`RedundancyMax` 正式对比。后续 service-rate gate 已通过，但 Merak 大模型
正式 gate 仍待执行；保留失败开发 run 是为了避免选择性呈现。

第一条通过的系统 gate 位于
`runs/two-node-v2/incremental-1c1s-stablecompute512-r2-20260907T180000Z`。其 compute workload
是保持模型输出尺度不变的 repeated-forward mean，用于验证 producer/service rate，而非语言模型
吞吐结果。NoCheckpoint max-rank loop 为 4,390.41 ms。peer-delta 同时写 local full update
2,691,663,920 bytes 和跨 host runtime delta 4,317,360 bytes；因此它降低的是 failure-domain
之间的 transfer，而不是当前实现的 aggregate service-resident bytes。当前 peer 多一个 publisher
和 snapshot，capture mean 最大值为 17.22 ms（full 为 5.88 ms），所以逻辑字节收益尚未转化为
训练时间收益。

后续 r3 A/B 复用 caller-owned immutable CPU snapshot，并从 local-full record 移除由 remote delta
单独覆盖的 runtime state。full/peer/off 的 max-rank loop 分别为 5,138.60/4,943.19/4,245.36 ms；
peer 的最大 rank mean capture 降至 2.25 ms，full 为 6.18 ms。该改动保留 20/20、lag=1 和相同恢复
合同，说明 r2 的额外 capture 是可消除实现成本；但短 run 的背景抖动明显，3.80% peer-vs-full
差异只能作为开发 A/B，正式结果仍需预注册的独立 paired trials。

恢复采用两层合同：checkpoint 边界的 model、AdamW 和 scheduler 必须按原始 tensor bytes 一致；
fresh DistributedDataParallel job 继续训练允许预注册的 `rtol=1e-6, atol=1e-7`，但所有状态必须
finite。该 run 继续 5 steps 的最大参数绝对误差为 `1.12e-8`。跨节点 collective 的规约顺序不
保证逐位确定，因此 continued-parameter bitwise equality 保留为诊断字段，不作为恢复正确性门槛。

## 当前结果与 AdaCheck 的可比性

| 维度 | AdaCheck 论文报告 | 我们当前实测 | 能否直接比较 |
|---|---|---|---|
| checkpoint cadence | Figure 12 的四个选定配置每个 iteration checkpoint；未覆盖全部模型/并行策略 | 真实 0.94B r20 达到 20/20 且每个 boundary local/global lag=1；8B r22 运行中 | 合同层面已出现可比较正点；尚缺 200-step、恢复和多模型 |
| training overlap | 远端 CPU optimizer 与训练重叠，报告接近零额外开销 | r20/r21 单次 paired steady overhead 2.92% | 尚不能直接比较；需预注册多 trial CI 与 sealed recovery |
| remote CPU optimizer | 论文设计包含 | 公开 artifact 无完整路径；本研究 CPU numerical materializer 有界但 r9 mean local operation 2.419 s | 只能比较机制合同，不能称 AdaCheck 复现 |
| 恢复正确性 | 论文系统的故障恢复结果 | toy 有数值 oracle；r7 仅验证 16-rank global-control loadability 和一步 continuation | 不能比较绝对 RTO；r7 尚缺 uninterrupted oracle 与 host fault |
| grouping | 相似计算图 worker 的固定分组 | DP-first 对称 catalog 下 fixed 等于 exact；随机错配显著更差 | 可作为“零优化空间”负对照 |
| 冗余带来的传输字节 | 论文使用 gradient redundancy reduction | peer-delta 同机 smoke 的逻辑 remote-service application bytes 比 full update 少约 99.84% | `full_remote` 是弱协议消融；仅为机制证据，不能作为 grouping 收益、AdaCheck 对比或端到端加速 |

早期 fast-step development runs 仍记录在上文。当前 stable-compute service-rate gate 已在两机
达到 20/20 和 lag=1，但还不是 Merak 大模型训练，也未达到 AdaCheck 论文的接近零开销结果。

AdaCheck 的论文数字不是本仓库复现实测值。任何最终图表都必须把 `paper-reported`、
`public-artifact-observed` 和 `ours-measured` 分开，不能在同一数轴上暗示相同软硬件条件。

### r51 catalog 与 BF16 capture 边界

`runs/two-node-v2/topology-capture-llama1b-ppdp-world-stable-r51-20260908T151832Z/`
保留了两个完整、world-scope AdaCheck detector capture。full-state frozen catalog 的 fixed
pairing 为 `(0,8)..(7,15)`，exact pairing 为 `(0,12)..(3,15),(4,8)..(7,11)`；其
pair-local logical tensor-byte proxy 分别为 `7,910,538,752` 和 `5,813,386,752` bytes，故
静态差为 26.5109%。它假设同 epoch rank requirements 已可用，未计 AdaCheck online detection、
gradient incremental encoding、serialization、NIC、queue、P99、pair migration 或完整恢复对象，
只能作为 full-state static opportunity upper bound。

同一 run 的相邻 epoch checksum activity diagnostic 则显示相反的必要事实：3,576 个
rank/path requirements 中只有 8 个变化，均为 5,056-byte `torch_rng_state`，合计 40,448 bytes；
fixed 与 exact 均为 40,448 bytes，gap 为 0%。该输出已显式标为
`peer_delta_offline_activity_diagnostic` 和 `eligible_for_runtime_gate=false`。它是对 activity
filter 的负向 sanity diagnostic，不是 AdaCheck online gain，也不能用其静态 failure coverage
替代完整 checkpoint 的恢复证明。

r52--r54 均有 `EXCLUDED.txt`，但没有 source capture、detector capture、frozen catalog、payload
或 opportunity output：r52 在 BF16 name/value NumPy conversion 失败，r53 在独立 BF16 tensor
hashing 失败，r54 在 Adam 零维 FP32 `step` tensor 的 raw-byte view 失败。三者不得进入表格统计、
图表、baseline 对比或研究结论。

r55 是修复这些 BF16 capture 边界后的成功 world detector artifact：epoch 1/2 frozen catalog
各有 `34,297,038,640` full logical bytes 和 `17,148,026,372` reduced logical bytes，static 与
temporal 的 fixed/profile-exact gap 均为 0%。在冻结 bundle、catalog、metadata、summary 与
provenance hash 校验后，32 个 raw source `.bin` 文件按 `RAW_TENSOR_CLEANUP.md` 删除；它们不能
恢复，也不影响保留的 frozen catalog 结论。

r56 的 two-epoch pre-optimizer catalog 则只验证一个较强的负对照：上一 epoch 的 frozen
unrestricted content-oracle plan、fixed plan 和 evaluated-epoch hindsight plan 在该 workload 上均
没有任何 pair-local byte advantage。同步 catalog hash 使 lag 达到 2，因此尽管 train launcher
均成功，root 仍为 `FAILED`。它不能用于 cadence、P99、training-time、online AdaCheck 或恢复比较，
也不能把其 `full_remote` grouping 输出接入 current mirror-DP `peer_delta`。

### Qwen2-MoE compatibility and AdaCheck EP boundary

r57/r58 是同一个两 host、16-rank BF16 `PP8/TP1/DP2`、`dp-pp`、sequence length 64、one-step
preflight 的两个显存条件。r57 无 ZeRO 时虽完成 FX trace/layer split，却在 AdamW `exp_avg`
初始化时约 44.54 GiB OOM；r58 仅加 `ZERO_STAGE=1` 即以 launcher `0/0` 完成，loss 为 12.5369、
iteration time 为 8.169 s。二者都使用 checkpoint mode `off`；r58 也没有 sealed checkpoint 或
update catalog。因此 r58 只说明该随机初始化 Qwen2-MoE 结构在 ZeRO-1 下可完成一个训练 step，
而不是 grouping、AdaCheck payload、1C1S、P99、恢复或端到端性能结果。

公开 `cf910be` 的 MPU 网格只有 PP/DP/SP/TP，没有 EP process group 或 expert placement；检查到
的 MoE 代码仅涉及 norm/checker 或 AMP bookkeeping。AdaCheck 论文却将 GPT-MoE/DeepSeek sparse
配置写为 `EP+MP`、`EP+ZeRO-1` 和 auto-generated plans，并在 Appendix B 给出 128 RTX 3090 GPU
上的 GPT-MoE。故公开代码不足以复现论文 EP 路径，当前 Qwen run 必须标作 no-EP sparse-architecture
compatibility preflight，不能称为真实 asymmetric expert placement。两个 launcher 新增的可选
`ZERO_STAGE=1` 及其 metadata 记录只保证配置可追溯，不改变这个科学边界。

r59 把这个 no-EP Qwen 配置延伸为两真实训练/采集 step 的 world-scope AdaCheck detector
metadata-only capture。两 epoch 共保存 32 份 per-rank object catalogs（冻结 `.jsonl` 内有 10,164
条 object metadata），约 97 MB、无 `.bin` payload；model-role detector evidence 在 second epoch
确认稳定。其 full/reduced logical metadata 为 `295,639,187,232` / `270,156,295,968` bytes，但无
reduced bundle、不能 reconstruct，且 `recoverable_payload_present=false`。因此它不进入 checkpoint
size、NIC、P99、runtime gate、1C1S、recovery 或 end-to-end 表格。

在该 frozen single-epoch full-tensor-metadata 的 pair-local objective 下，fixed mirror 和
`profile_exact` 均取 `(0,8)..(7,15)`，都是 `255,954,378,528` bytes（0% exact gain）；random 为
`284,985,777,952` bytes。它说明随机错配会更差，并将大型 MoE 放入“固定镜像在当前静态目标下已最优”
的负对照集合；这不是在线 AdaCheck grouping treatment，不能从 0% 推出端到端 checkpoint 结论。

r60 为未来 Qwen checkpoint treatment 提供匹配的训练基线：它固定 BF16、`PP8/TP1/DP2`、`dp-pp`、
ZeRO-1、sequence length 64、GA1、constant LR 和 seed 20260908，在 mode `off` 下完成 20/20。
node-0 step 1 的 warmup 为 8.100 s；steps 2--20 的 19 个 logged iteration times 的算术均值为
`4.660157895 s`，范围 `4.600--4.697 s`。r60 没有 checkpoint/service/catalog/detector/grouping/
recovery/EP；它只能作为此配置未来 matched checkpoint 条件的 NoCheckpoint reference，不能称
1C1S、AdaCheck、grouping 或 end-to-end checkpoint 结果，也不能用来估计 r59 metadata capture 的开销。

### GPT dense 6.33B TP2 preflight and exclusions

r61/r62 是 BF16 `PP4/TP2/DP2`、`dp-pp`、sequence length 128 的零步实现诊断，不属于 GPT
模型实验。r61 的 `GPT2Attention` 在 FX symbolic tracing 中因 `q_len > 1` control flow 抛出
`TraceError`；注册为 FX leaf 后，r62 才完成 trace/layer split，却在 stage-0 first forward 以
`shape '[1, 128, 4096]' is invalid for input of size 262144` 失败。r63 改为 eager-only 后完成一
个 step（launcher `0/0`、16 terminal digests、loss 11.7812、1.792 s），确认注意力路径问题已被
隔离；但其 `n_inner=14336` MLP 未 TP shard，所以不能作为完整 TP2 性能结果。

r64 以这个不完整 mapping 实际跑完五个训练 step，但发现问题后在 terminal digest 期间被主动终止，
故 root `FAILED`/`EXCLUDED`，五个 logged step 不进入结果。最小修复补齐 MLP
`(2,7,col)/(7,2,row)` mapping，并保留 eager workload handling 和 `GPT2Attention` FX leaf；focused
workload suite 通过 13 tests。r65 因而完成完整 TP2 的一 step preflight（launcher `0/0`、16 terminal
digests、loss 11.5938、1.739 s、stage params 0.831--0.861B），证明 MLP sharding 生效。r63--r65
均为 mode-off implementation/correctness evidence：没有 checkpoint/catalog/detector/grouping/recovery
observation，不能写为 GPT-6.33B 的 NoCheckpoint baseline、1C1S、AdaCheck、grouping、吞吐或模型结果。

r66 是修复后完整 TP2 的首个 5-step NoCheckpoint reference：node17/node8、BF16
`PP4/TP2/DP2`、`dp-pp`、seq128、GA1、constant LR、seed 20260908、mode off；launcher `0/0`、5/5
和 16 terminal digests。可复现配置为 `STEPS=5 PP=4 TP=2 DP=2 MESH_ORDER=dp-pp SEQ_LENGTH=128`
`GRADIENT_ACCUMULATION_STEPS=1 SEED=20260908 LR_SCHEDULER_TYPE=constant`，执行
`bash tools/run_two_node_merak_1c1s.sh gpt_6b off <run-dir>`。node-0 step 1 warmup 为 1.723 s；
steps 2--5 的 n=4 平均为 0.4775 s、range 0.467--0.505 s。它仍没有 checkpoint/service/catalog/
detector/grouping/recovery，故只是短 NoCheckpoint reference。

r67 固定 mirror groups `(0,8)..(7,15)`，以 `peer_delta`、`MATERIALIZE=1`、`SEALED=1`、GA64、
shared-tensor local transport、`MAX_INFLIGHT=4`、`CPU_THREADS=2` 和 strict `MAX_LAG_STEPS=1` 跑 3
steps。其完整 driver 为 `env STEPS=3 CONTINUE_STEPS=1 PP=4 TP=2 DP=2 MESH_ORDER=dp-pp SEQ_LENGTH=128`
`GRADIENT_ACCUMULATION_STEPS=64 SEED=20260908 MAX_INFLIGHT=4 LR_SCHEDULER_TYPE=constant`
`MATERIALIZE=1 SEALED=1 CPU_THREADS=2 STRICT_GATE=1 MAX_LAG_STEPS=1`
`LOCAL_TRANSPORT=shared_tensor bash tools/run_two_node_merak_1c1s.sh gpt_6b peer_delta <run-dir>`。
虽然 all-rank epochs contiguous，global manifests 已 sealed 到 epoch 3（control epochs 0--3），
`analysis.json` 的 boundary lag 为 `1,2,2`，max global/rank lag=2 超过配置 1，root 因此
`FAILED`（launcher `1/143`）。48 local-full updates 的 max commit/queue/transport/serialize 为
23.561/18.715/20.505/3.767 s；48 remote deltas 的 payload/wire 661,456/774,830 B、max commit
534.165 ms。这是 full-TP2 sealed-path tail 诊断，未跑 restore 或 materialized-state oracle；不能称
1C1S、P99、recovery、end-to-end 或 AdaCheck pairing A/B。r66 的 GA1 与 r67 的 GA64 也使二者不构成
checkpoint-overhead 对照。

r68/r69 是第一个训练 workload 完全匹配的 GPT full-TP2 3-step ordered pair：node17/node8、BF16
`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64、constant LR 和 seed 20260908；顺序固定为 r68 后 r69，未
随机化或交错。r68 mode-off `COMPLETE`（launcher `0/0`、16 terminal digests）；rank-0 iter times
11.496/10.294/10.290 s，steps 2--3 的 n=2 mean 为 10.292 s。r69 仅打开 fixed mirror
`peer_delta` sealed path（`MATERIALIZE=SEALED=1`、shared-tensor、strict lag=1），rank-0 为
13.724/12.446/12.265 s，steps 2--3 mean 12.3555 s。两窗口的观测差为 2.0635 s（20.05%），但样本
仅各两步、执行顺序固定且 r69 门禁失败，不能报告为 checkpoint overhead 或性能结论。

r69 的 all-rank epochs contiguous，global manifests control epochs 为 0--3，三 epoch 都 sealed；
然而 local rank lag 是 `1,1,1`，global lag 是 `1,1,2`。严格合同要求二者都不超过一，故 final
global lag=2 使 r69 `FAILED`（launcher `1/143`），不是 1C1S success。其无 final digest、restore 或
materialized-state oracle。该 run 的 48 local-full updates 共享交接 161,975,745,024 B，max
capture/commit/queue/transport/serialize 为 2.049/16.499/4.223/13.102/3.570 s；48 remote deltas 的
payload/wire 为 661,840/775,212 B、max commit 690.948 ms。这些是一次失败的 tail 诊断，不是 P99、
recovery、end-to-end 或 AdaCheck pairing A/B 证据。

r70 是同一 GPT full-TP2 训练配置的 20-step relaxed-lag two 数据平面诊断：node17/node8、BF16
`PP4/TP2/DP2`、`dp-pp`、seq1024、GA64、constant LR、seed 20260908，fixed mirror `peer_delta`，
`MATERIALIZE=SEALED=1`、shared-tensor、`MAX_INFLIGHT=4`、`CPU_THREADS=2`、`MAX_LAG_STEPS=2`。
训练 launcher 为 `0/0`；analysis 有 16 valid ranks、contiguous/sealed global control epochs 0--20 和
final committed epoch 20，故每步 capture/commit 观察完整。其 320 local-full paths 的 shared handoff 为
1,079,838,300,160 B，max capture/commit/queue/transport/serialize 为
2.013/20.260/6.001/16.408/3.847 s；320 remote deltas 的 payload/wire 为
4,413,912/5,170,091 B，max 为 0.285/662.986/2.447/465.755/530.477 ms。

但是 r70 的 max rank/global boundary lag 为 2（steps 3/8/12/15），只符合其放宽的 lag-two 设置，
并不符合 `lag<=1` 的 strict 1C1S 定义。更重要的是，训练结束后活跃 shell 中同一 runner script 被编辑，
post-analysis 出现 `unexpected EOF`，根运行以 exit 2 `FAILED`，未产生 summary 或 phase timing。因此
r70 只能作为 infrastructure-contaminated development evidence：不得列为正式 20-step 成功或 paired
overhead，也不得据这些单次 maximum 声称 P99。`RUN_RESTORE=0` 且无 materialized-state oracle，故没有
recovery、end-to-end 或 grouping treatment 结论。

r71 提供同训练负载的 clean NoCheckpoint reference：node17/node8、BF16 `PP4/TP2/DP2`、`dp-pp`、
seq1024、GA64、constant LR、seed 20260908、mode off，完成 20/20，launcher `0/0`，且有 16 个
iteration-20 train digests。driver phase wall 为 270.875787752 s（含 setup/teardown）。rank-zero step 1
warmup 是 11.663 s iter / 11.665 s wall；steps 2--20（`n=19`）iter mean/min/max 为
10.306947368/10.277/10.333 s，step-wall 为 10.308315789/10.279/10.335 s。r71 不含 checkpoint、
service、publication、detector、grouping 或 recovery，故仅是后续相同 workload checkpoint treatment 的
baseline，不是 1C1S、AdaCheck、checkpoint-overhead 或 end-to-end 结果。

r72 是 r71 同训练负载的 sealed `peer_delta` restart diagnostic：fixed mirror、shared-tensor、
`MAX_LAG_STEPS=2`，训练和 restore launcher 都为 `0/0`，16 rank valid，global manifests 0--20 committed。
它有 320 local-full（shared handoff 1,079,838,300,160 B）和 320 remote-delta（payload/wire
4,416,472/5,172,660 B）路径；但 lag 在 steps 3/7/11/19 达到 2，故是 relaxed-lag two，不是严格 1C1S。
local mean/max snapshot-ready/shared-copy/put-service/commit 为 315.399/570.244、1203.982/3691.506、
6769.486/9445.333、8684.780/14485.531 ms；remote serialize/commit 为 11.431/494.005、
78.474/689.328 ms。这些是一项 single-run path diagnostic，不是 P99。

r72 的 train/restore phase 分别为 881.574255546/425.665780597 s，包含 drain/restart，不能直接当作
每步开销。steps 2--20 的 rank-zero step-wall mean 为 13.532842105 s；相对 r71 的有序单次差为
+3.224526316 s/+31.28%。r71/r72 是独立分布式 jobs，NCCL reduction 不保证跨 job bitwise checksum
一致，因而 16/16 training digest 差异不否定这个时间观察；它是单次、固定顺序 paired condition 的
development observation，而非可报告置信区间或论文主结论。更关键的是，健康 services 下的 peer worker
restart 虽然完成，16 份 restored-boundary epoch-20 digest 与 r72 自身训练边界的 model/optimizer
checksum 全部不一致（仅 LR scheduler 相同）。`VALIDATE_MATERIALIZED_STATE=0`，因此没有自动数值 oracle；
step 21 的完成仅证明 loadability/liveness，不证明 numerical restoration 或 trajectory equivalence。r72
不进入 recovery、strict 1C1S、e2e、P99 或 grouping 主结果。

r73 是 r72 的同 seed/configuration uninterrupted 21-step reference：`COMPLETE`、launcher `0/0`、
16 terminal digests、train phase 279.848533485 s，rank-zero step-21 loss 为 6.8906（iter/wall
10.338/10.340 s）。r72 的 `recovery-oracle.json` 虽为 `valid=false`，但这是 fail-closed：
`VALIDATE_MATERIALIZED_STATE=0` 使 16 份 materialized numerical oracle 均缺失，比较器因而把
`checkpoint_boundary_numerically_close` 置为 false；它不证明 numerical divergence。r72 自身 epoch-20
boundary 的 model/optimizer 仍是 bitwise 16/16 mismatch，scheduler 一致，故仅有 loadability/liveness。

r74 是 r73 non-scalar reference 的独立 replicate，`COMPLETE`、launcher `0/0`；r73/r74 的 16/16
terminal state/model/optimizer/scheduler digest 逐位一致，iteration-21/loss 都是 6.8906（终端 schema 无
独立 runtime checksum）。r75 打开 `MERAK_FORCE_SCALAR_ADAMW=1` 后也 `COMPLETE`、launcher `0/0`，其
step-21 loss=6.8438，恰与 r72 restored loss 一致。由此此前 6.8906 vs 6.8438 是 optimizer-kernel
confound，而不是恢复数值证据。未来 runner 全条件统一 scalar。scalar oracle `valid=false` 仍只是 oracle
缺失造成的 fail-closed verdict，不能误写为 numerical failure；r72 依旧不进入 recovery 主结果。

r76 是 scalar `VALIDATE_MATERIALIZED_STATE=1` 的 20-step 扩展诊断。20 个 scalar training step 全部日志
完整，rank-zero loss 从 11.9375 下降到 7.3750；poll wall 仅在 step 4/8 达到 23.265/22.677 s。然而最终
16 rank 并发 materialized-state GET 使 node8 checkpoint service 的 anonymous RSS 达约 301,945,960 kB，
被 Linux OOM killer 杀死，rank 随后报告 connection closed mid-frame。r76 root `FAILED` exit 1，launcher
`local=143/remote=1`，没有 metrics 或 final digests，故没有数值 oracle 或 recovery verdict。它是 oracle
scalability negative，不是 numerical failure，也不能用于 P99、overhead 或 grouping。

最小服务修复是让 `ReplicaTCPServer.materialized_response_lock` 覆盖 materialized snapshot 的 serialize
和 send，普通请求仍不串行；focused replica-service suite 14 tests passed。该修复的效果仍须由干净 scalar
oracle run 验证。

这不推翻 r71/r72 的时间 development observation：独立 distributed jobs 的 NCCL reduction 不承诺跨 job
bitwise identity，checksum 差异不是时间比较失效的理由。该比较仍因 `n=1` 与固定执行顺序不能成为正式
CI 或论文主结论。最小下一诊断是在相同 scalar peer-restore 边界开启
`CKPT_VALIDATE_MATERIALIZED_STATE=1` 的一步 run，先得到 component-level numerical oracle，再决定是否
需要 tensor-level 定位。

分组主实验必须采用同一套 AdaCheck-style redundancy reduction、snapshot、transport、commit 和
recovery，只替换 pairing policy。核心 treatment 是固定/AdaCheck pairing 对完整 pairwise
redundancy matrix 上的最优 pairing；`full_remote` 对 `peer_delta` 仅保留为协议消融。

两个冻结 nnScaler LLaMA-3-8B 布局提供了独立的非对称 placement 负对照。基于 generated
`add_full_map` 的 model slices，gds4/gds8 的 64 条跨 host 边 overlap 全部为 0；fixed mirror
与 constrained exact matching 均无改善，静态补全量分别为 48.230/57.925 GB。它们没有
optimizer/runtime placement，也没有 AdaCheck detector 数据，因此不进入端到端主表。该结果
说明固定 rank-to-host 映射可能先消灭所有可利用冗余；后续应把 balanced remap 的机会界与
training communication cost 分开报告，再决定是否扩展为联合 placement/grouping 优化。
枚举全部 6,435 个平衡划分得到的静态上界能把两布局 completion 降到 0，但会把可见 DP replica
分到跨 host；在测量或建模训练通信前，这个 100% 只能写为 opportunity upper bound。

## 证据与资源记账

| 字段 | 必须记录的含义 |
|---|---|
| evidence level | M0 measurement-only、M1 rank-local、M2 global one-host、M3 durable、M4 efficient materialized 1C1S |
| workload fidelity | toy DDP、stable-compute 或真实 Merak model；不得互相替代 |
| commit scope | captured、rank ACK 或 global manifest；只有最后一项可称全局 checkpoint |
| failure model | 无注入、service refusal、host-stack loss、双 host loss，以及是否 fresh job |
| protection substrate | full remote log、peer local-full+runtime delta、rolling base 或 CPU materializer |
| retention/capacity | `B`、`G/step`、base epoch、log depth、service cap、peak resident bytes 和 headroom |
| rate evidence | producer GB/s、service GB/s、queue lag、queue-full、final drain；final drain 不能冒充稳态 1C1S |
| network scope | logical local bytes、cross-host application bytes、实际 NIC bytes分别报告 |
| shared handoff scope | `cumulative_shared_handoff_bytes` 是本机累计交接量，不是 peak resident memory；local TCP=0 也不包含 Unix 控制、global collective、跨机 delta 或恢复流量 |
| trial status | development single run、预注册 paired trial 或 excluded/negative evidence |

按 BF16 model、FP32 master、Adam moments 和 FP32 gradients 的下界，`B ~= 14P`、`G ~= 4P`，
append-only 200-step resident payload 为 `B+200G ~= 814P` bytes。1B/6.33B/8.03B/14.32B
约为 0.814/5.15/6.54/11.65 TB per host；因此 64 GiB service 下的大模型 append-only 运行在
实验前即可判定不可行，不能靠增大 queue 掩盖。

## 论文级结果矩阵（预注册）

每个模型先做 20-step gate；只有显存、训练、连续提交和恢复全部通过，才运行 200 steps。所有正式
trial 使用相同 BF16、sequence length、global batch、AdamW、service 容量和运行顺序；
2026-09-13 起 grouping 归因采用同一 common runtime 的三臂：`AdaCheckStyleFixed`、
`HostGlobalFixed`、`M1Object`。NoCheckpoint 仅是训练成本下界。

| 模型 | 并行配置 | NoCheckpoint | AdaCheckStyleFixed | HostGlobalFixed | M1Object | 200-step 状态 |
|---|---|---|---|---|---|---|
| LLaMA-3-8B | DP-first，候选 PP4/TP2/DP2 | r109 15-step reference | r110 是 fixed zero-choice control，不是完整三臂新基线 | 待 active-runtime gate | 待 active-runtime gate | 冻结 |
| GPT dense 6.33B | `PP4/TP2/DP2`、`dp-pp` | r105 15-step reference | r106 是 fixed zero-choice control，不是完整三臂新基线 | 待 active-runtime gate | 待 active-runtime gate | 冻结 |
| Qwen2-MoE 14.32B | `PP8/TP1/DP2`、`dp-pp`、BF16、seq64、ZeRO-1 | r60 20-step reference | 未运行 | 未运行 | 未运行 | EP 未实现，冻结 |
| DeepSeek-V2-Lite 15.71B | 由 Merak preflight 选择 | 待测 | 待测 | 待测 | 待测 | 冻结 |

## 当前研究判据

1. `committed` 而不是 `captured` 才计为 checkpoint；必须连续、无 drop/coalesce。
2. strict 1C1S gate 要求每个 rank 的远端提交 lag 最大值不超过 1 step，且 fresh job 能恢复后继续训练。
3. DP-first LLaMA-like-1B 在该冻结 catalog 与当前 pair-local 目标函数下，fixed pairing 等于 exact optimum，因此它是负对照，不能虚构 grouping 收益。
4. 只有存在合法的非对称 holder/requirement 或资源代价，且不破坏恢复预算时，才把冻结
   `redundancy_max/profile_exact` opportunity 提升为三臂 common-runtime gate。
5. 论文主表必须同时报告 total checkpoint bytes、跨机 NIC bytes、training wall time、提交频率与恢复时间；单独的 byte reduction 不能推出训练加速。
6. 当前系统的首要工程 gate 是使 full local update 的稳定服务率不低于每步生产率；调大队列只能诊断吞吐，不能把 lag 隐藏成 1C1S 成功。
7. 当前两机 toy gate 已满足正确性与 cadence；r3 将 peer overhead 从一次开发 run 的 21.98%
   降至另一次的 16.44%，但仍未达到 AdaCheck 的近零开销 claim。下一阶段接入真实 Merak 大模型，
   并用独立 paired trials 区分 workload overlap、local publication 成本与 run noise。
8. 频率曲线同时报告 capture/update rate、global commit rate、最大/稳态 staleness 和 throughput。
   `lag <= k` 不是每 k 步一个 checkpoint；任何跳步方案必须证明可恢复状态与 uninterrupted AdamW
   oracle 等价。
9. raw global control 不是 sealed recovery point。真实 fault matrix 需等待 service pin/retention ACK
   和 own-runtime local copy；否则只能报告 healthy-service worker restart。

## 2026-09-11 SemanticFixed short-gate status

| Run | Model/config | SemanticFixed bytes | Non-blocking/cadence | Recovery | Verdict |
|---|---|---:|---|---|---|
| r99 | GPT-6.33B, BF16, `PP4/TP2/DP2`, DP-first, seq1024, GA80, 16 ranks | `26,995,998,208 B` logical per update over 16 ranks | capture max `2.116 s`; poll max `35.140 s`; queue full at epoch 7 | not run | negative gate |
| r100 | exactly the r99 workload after background commit and ingest/materializer lock separation | `26,995,998,208 B` logical/epoch; 10 epochs `269,959,982,080 B` logical, `269,964,256,072 B` payload, `269,970,084,334 B` wire | all ten steps and global 0--10 complete; poll max `0.771 s`; step mean/P99/max `19.128/22.670/22.765 s`; five visible checkpoints, rank/global lag 4/5; max edge `45.592 s`; final drain `3,019.427 s` | strict materialized oracle: model close 16/16, optimizer close 0/16; no fresh restore | negative recovery gate; second BF16 rounding diagnosed |
| r101 | r100 plus raw BF16 gradient and exact step multiplier | `26,995,998,208 B` logical/epoch; 10 epochs `269,959,982,080 B` logical, `269,964,283,752 B` payload, `269,970,112,020 B` wire | all ten steps/global 0--10; profile capture/poll max `2.944/0.726 s`; step mean/P99/max `19.075/21.261/21.312 s`; six visible checkpoints, rank/global lag 3/4; edge mean/max `26.640/34.815 s`; final drain `3,040.828 s` | materialized model+optimizer close 16/16; fresh epoch-10 restore; steps 11--15 equal r86; numerical oracle valid | complete bounded-lag baseline gate; not strict 1C1S or grouping treatment |
| r104 | same workload, frozen r2 detector and fixed equivalent DP-peer reuse | remote runtime-only `81,408 B/epoch`; local p/o recovery-operation stream `26,995,998,208 B/epoch`; 10-epoch remote payload/wire `2,372,072/2,750,010 B` | all ten steps/global 0--10; six visible checkpoints; rank/global lag 4/4 violates configured lag two; steady mean/max `13.287/13.952 s`; remote edge mean/max `113.523/373.408 ms`; combined final drain about `2,726.014 s` | materialized model+optimizer close 16/16, but cadence exception stops the train launcher before fresh restore | negative cadence/service-rate gate; not recovery, 1C1S, total-checkpoint remote bytes, or M1 |
| r105/r106 | GPT dense 6.33B; node17/node8, BF16 `PP4/TP2/DP2`, `dp-pp`, seq1024, GA80, scalar AdamW; fresh NoCheckpoint 15-step / SemanticFixed fixed-peer 10+5 | r106 remote runtime-only logical/payload/wire `814,080/2,370,792/2,748,739 B`; local p/o operation logical `269,959,982,080 B` | r106 global 0--10; visible 6/10; rank/global lag 4/4; steady mean `13.294 s` vs r105 `12.755 s` (`+4.22%`, single ordered); remote edge mean/max `340.429/1133.870 ms`; finalization max `1494.986 s` | all required 16-rank sets complete; materialized p/o and fresh steps 11--15 pass numerical oracle, loss exactly equal | recovery pass, cadence fail; zero-choice DP2 common-runtime control, not M1/non-blocking/1C1S/host fault |
| r108 | LLaMA-3-8B two-observation detector; node17/node8, BF16 `PP4/TP2/DP2`, `dp-pp`, world scope | metadata full/reduced `290,598,506,544/145,297,178,116 B`; no recoverable payload | 2,653 stable components, 5,348 stable records; catalog hash frozen | not applicable | complete detector input for r110; sizes are metadata, not transfer |
| r109/r110 | LLaMA-3-8B; same 16-rank layout, seq1024, GA96, scalar AdamW; fresh NoCheckpoint 15-step / SemanticFixed fixed-peer 10+5 | r110 remote runtime-only logical/payload/wire `814,080/2,371,432/2,749,370 B`; local p/o operation logical `363,248,839,680 B` | r110 global 0--10; visible 8/10; rank/global lag 1/2; steady mean `22.100 s` vs r109 `21.512 s` (`+2.735%`, single ordered); remote edge mean/max `23.456/49.621 ms`; local shared-path max `20.728 s`; finalization max `2105.938 s` | all required 16-rank sets complete; materialized p/o and fresh steps 11--15 numerical oracle valid at `1e-4`; max loss abs/relative error `0.0002/2.6692e-5`; not bitwise | recovery and bounded-lag-two pass, not strict 1C1S; zero-choice DP2 common-runtime control, not M1/non-blocking/host fault |
| r187/r196--r199 | LLaMA-3-8B 8.03B; 16-rank BF16 `PP4/TP2/DP2, pp-dp`, seq128, scalar AdamW; NoCheckpoint plus common-runtime Ada/Host/M1; base + 2 source, node8 checkpoint-service loss, fresh epoch-2 restore + 1 | Ada/Host/M1 suffix application `72.652/36.326/36.326 GB`; base + suffix total `326.927/290.601/290.601 GB`; M1 vs Ada `-50.000/-11.111%` | suffix max PUT `48.184/27.121/23.708 s`; max distributed commit `60.688/37.606/33.130 s`; final drain `130.482/83.348/76.856 s`; lag `2/2/2` | r199 valid; three arms exact source/restore loss and exact 16-rank final full-state digest after service loss | first recovery-valid LLaMA-8B M1 transfer gain; one sample per arm, final drain required, not sustained non-blocking, strict 1C1S, compute-host loss or changed-layout GPU evidence |
| r201 | GPT-6.33B; 16-rank BF16 `PP4/TP1/DP4, pp-dp`, seq128, scalar AdamW; M1 2+1; base upload parallelism 4 | no suffix: node17 base reached 16 records / 354.353 GB, node8 reached 12 / 270.450 GB | no training step; node8 object service OOM-killed at about 289.7 GB RSS | failed before base seal; ranks report connection refused | `FAILED`, excluded; identifies four-rank base serialization/upload memory peak |
| r202 | same GPT-6.33B M1 contract as r201; only base upload parallelism `4 -> 2` | no suffix; node8 reached 14 base records / about 312.4 GB | no training step | service killed near 321.2 GB RSS | `FAILED`, excluded; confirms steady rank-local base duplication rather than upload concurrency |
| r203 | BERT/GPT-2/LLaMA-3B x `PP4/TP2/DP2`,`PP4/TP1/DP4`; 16-rank two-step BF16 metadata | Ada/Random/RedMax equal under required cross-host pairs | / | static one-host-loss feasibility only | failure-constrained fixed-placement negative control; not opening motivation |
| r204 | same six r203 catalogs; arbitrary rank-level perfect matching | RedMax vs Ada fixed `99.997%--99.99996%` logical reduction | / | host-loss constraint deliberately omitted | exact-DP-replica upper-bound diagnostic only; not opening motivation or runtime/wire/recovery evidence |

r99 has only global epochs 0--2 and cannot be reported as a completed AdaCheck
baseline. R100 removes its foreground control/lock failure but also is not a
completed baseline: optimizer numerical validation fails before restore. The
standalone communication profile remains measurement-only and does not select
grouping, transport, commit, or recovery actions. R101 validates raw BF16 plus
the exact multiplier under the unchanged tolerance and is the first complete
live SemanticFixed gate. Its six visible checkpoints in ten steps and max
global lag four are reported honestly; the `strict_1c1s_gate` summary field means
the configured bounded-lag validator was enabled, not that lag one was achieved.
After r104, cadence and recovery are separate verdicts. R106 and r110 now pass
the fail-closed recovery contract with complete rank sets and exactly five
continuation steps at the preregistered BF16 loss tolerance
`rtol=atol=1e-4`. R106 remains cadence-invalid; r110 passes configured
bounded-lag two but not strict lag one. Both are infrastructure gates only:
symmetric DP2 has one legal peer per rank and no nontrivial grouping choice.
Because there is no positive M1 result and the recovery-readiness tail remains
large, the 20/50/200-step and four-host/32-rank gates stay closed.

## 2026-09-13 ObjectRecovery / M1 pre-runtime gate

| Evidence | Implemented | Still missing | Claim boundary |
|---|---|---|---|
| Same-layout ObjectRecovery | checksummed metadata, survivor-service selection, exact protection-plan validation, complete p/o/runtime live apply | active training-produced service payload/metadata and fresh 16-rank invocation | CPU correctness only; no end-to-end result |
| Changed-layout recovery | FQN/role/global-slice addresses and CPU multi-source assembly | persisted FQN schema plus service-backed PP/TP reshard and runtime-state transformation | not executable end to end |
| Active SemanticFixed plan | detector/semantic provenance; same PP/TP, different-DP candidates; epoch/version checksums; direct fallback; host-loss closure | distributed checksum exchange, trainer hook, transport and sealed commit | plan/test evidence only |
| r115 opportunity | ten training steps; stable step-9/10 metadata; Fixed/M1 equal host-global bytes; predicted max publisher load `1,034,412,000 -> 258,607,584 B` | payload, measured wire bytes/`max_time`, service commit and recovery | metadata proxy; excluded from formal M1 figures |
| r116 CPU/TCP common-runtime gate | three arms; per-epoch sealed bundles/control; either-host loss; epoch 1/2; exact AdamW replay; direct mismatch fallback | distributed checksum-only exchange, live pre-step hook, 16-rank 10+5 | correctness/attribution only; not GPU performance evidence |
| r117 CPU/Gloo semantic gate | three arms x confirmed/mismatch; four processes; world-size explicit Gloo; hash consensus; JSON-only collective; publisher-local payload | distributed service commit/control, base+suffix restore, trainer hook | correctness only; no transport timing or GPU claim |
| r118 distributed commit gate | per-rank bundles; JSON manifest gather; rank-zero control-first seal; external locator; sealed encoding reconstruction; either-host recovery; 18 tests | sealed base, base+suffix fresh training continuation, Trainer callback/launcher | correctness only; no GPU, `max_time`, cadence or non-blocking claim |
| r119 fresh composite gate | sealed base + two suffixes; three arms x either single service-host loss; fresh model/AdamW/scheduler/runtime restore; 6 tests | real 16-rank Trainer continuation and uninterrupted trajectory oracle | CPU/TCP/Gloo correctness only; same-layout restore |
| r120 16-rank service-scale gate | ten suffixes; three arms x either host loss; exact target GET-source sets and replay; Ada/Host/M1 logical `5,120/1,280/1,280 B`; planned max publisher `32/32/8 B` | real GPU/NIC timing, training overlap and reference trajectory | CPU/Gloo/loopback accounting only; M1 restore GET count `620` versus fixed `440` |
| r125--r131 live GPT-2 common runtime | node17/node8, 16-rank BF16 `PP4/TP1/DP4`; NoCheckpoint plus three checkpoint arms; node8 service loss; fresh epoch-10 restore and continuation to 15; exact reference loss and final digest | physical compute-host loss, changed-layout reshard, repeated timing, shared-runtime tail fix | first minimal fixed-layout M1 transfer-and-continuation gate: suffix logical Ada/Host/M1 `13.054/3.263/3.263 GB`, suffix max PUT `1887.923/515.256/425.787 ms`; not the repeatable M1 headline; all arms lag 10 and fail strict 1C1S |
| r134--r139 parallel-PUT GPT-2 gate | same node17/node8 16-rank BF16 `PP4/TP1/DP4`; shared concurrent two-service suffix commit; three recovery-valid arms plus r125 reference | repeats/statistics, physical compute-host loss, changed layout, all-publication M1 load objective | r139 raw suite valid; Ada/Host/M1 suffix logical `13.054/3.263/3.263 GB`, suffix max PUT `2466.345/448.136/430.046 ms`, source phase `93.700/79.043/73.825 s`; exact losses/digests; lag `10/9/9`, non-1C1S; M1-vs-Host timing only diagnostic |
| r140--r142 all-publication M1 recovery gate | same node17/node8 16-rank GPT-2 BF16 `PP4/TP1/DP4`; M1 balances local+remote publications without adding remote bytes; r141 node8 service loss + fresh 10+5 restore | physical compute-host loss, changed layout, stable non-blocking/RTO | r140 `99 passed`; r142 suite valid; M1 vs Ada suffix logical `-75%`, total application `-44.008%`; exact loss/digests; planned max publisher `258.607 -> 77.328 MB` |
| r143/r145--r148 repeated M1 load gate | three alternating HostGlobal/M1 source runs; same detector/bytes/loss/digests; run is the repeat unit | only r138/r141 are recovery-valid; no statistical inference from n=3; final drain/non-1C1S | M1 vs Host mean reduction: distributed commit `36.318%`, max queue `31.615%`, max drain `30.566%`, source phase `8.252%`; suffix max PUT only `0.717%` with overlapping ranges; planned max `-70.098%` |
| r150 PP2 detector input | node17/node8 16-rank GPT-2 BF16 `PP2/TP1/DP8, pp-dp`; 2-step world-scope metadata-only capture | parameterized 3+5 four-arm/two-service-loss gate, then 10+5 if valid | complete; 747 objects, 699 stable components, 5,624 stable records; catalog hash `f9bf...e224`; full/reduced catalog `20.987/2.612 GB` are metadata, not transfer |
| r151--r160 PP2 dual service-side gate | same 16-rank GPT-2 BF16 `PP2/TP1/DP8`; 3 source + fresh epoch-3 restore + 5; node8 and node17 checkpoint services each failed in separate suites | live common-runtime transfer, exact reference loss/digest and same-layout recovery for Ada/Host/M1 | r156/r160 valid; M1 vs Ada suffix logical `-87.5%`, total application `-26.148%`; both failure sides exact; lag 3/non-1C1S; service-only, not compute-host fault |
| r161 PP2/common-runtime regression | CPU/analyzer/service admission and same-layout hash guard | reject config, capacity, fault-side, step-range, EXCLUDED input and unequal-layout recovery before mutation | `125 passed`; does not implement changed-layout resharding |
| r162--r166 PP2 10+5 formal gate | same layout; 15-step reference plus three checkpoint arms; ten suffixes; node8 checkpoint service loss; fresh epoch-10 restore + 5 | second fixed-`P` live M1 transfer/recovery result under identical detector/encoder/transport/service/recovery | r166 valid; M1 vs Ada suffix logical `-87.5%`, total application `-51.346%`; Host=M1 logical; planned max publisher `353.117 -> 77.328 MB`; lag `10/9/8`, final-drain/non-1C1S; timing single sample |
| r167 changed-layout CPU gate | materialized source `PP4/TP1/DP4` to target `PP2/TP1/DP8`; model/master/Adam moments/step; one storage host unavailable | FQN/role/global-slice assembly and pre-mutation changed-ownership validation | `16 passed`; exact canonical oracle and fail-closed negative cases; in-memory CPU only, no service/suffix/RNG/sampler/trainer restart |
| r168 owned-snapshot regression | shared SemanticFixed runtime CPU copy-elision; detector/objects/plan/wire/seal/recovery unchanged | focused correctness and mutation-lifetime regression | `91 passed`, one warning, exit zero; implementation gate only |
| r169 PP2 M1 owned diagnostic | node17/node8, GPT-2 BF16 `PP2/TP1/DP8`; M1 only; 10 source + node8 service loss + fresh epoch10 restore + 5 | measure shared-runtime copy reduction without changing grouping bytes | exact losses and 16 digests; logical suffix `3.263 GB`; externalize `-25.14%`, max drain `-7.28%`, source phase `-4.37%` vs r165; max PUT/RTO worse, lag 8; not a fair three-arm treatment |
| r170--r172 PP2 owned-runtime four-arm | r162 NoCheckpoint plus owned Ada/Host/M1; GPT-2 BF16 `PP2/TP1/DP8`; 10+5, node8 service loss | fair post-copy-elision M1 transfer/recovery gate under one detector/encoder/transport/fault contract | r172 valid; exact losses/digests; M1 vs Ada suffix logical `-87.5%`, total application `-51.346%`; Host=M1 logical, planned max publisher `-78.101%`; lag `10/9/8`, final-drain/non-1C1S; timing single sample |
| r173--r181 PP2 owned-runtime recovery repeats | three alternating Ada/Host/M1 blocks; same r162 reference, GPT-2 BF16 `PP2/TP1/DP8`; each 10+5 with node8 service loss | promote transfer/tail observation to run-level `n=3` while retaining exact recovery | 9/9 checkpoint runs exact loss/digest; M1 vs Ada suffix logical/total app/max PUT `-87.5/-51.346/-88.205%`; M1 vs Host max PUT/source/drain `-34.562/-10.352/-38.366%`; step wall worse than Ada and NoCheckpoint, lag `10/9/8`, not non-blocking |
| r205 deduplicated-base smoke | GPT-2 BF16 `PP4/TP1/DP4`; M1; 1 source + service loss + 1 continued | live gate for one shared heavy base per `(PP,TP)` plus rank overlays | runtime/recovery succeeds for all 16 ranks; post-hoc analyzer passes after stale base-id check fix; original FAILED retained, not a treatment |
| r206 GPT-6.33B deduplicated-base M1 | node17/node8, BF16 GPT-6.33B `PP4/TP1/DP4`; M1; base + 2 source, node8 service loss, fresh epoch-2 restore + 1 | remove r201/r202 capacity blocker and admit recovery-valid M1 treatment candidate | COMPLETE; exact final loss/digests; 4 shared + 16 overlays + control/service; base `88.589 GB`, suffix application `25.313 GB`, base+suffix `113.901 GB`; suffix max PUT `19.021 s`, max commit `25.147 s`, drain `65.836 s`, lag 2; Ada r207 and Host control still required |
| r207/r208 GPT-6.33B Ada capacity gate | same deduplicated-base AdaCheckStyleFixed contract as r206 | size one fair service limit before comparison | r207 FAILED at `170.550 GB` under 160 GiB while admitting suffix 2; no recovery/treatment claim. r208 retries with common 224 GiB capacity; Host/M1 must match it |
| r208 GPT-6.33B Ada admitted arm | node17/node8, 16-rank BF16 `PP4/TP1/DP4`; Ada; 224 GiB service; base + 2 source, node8 service loss, restore + 1 | fair primary baseline for same-capacity GPT suite | COMPLETE/exact recovery; suffix/total application `101.246/189.834 GB`; max suffix PUT/commit/drain/RTO `66.003/82.773/180.847/373.321 s`; mean source step `3.715 s`; Host r209 and same-capacity M1 pending |
| r209 GPT-6.33B Host admitted arm | same r208 contract; HostGlobalFixed; 224 GiB service | isolate publisher/load scheduling at the M1 byte volume | COMPLETE/exact recovery; suffix/total application `25.313/113.901 GB`; max suffix PUT/commit/drain/RTO `21.009/34.653/85.414/325.456 s`; mean source step `3.694 s`, lag 2; same-capacity M1 r210 pending |
| r210/r211 GPT-6.33B M1 strict suite | same r208/r209 contract; M1; r211 recomputes r200/r208/r209/r210 | close the second dense-model recovery-valid M1 gate | `valid=true`, exact loss/final digests; M1 suffix/total application `25.313/113.901 GB`, `-74.999/-40.000%` vs Ada; max PUT/commit/drain/RTO `19.346/25.718/66.854/317.163 s`, `-70.688/-68.930/-63.033/-15.043%`; source phase `-3.341%`; mean source step `4.113 s`, `+10.715%` vs Ada and not non-blocking; lag 2/final drain |
| r212--r219 DP-invariant sparse runtime calibration | node17/node8, GPT-2 BF16 `PP2/TP1/DP8`; r218 M1 and r219 Ada share detector/base/encoder/transport/node8-service-loss/recovery, 2+1 | remove duplicate foreground copy/hash work without weakening semantic/recovery checks | `35 passed`; r218/r219 COMPLETE and exact final digests; M1 vs Ada suffix logical/application `-87.5/-87.475%`, total application `-60.754%`, max PUT/commit/drain/RTO `-85.458/-79.440/-80.252/-22.163%`, source-step mean `-26.270%`; M1 remains `+12.968%` vs first-two-step NoCheckpoint mean, lag 2/final drain; r216/r217 remain diagnostic failures |
| r220/r221 TP-local semantic correction | r220 LLaMA `PP4/TP2/DP2` failure diagnostic; r221 GPT-2 `PP2/TP1/DP8` M1 2+1 recovery gate | reject cross-TP equality and bind each object to one PP/TP coordinate; reuse one M1 publisher snapshot for both services | r220 FAILED before restore on real epoch-2 cross-TP mismatch; r221 COMPLETE/exact recovery after root fix; `50 passed`; staged bytes `0.653 -> 0.326 GB` at unchanged transfer; r222 LLaMA pending |
| r222 corrected LLaMA M1 | node17/node8, LLaMA-8B BF16 `PP4/TP2/DP2`; TP-local DP objects, M1, 2+1, node8 service loss | first dense live gate after the final object-boundary and single-publisher correction | COMPLETE/exact r187 loss and all 16 digests; suffix logical/application `36.325/36.327 GB`, total application `163.464 GB`; max suffix PUT/commit/drain/RTO `29.748/33.871/69.231/242.896 s`; source-step mean `1.332 s`, about `+4.9%` vs matching NoCheckpoint; r223 Ada and Host control pending, so no M1 effect size yet |
| r223 corrected LLaMA Ada | exact r222 contract; AdaCheckStyleFixed, 2+1, node8 service loss | corrected primary baseline for LLaMA M1 attribution | COMPLETE/exact r187 loss and all 16 digests; Ada-to-M1 suffix/total application `72.652 -> 36.327/199.789 -> 163.464 GB` (`-49.999/-18.182%`); max suffix PUT/commit/drain/source phase `54.041/62.203/122.724/1225.926 -> 29.748/33.871/69.231/1170.703 s`; mean source step `1.514 -> 1.332 s`; Host r224 pending, lag 2/final drain |
| r224/r225 corrected LLaMA strict suite | r187 NoCheckpoint + corrected Ada/Host/M1, common r185 detector and 2+1 node8-service-loss contract | close TP-local corrected LLaMA attribution | `valid=true`, zero loss error and exact 16-rank digests; M1 vs Ada suffix/total application `-49.999/-18.182%`, max PUT/commit/drain/source/RTO `-44.953/-45.549/-43.588/-4.505/-3.161%`, source-step mean `-12.021%`; Host=M1 bytes and M1 planned max publisher `-49.965%`, but Host-to-M1 max PUT/step are `+6.517/+5.047%` while commit/drain improve `2.140/3.344%`; lag 2/final drain, no sustained nonblocking claim |
| r226 corrected GPT-6.33B M1 | node17/node8, BF16 `PP4/TP1/DP4, pp-dp`; corrected TP-local DP-invariant M1, deduplicated base, 2+1 and node8-service loss | admitted raw M1 arm before same-runtime Ada/Host attribution | COMPLETE; exact r200 loss and 16 final digests; base/suffix/total application `88.589/25.313/113.901 GB`; max suffix PUT/commit/drain/RTO `22.094/24.686/49.902/336.095 s`; source-step mean `1.2545 s` (`+11.8%` vs matching NoCheckpoint), lag 2/final drain; r227c Ada active, no single-arm effect claim |
| r227c corrected GPT-6.33B Ada | exact r226 contract; AdaCheckStyleFixed, 2+1 and node8-service loss; r227/r227b are preflight-only rejects | corrected primary Ada baseline before Host/load attribution | COMPLETE; exact r200 loss and 16 final digests; base/suffix/total application `88.589/101.246/189.834 GB`; max suffix PUT/commit/drain/RTO `70.527/79.042/158.259/417.325 s`; source-step mean `1.517 s`, lag 2/final drain; r228 Host active and all M1 deltas provisional until r229 |
| r228/r229 corrected GPT-6.33B strict suite | r200 NoCheckpoint + corrected Ada/Host/M1, common r186 detector and 2+1 node8-service-loss contract | close corrected second dense-model attribution and 16-rank stopping gate | `valid=true`, zero loss error and exact 16-rank digests; M1 vs Ada suffix/total application `-74.999/-40.000%`, max PUT/commit/drain/source/RTO/step/combined phase `-68.673/-68.769/-68.468/-2.967/-19.465/-17.304/-5.966%`; M1 is `+11.8%` vs NoCheckpoint, lag 2/final drain; Host=M1 logical bytes, M1 planned max publisher `-74.754%`, commit/drain improve but max PUT/step/source/combined/RTO worsen, so stable load timing is unproven; stop GPU work |

Historical frozen catalog analysis retains its recorded `fixed`,
`random-feasible`, `redundancy_max` and `profile_exact` labels. New experiments
use exactly AdaCheckStyle, GeminiStyleAdjacent, RandomFeasible and
RedundancyMax; greedy is retired and profile makespan is an M1 tie-break, not a
fifth policy. Formal common-runtime
attribution uses `AdaCheckStyleFixed` (primary fixed-worker baseline),
`HostGlobalFixed` (same host-global pool, deterministic publisher) and
`M1Object` (under fixed `P`, failure-domain-aware publisher/destination/load
schedule). M2 alone owns parallel-plan and logical-to-physical host search.
All three share detector, encoder, encoded objects, transport, service limits,
failure sets, sealed commit and recovery oracle.

The active suffix bytes are confirmed BF16 gradients plus runtime state used to
replay the shared AdamW transition. They are not a literal static parameter
(`p`) or optimizer (`o`) snapshot. The detector's redundancy reduction is held
common; AdaCheckStyleFixed-to-M1 measures pair-local duplicate protection that
the grouping/object publication policy avoids. HostGlobalFixed-to-M1 isolates
publisher load at approximately equal logical bytes. r120 additionally shows a
recovery-read tradeoff, so publication and restore traffic must be plotted
separately.

r131 is the first minimal live, recovery-valid M1 transfer-positive gate for one
GPT-2 layout; it does not close the repeated headline timing/non-blocking gate.
The detector, encoder, transport, service limit, failure set and
same-layout recovery are identical across checkpoint arms. M1 versus AdaCheck
reduces ten-suffix logical transfer exactly 75% and base-plus-suffix application
transfer 44.008%. HostGlobalFixed versus M1 remains the required load-balancing
ablation: their suffix logical bytes are equal and their application difference
is only 5,920 B framing, while observed suffix max PUT is 515.256 versus
425.787 ms. Because each arm has one ordered sample and all checkpoint arms
finish only through final drain with lag 10, this is not yet stable timing,
non-blocking or 1C1S evidence.

r133 is a source-only M1 stage-profile diagnostic, not a treatment row. Its
160 rank-epochs show mean capture/checksum/externalize/distributed-commit of
`73.6/350.9/197.5/2203.0 ms`; distributed commit is dominated by collective
waiting behind rank 12's serial two-service bundle commits. Maximum
enqueue-to-commit/publisher-queue tails are `24.025/21.217 s`, and final drain
is approximately `23.80 s`. The next gate is a minimal two-service concurrent
commit shared by all three arms, followed by repeated recovery-valid comparison.

r134 implements and validates the common-runtime concurrent two-service suffix
commit (`99 passed`). In the M1-only r133-to-r135 diagnostic, mean distributed
commit falls `2203.0 -> 1591.1 ms`, final drain falls `23.814 -> 17.120 s`, and
source phase falls `89.344 -> 73.977 s`, with identical logical bytes and final
digests. Max remote PUT worsens `471.384 -> 556.007 ms`; this is a throughput
result with a contention tradeoff, not a max-time result.

r139 is the post-change recovery-valid four-arm suite over r125/r137/r138/r136.
M1 versus AdaCheck retains 75% suffix logical and 44.008% total application-byte
reduction; its suffix max PUT is `430.046` versus `2466.345 ms`. HostGlobal and
M1 remain equal in logical bytes, while max PUT is `448.136` versus `430.046 ms`
in one ordered sample, too small and unrepeated for a stable load-balancing
claim. All final full-state digests and losses match; node8 service-only fault,
same layout and final-drain boundaries remain. All checkpoint arms fail strict
1C1S with lag `10/9/9`.
