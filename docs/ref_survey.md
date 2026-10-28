论文调研记录：
1. 所有实验要么呼应前面的创新点或者design特殊的地方，要么证明比同类研究优秀
2. 如果只考虑 ckpt size/time/traffic，大部分研究都做的是 microbenchmark，只有考虑 training accuracy 才有必要跑端到端(CheckFreq)
3. 现有 checkpoint 工作对实验时人为注入 failure 的方法并不统一，大部分是模拟 x 个rank/node失效，然后计算当前 ckpt group能不能覆盖掉；GEMINI AdaCheck 则是直接用数学计算求得 recovery wasted time
   

论文/研究
系统设计的目标
Ckpt grouping 
Failure 

实验 baseline
实验软硬件配置
实验model与并行配置

实验指标
实验细节
创新点
问题或借鉴

--------------------------------------------

AdaCheck
(FAST'26)

适应多样化并行策略，识别并消除冗余来减小ckpt大小


以worker为单位划分，“根据模型并行性来划分检查点组，确保组内的worker之间共享相似的计算图，从而较多地重叠检查点通信和模型计算”
故障以worker/rank为单位，故障场景是同一个检查点组内worker失效；故障恢复时从CPU加载检查点，SSD恢复只作为备用不涉及实验

- CheckFreq
- GEMINI
- No Ckpt

8*A800, 80G, 32GPUs, training bandwidth 800Gbps;
4*3090, 24G, 128GPUs, training bandwidth 100Gbps;
CUDA 11.7, NCCL 2.14.3, PyTorch 2.0.0

- A800: 
  - LLaMA-7B(ZeRO-3 + ZeRO-1 + nnScaler-gen)
  - LLaMA-30B(ZeRO-3 + MiCS + nnScaler-gen)
  - Deepseek-V2-Lite(EP+MP  +   EP+ZeRO-1   + nnScaler-gen);
- 3090: 
  - GPT-1.4B(ZeRO-3 + ZeRO-1 + nnScaler-gen)
  - GPT-7B(ZeRO-3 + MiCS + nnScaler-gen),
  - GPT-MoE(EP+MP  +   EP+ZeRO-1   + nnScaler-gen)

- Ckpt size
- Ckpt frequency
- Recovery wasted time
- Training time
- Effective Training throughput under different failure rate

没有说明实验中注入故障的方式；
未说明实际检查点组大小

Redundancy detector, 极大地减小ckpt size


GEMINI 
(SOSP'23)

in-memory checkpoint，并解决两个问题：最大化recovery概率、最小化检查点传输流量对训练的干扰
以机器node为单位划分，提出group + ring分组方式并且数学公式化；然而实际上分组算法是按照序号分，未考虑任何冗余or并行

机器级别故障，但是也提出 software/hardware failure；ckpt组内failure通过跨机CPU副本恢复；整组failure退化为SSD
- 两个不同保存频率的SSD检查点系统
-  No ckpt

8*A100, 40G, 128 GPUs，training bandwidth 400Gbps；
8*V100, 32G, ?, training bandwidth 100Gbps;
CUDA-11.6, DeepSpeed-v0.7.3, PyTorch-1.13, nccl-v2.14.3

GPT-2 10/20/40/100B；
BERT 40/100B;
RoBERTa 40/100B;
DeepSpeed ZeRO

- ckpt frequency
- Recovery wasted time/ overhead of failure recovery
- Training time
- Effective training time under different failure rate
- Ckpt traffic interleaving(ablation)

Ckpt grouping的目的是最大化recovery 概率，后续实验都在这个前提下进行，所以grouping并不作为实验指标
replica placement optimization + communication scheduling
group不固定，而是可优化变量；
可进一步加入parallel topology、redundancy 与fault-domain transfer cost

Capricorn
(IEEE CLUSTER)
专门面向MoE模型的动态性，最大程度掩盖ckpt traffic，降低in-memory checkpoint开销
声称 CPU 检查点，ckpt 组划分和 GEMINI 相似
没有具体说明如何注入故障，可能因为论文重点不在恢复概率？and 推测是 rank 级别故障

- TorchSnapshot
- GEMINI

4*3090, 24G, ?, training bandwidth 100Gbps;
CUDA 11.8, cuDNN 8.7.0, Pytorch 2.2.0, and NCCL 2.19.3
QWen/Deepseek 8/16/32 experts routed;
ZeRO-1
- Training thruoughput
- Training time
- Ckpt frequency



竟然早早提出了ckpt四大指标：accuracy, storage, bandwidth, frequency

ECCHECK
(ICDCS'25)

CPU内存检查点，减少复制，极可靠
将 n 个节点划分为 k 个数据节点和 m 个校验节点，用 Cauchy Reed-Solomon 纠删码（GF(2) 上纯 XOR 实现）对内存检查点分布式编码，可容忍任意 m 个并发节点故障
只划分数据节点和校验节点，不太相关，因为此研究并不利用参数冗余，也不会建立检查点副本
机器级别故障；痛击GEMINI弱点：同一组内所有机器故障

- CheckFreq
- GEMINI
- No Ckpt

?*A100, 40G, 16 GPUs；training bandwidth 100Gbps;
?*V100, 32G, 32GPUs

GPT-2/BERT/T5，1.6/5.3/20B
TP=4、PP=4; Megatron-LM
- Ckpt time
- Training time
- Recovery time
- Recovery rate under different failures




Fault-Tolerant Hybrid-Parallel xxx(REFT)(arXiv)

hybrid-parallel下低开销in-memory checkpoint与快速恢复，很像把 EECHECK 和 GEMINI 缝起来

指定处于相同 PP stage为一组

机器级别故障
- GEMINI
- No ckpt
512×V100, 16G；training bandwidth 100Gbps p2p;

Llama-2 1.3/2.7/6.7/34B；hybrid DP+TP+PP
DeepSpeed

- Training time
- Ckpt frequency
- Training throughput
- Recovery/load time


hierarchical async snapshot + redundancy 
冗余“层级“会直接影响恢复代价;
这个研究的limitation里提到了根据GPU-CPU通信拓扑优化检查点传输

FlowCheck 
(EuroSys'25)
解耦checkpoint与training，尽量不阻塞训练
通过从正常训练的网络流量中提取完整的梯度信息来更新检查点。FlowCheck部署了一个流量镜像网络来支持这种设计
无；DP group用于traffic redundancy，不是checkpoint replica group；checkpoint存于CPU checkpoint server
Gradient loss (?) / training node failure；从checkpoint server恢复

- PyTorch ckpt
- CheckFreq

4*A100, 40G；32 GPUs；training bandwidth 100Gbps NIC；checkpoint server 2×100Gbps NIC

BERT 110M；RoBERTa 330M；
GPT-3 1.3B；Llama2-7B；
DP / DP-MP
- Training time
- ckpt save time
- effective training time
- recovery rate





Malleus
(SIGMOD'25)

提出了一个混合并行训练框架，动态straggler下自适应调整hybrid parallelism

无；GPU group = TP group，用于训练并行而非checkpoint replica
故障以worker/rank为单位

- Megatron-LM
- DeepSpeed
8*A800, 80G; 64GPUs；training bandwidth 400GB/s NVLink；200GB/s InfiniBand (??)
LLaMA-32B/70B/110B；32/64/64 GPUs；DP+TP+PP；DeepSpeed + ZeRO-3
- training time
- Step time
- Overhead xxx (unrelated)


这个工作考虑了grouping带来的收益

Check-N-Run 
(NSDI'22)

差分checkpoint + 量化，工业级别的推荐模型;
检查点保存和训练解耦

没有涉及分组

机器级别故障，只有SSD存储
PyTorch/epoch checkpoint；差分与量化方案
16*V100，猜测在128/256GPUs; 自研实验框架，没有更多软硬件说明
"Each GPU is able to communicate directly with GPUs on a different node through a dedicated RoCE NIC, without involving a host CPU"
推荐模型；embedding MP + MLP DP
没有具体讨论模型及并行
- write bandwidth/occupatino
- Ckpt size
- Ckpt frequency
- Training stall(ablation, decoupled)
- Ckpt accuracy(for quanitization)





Just-In-Time Checkpointing 
(EuroSys'24)
按失败触发checkpoint，避免周期性checkpoint开销；恢复时少量replay
无
training failure；failure后即时checkpoint并恢复
periodic checkpointing


recovery time；steady-state overhead；replayed work
恢复开销降至秒级；稳态训练开销接近0

可作为极端故障场景下的补充机制？而不是单纯SSD

ByteCheckpoint
(NSDI'25)

让checkpoint表示与训练parallelism解耦；支持parallelism重配置

无
重点是GPU数量/parallelism变化下的恢复与重分片，不是副本故障
DCP；MCP
A100 80G；H800 80G；InfiniBand
vDiT 4B：32→64 / 128→64 GPU，ZeRO-2；tGPT 70B：TP=4、PP=8、DP=75/150（2400/4800 GPU）

TBlock；TSave；TLoad；TReshard；ETTR




CheckFreq
(FAST'21)
高频checkpoint；首次提出差分检查点
无
preemption / process-node failure等
epoch-boundary checkpoint；BERT为固定iteration baseline
1080Ti 11G或V100 32G；8 GPUs/server
ResNet18/50；ResNext101；DenseNet121；VGG16；InceptionV3；BERT-Large；1GPU/8GPU
runtime overhead；checkpoint frequency；recovery time；end-to-end training time




UCP
(NIPS'24)
checkpoint与训练parallelism及硬件配置解耦；支持大规模parallelism重配置

无
hardware/resource change；scale-in/out与parallelism重配置
distributed checkpoint转换/重分片方案
64 A100→48 A100
7B–1T；GPT/LLaMA/sparse MoE/BLOOM-176B/Phi-3.5-MoE；DP/TP/PP/SP/ZeRO组合
reconfiguration time；coverage；correctness；save overhead
1T模型重配置<3 min；重配置时间最高降低约14–257×

可以采纳

