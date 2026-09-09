# CHI3D Table 3 复现结果

BUDDI (CVPR 2024) Table 3，CHI3D pair s03。全部从 `datasets/original/CHI3D` 重建，
未使用作者发布的任何 auxiliary 数据（那些链接在 DATA.md 里标着 "coming soon"）。

单位 mm。格式：PER PERSON h0 / h1，JOINT = 两人拼接后做 Procrustes 的 PA-MPJPE。

| 行 | 论文 | 复现 | Δ | n |
|---|---|---|---|---|
| BEV (raw SMPL)  | 50 / 52 / 96 | **51.1 / 53.1 / 97.4** | +1.1 / +1.1 / +1.4 | 440 |
| BUDDI (gen.)    | 53 / 53 / 80 | **53.0 / 52.2 / 82.6** | +0.0 / −0.8 / +2.6 | 429 |
| BUDDI           | 48 / 47 / 68 | **50.3 / 47.1 / 69.9** | +2.3 / +0.1 / +1.9 | 429 |
| *BEV-init (SMPL-X 空间, 额外)* | — | *48.2 / 48.1 / 94.4* | — | 429 |

论文的核心结论在复现中保持：JOINT PA-MPJPE 上 BUDDI (69.9) < BUDDI gen. (82.6) < BEV (97.4)，
即学到的 proxemics 先验确实改善了两人的相对重建。

## 复现方式

```bash
./datasets/scripts/CHI3D/repro/run_all.sh prep          # Step 0-6
./datasets/scripts/CHI3D/repro/run_all.sh fit  bev_init|buddi_gen|buddi
./datasets/scripts/CHI3D/repro/run_all.sh eval bev_init|buddi_gen|buddi
python datasets/scripts/CHI3D/repro/s07_eval_bev_raw.py  # BEV 行，无需拟合
```

## 数据规模

s03：126 动作 × 4 相机 = 504 接触帧。ViTPose 检出 2 人 455 张、1 人 47 张、3 人 2 张。
`chi3d_items_for_eval`（要求四套检测对两人都非 −1）入选 **440**；拟合类的三行另有 11 个
接触帧未进入 `val_optimization.pkl`，故 n=429。

## 协议偏差（报数时必须一并说明）

1. **OpenPose 通道是 ViTPose 检测的副本**。仓库要求四套检测齐全，缺一套则 `process_chi3d.py:233-235`
   直接 FileNotFoundError、且 `chi3d_items_for_eval` 会丢弃样本。`use_hands: False` 下优化实际用
   vitpose，openpose 仅用于缺人回退和两个脚趾点拷贝，两者相同时均为空操作。但真 OpenPose 是独立
   信息源，那 47 张 ViTPose 只检出 1 人的图在论文里可能能进评测集。
2. **BEV 是本地重跑的**，与作者当年发布的估计存在版本差异。
3. **只抽了接触帧**（评测集本就只含接触帧，但与作者的全量抽帧流程不同）。
4. **`SMPLX_to_J14.pkl` 取自 SMPLer-X 公开镜像**（官方 essentials.zip 不含 joint_regressors/，
   已验证）。`J_regressor_h36m.npy` 取自 third-party/ROMP，且在 chi3d_eval.py 中是死代码。
5. **BEV-init 行是 SMPL-X 空间的 BEV**，与论文的 BEV 行不同量纲，仅作内部参考；
   与论文对照请用 `s07_eval_bev_raw.py` 的 raw SMPL 版。

## 途中修复 / 补齐的上游问题

| 问题 | 位置 | 处理 |
|---|---|---|
| `joint_regressors/` 官方从未发布，三个 eval 脚本 import 期引用 | chi3d_eval.py:33-39 等 | `s00_fix_essentials.py` |
| `correspondance.py` 的 CHI3D 分支跑完不 dump（只有 Hi4D 分支有） | correspondance.py:217-247 | `s03_build_correspondence.py` |
| `train_val_split.npz` 无任何脚本生成 | chi3d.py:59-61 | `s05_make_split.py` |
| `pcl_pcl_pairwise_distance` 默认 use_cuda=True，CPU 调用方崩溃 | distance.py:25 | **唯一一处修改上游代码**：改为跟随输入张量设备 |
| raw-BEV 评测分支未写完（取数函数定义了但全仓库无人调用） | evaluation/utils.py:109 | `s07_eval_bev_raw.py` |
| `ResultLogger.topkl` 无条件写 `evaluation/temp/`，该目录不存在 | evaluation/utils.py:367 | `run_all.sh` 里 mkdir |
| `ESSENTIALS_HOME` 未设置 → KeyError | chi3d_eval.py:28 | `run_all.sh` 里 export |

## 自训模型的对应行

[RESULTS_interx6.md](RESULTS_interx6.md) 记录了同一张表上的第四行——InterX 六视角 + CHI3D
混训模型（`interx6_chi3d_cond_bev`, epoch 323）的 **49.1 / 47.5 / 70.6**，n 同为 429，
与上表 BUDDI 官方权重那行的 50.3 / 47.1 / 69.9 相当。那份文档给出从 InterX 抽帧到
`chi3d_eval.py` 输出的完整链路，以及三条本表没有的协议偏差。
