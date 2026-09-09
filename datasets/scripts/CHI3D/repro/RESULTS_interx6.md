# InterX 六视角 + CHI3D 混训模型：CHI3D Table 3 数值的完整 pipeline

[RESULTS.md](RESULTS.md) 记录了论文 Table 3 三行基线怎么从 `datasets/original/CHI3D` 重建。
本文记录**第四行**——本轮自训模型 `interx6_chi3d_cond_bev` (epoch 323) 的
**49.1 / 47.5 / 70.6**——是怎么产生的，从 InterX 抽帧一直到 `chi3d_eval.py` 打印那三个数。

链路共五段：InterX 数据构建 → CHI3D 数据构建 → 混合训练 → checkpoint 选择 → 拟合与评测。
只有最后一段属于 CHI3D repro，前四段是新增的。

---

## 1. 结果

单位 mm。PER PERSON = 每人单独 Procrustes 对齐后的 PA-MPJPE；
JOINT = 两人 14+14=28 个关节拼成一组、**只对齐一次**（所以两人的相对朝向和间距误差不会被吃掉）。

| 行 | 论文 | 复现 | n |
|---|---|---|---|
| BEV (raw SMPL) | 50 / 52 / 96 | 51.1 / 53.1 / 97.4 | 440 |
| BUDDI (gen.) | 53 / 53 / 80 | 53.0 / 52.2 / 82.6 | 429 |
| BUDDI 官方权重 | 48 / 47 / 68 | 50.3 / 47.1 / 69.9 | 429 |
| **BUDDI ep323（本轮自训）** | — | **49.1 / 47.5 / 70.6** | 429 |

前三行取自 [RESULTS.md](RESULTS.md)。**官方权重那行是与 ep323 同日、同一份代码重新算的**，
数值与 RESULTS.md 一字不差（50.3 / 47.1 / 69.9）——这本身就是「两次评测的样本集完全相同」的强证据。

**结论只能表述为「与官方相当」**：JOINT 差 0.7mm（1%），PER PERSON 一列略优（49.1 vs 50.3）、
一列略差（47.5 vs 47.1）。429 个样本上这些差异都在噪声内，既不能说更好，也不能说更差。
作为量级参照：同一条流水线上 BEV 初始化是 97.4，两个先验都把它拉到 70 附近。

三档对齐强度一并列出，能看清 Procrustes 吃掉了什么：

| | est_mpjpe h0 / h1<br>（不对齐） | est_scale_mpjpe h0 / h1<br>（仅对齐尺度） | est_pa_mpjpe h0 / h1<br>（相似变换对齐） |
|---|---|---|---|
| 官方权重 | 135.6 / 133.4 | 66.1 / 67.0 | 50.3 / 47.1 |
| ep323 | 137.9 / 139.2 | 66.0 / 69.1 | 49.1 / 47.5 |

---

## 2. 逐动作分解（JOINT PA-MPJPE）

| 动作 | 官方权重 | ep323 | Δ |
|---|---:|---:|---:|
| Grab | 54.9 | 57.3 | +2.4 |
| Handshake | 55.1 | 55.0 | −0.1 |
| Hit | 69.2 | 66.3 | **−2.9** |
| HoldingHands | 55.9 | 56.0 | +0.1 |
| Hug | 108.0 | 111.8 | +3.8 |
| Kick | 94.8 | 102.0 | **+7.2** |
| Posing | 82.5 | 85.3 | +2.8 |
| Push | 67.5 | 65.6 | **−1.9** |

5 类更差、2 类更好、1 类持平。

> **这张表只能当观察，不能当证据。** s03 每个动作类别只有 15 个左右的动作实例 × 4 机位，
> 这个量级的分箱读数在本项目已经误导过两次（见 plan 里 V1 和 V1b 的记录）。
> 值得注意但需要更多数据才能确认的一点：退步最大的两类里 Hug 是接触最密集的类别，
> 与下游 FlickrCI3D 评测中「体表/betas 而非关节」的诊断方向一致。

---

## 3. 完整链路

### A. InterX 数据构建

```bash
# A1  抽帧 + 按演员组切分（stride 40 和 by-group split 都已是脚本默认值）
python llib/data/preprocess/utils/process_interx.py

# A2  六视角渲染 + BEV：读 processed.pkl，写 processed_mv.pkl
python llib/data/preprocess/utils/process_interx_bev.py \
  --vis-dir datasets/processed/InterX/diagnostics/bev_vis_mv

# A3  8cm 穿模排除清单（复用已有的 penetration_report.pkl，不重扫 GPU）
python llib/data/preprocess/utils/check_interx_penetration.py --exclude-threshold 0.08
```

| 阶段 | 产物 | 实测 |
|---|---|---|
| A1 抽帧 | `processed.pkl` | 4735 takes / 5659 互动段 / **49,111 帧**（stride 40 = 3fps） |
| A1 切分 | `train_val_split.npz` | 按 `take_id[:4]` 演员组切，按帧数贪心均衡 → 52 / 4 / 3 组 = 89.6 / 5.0 / 5.3%，两两无交集 |
| A2 六视角 | `processed_mv.pkl` | **294,666 视角**（49,111 × 恰好 6）/ 402MB / 约 12h |
| A3 穿模 | `penetration_exclude_list.pkl` | 10,244 帧（帧级 10.4%） |

六视角的相机：每帧抽一个基准 yaw，然后每 60° 一个，覆盖整个 360°；
pitch/roll/dist 每视角独立采。仿 CHI3D 的 4 机位做法（论文 p.6：
*"we use all camera views of the MoCap datasets, i.e. 4/8 cameras for CHI3D/Hi4D"*）。

`information_missing` 全量 23.9%（BEV 在合成灰模渲染上检测失败 18.3pp + 投影退化守卫拦下 5.6pp）。
这些视角保留在训练集里，[train_module.py:523-524](../../../../llib/methods/hhc_diffusion/train_module.py#L523-L524)
会把它们的 guidance 置零，当作无条件样本训练，而不是喂一份「自信但捏造」的零填充条件。

### B. CHI3D 数据构建

```bash
# B1  补齐训练侧受试者（s03 此前已备）
SUBJECTS="s02 s04" ./datasets/scripts/CHI3D/repro/run_all.sh prep

# B2  产出扩散训练用的 pkl
python datasets/scripts/CHI3D/repro/s08_make_diffusion_pkl.py

# B3  门禁（返回非零退出码即拒绝开训）
python datasets/scripts/CHI3D/repro/s09_verify_diffusion_pkl.py
```

| | contact 帧 | BEV 过滤后 | subjects | 4 机位分布 |
|---|---:|---:|---|---|
| `train_diffusion.pkl` | 980 | **911**（流失 7.0%） | `{s02: 451, s04: 460}` | 227 / 219 / 233 / 232 |
| `val_diffusion.pkl` | 492 | **458**（流失 6.9%） | `{s03: 458}` | 117 / 116 / 113 / 112 |

980 / 492 而不是 988 / 504，是 [chi3d.py:327-330](../../../../llib/data/preprocess/chi3d.py#L327-L330)
的 `frame_ids` 上游 bug——见第 4 节，那里有它对评测样本数的第二次影响。
`s09` 的期望值是照着同一段算术算的，所以不会把这件事误诊成「缺 subject」。

**本轮为接入 CHI3D 做的一处必要修改**：`chi3d.py` 原本只要 openpose / vitpose / vitposeplus / bev
四者任一为 −1 就置 `information_missing=True`，而训练侧据此把整条 guidance 置零——于是 BEV 完全正常、
只是某个关键点通道缺人的样本，也被当成无条件样本浪费掉。新增
[chi3d.py:428](../../../../llib/data/preprocess/chi3d.py#L428) 的 `bev_missing` 字段（只看 BEV），
[single.py:427](../../../../llib/data/single.py#L427) 改读它——CHI3D train 因此从 38 条降到 0 条，
救回 911 里的 4.2%。InterX 侧两者等价，不受影响。

### C. 混合训练

```bash
python llib/methods/hhc_diffusion/main.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml \
            llib/methods/hhc_diffusion/configs/config_buddi_interx_chi3d_cond_bev.yaml \
  --exp-opts logging.base_folder=demo/diffusion/training \
             logging.run=interx6_chi3d_cond_bev logging.logger=tensorboard
```

配置是分层覆盖，`config_buddi_v02_cond_bev.yaml` 未改动。日志实测：

```
InterX train  264,156 → 穿模过滤后 236,178      CHI3D train  911
合计 237,089        steps_per_epoch = 463
--- interx share per batch: 80.00%
--- chi3d  share per batch: 20.00%
```

`[0.8, 0.2]` 的 chi3d 配额与论文一致（batch 512 × 0.2 = 102 条/step，911 条训练集
→ 每 1000 步过 112 遍）。差别在于论文那 20% 旁边是 60% 的野外 Flickr 伪真值，这里旁边是合成渲染。

`val_names: ['chi3d', 'interx']` 的**顺序有语义**：
[diffusion_trainer.py:252-309](../../../../llib/training/diffusion_trainer.py#L252-L309)
每个验证集覆写一次 `ckpt_metric` 并返回最后一个，所以只有 **interx** 驱动 best-checkpoint 和 LR 退火；
CHI3D val 照评、进 TensorBoard，但不影响训练轨迹（这一点对第 5 节的偏差 7 很重要）。

| | 实测 |
|---|---|
| 起止 | `2026-08-24 12:01:46` → `2026-08-25 13:06:16` = **25h04m** |
| 停在 | **epoch 725 / 335,675 步**（= 725 × 463；配置写的是 1800 epoch，指标走平后手动停） |
| checkpoint | 242 个，每 3 epoch（1389 步）一存 |

停训依据：InterX 的 in-loop `pairwise_pa_mpjpe` 在最后 15 万步只改善 0.7%（0.0553 → 0.0549）。
下一节的完整采样评测确认了这个判断——实际上 164 epoch 之后就已经平了。

### D. checkpoint 选择

```bash
python llib/methods/hhc_diffusion/evaluation/eval_conditional.py \
  --exp-cfg demo/diffusion/training/interx6_chi3d_cond_bev/config.yaml \
  --checkpoint-dir demo/diffusion/training/interx6_chi3d_cond_bev/checkpoints \
  --dataset-name chi3d --dataset-split val --batch-size 64 \
  --sweep-n 10 --skip-steps 10 --output-folder demo/diffusion/eval/v4_chi3d
```

CHI3D val 只有 458 条而 `n_used = (len//bs)*bs`，所以用 `--batch-size 64`（448/458）；
用 256 只会评 256 条。扫了 11 个 checkpoint，同时跑 guidance 置空对照：

| epoch | 条件 pa_h0 | pa_h1 | **pa_h0h1** | **pcc@0.10** | 置空 pa_h0h1 | 置空 pcc@0.10 |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 87.9 | 140.6 | 193.6 | 0.059 | 288.2 | 0.039 |
| 83 | 58.2 | 56.7 | 86.7 | 0.251 | 253.3 | 0.011 |
| 164 | 54.9 | 56.4 | 84.3 | 0.333 | 257.5 | 0.038 |
| 242 | 55.4 | 55.8 | 84.5 | 0.351 | 265.0 | 0.037 |
| **323 ← 选定** | 54.5 | 55.6 | 84.2 | **0.355** | 262.2 | 0.021 |
| 404 | 54.0 | 55.2 | **83.9** | 0.326 | 264.1 | 0.040 |
| 485 | 54.2 | 55.4 | 84.3 | 0.352 | 264.1 | 0.027 |
| 563 | 54.3 | 55.2 | 84.1 | 0.350 | 265.2 | 0.040 |
| 644 | 54.1 | 55.2 | 84.2 | 0.336 | 265.7 | 0.041 |
| 686 | 54.3 | 55.3 | 84.3 | 0.349 | 265.3 | 0.040 |
| 725 | 54.1 | 55.2 | 84.0 | 0.349 | 264.4 | 0.040 |

**条件确实在起作用，幅度很大**：`pa_h0h1` 条件 84.0 vs 置空 264.4（3.1 倍），
`pcc@0.10` 0.349 vs 0.040（8.7 倍）。置空对照几乎完全崩掉，说明模型真在依赖 BEV 条件，
不是靠记忆先验蒙。

选定：

```
demo/diffusion/training/interx6_chi3d_cond_bev/checkpoints/2026_08_24-23_11_43__0000000323__0000000463__32.32.pt
```

**这是一个任意的平局裁决，不是「它更好」。** epoch ≥164 的 9 个 checkpoint：

```
pa_h0h1   范围 83.92–84.48   极差 0.57mm   std 0.16
pcc@0.10  范围 0.326–0.355   极差 0.029    std 0.0095
pcc 基于 n=105 条有接触标注的样本，比例 0.345 的标准误 = sqrt(0.345*0.655/105) ≈ 0.046
                                                   → 极差 0.029 只有 0.62 个标准误
```

`pcc` 的 checkpoint 间差异完全落在抽样噪声里，拿它排序等于在拟合噪声。选 323 的实际理由只是
「它在两个指标上都不是最差的」。**推论：换 404 或 725 去跑本文的拟合评测，结果差别应当在噪声内。**

> 注意 `eval_conditional.py` 的 54.1 / 55.2 / 84.0 **不能**和第 1 节的 Table 3 数字直接比：
> 前者默认 `--joints model`（25 个 OpenPose 关节），后者用 J14（14 个 LSP 关节，经 `SMPLX_to_J14.pkl`）。

### E. 拟合与评测

```bash
R=datasets/scripts/CHI3D/repro
CKPT=demo/diffusion/training/interx6_chi3d_cond_bev/checkpoints/2026_08_24-23_11_43__0000000323__0000000463__32.32.pt \
CKPT_CFG=$R/cfg/interx6_chi3d_ep323_bs1.yaml \
BASE_FOLDER=demo/optimization/chi3d_interx6 \
  $R/run_all.sh fit buddi

BASE_FOLDER=demo/optimization/chi3d_interx6 $R/run_all.sh eval buddi
```

拟合配置是 [buddi_cond_bev.yaml](../../../../llib/methods/hhcs_optimization/configs/buddi_cond_bev.yaml)，
**与官方那次逐项相同，唯一变量是 `CKPT` / `CKPT_CFG`**（两次运行落盘的 `config.yaml` 已逐项比对确认）。
关键设置：

```
max_iters [100, 100]        两阶段各 100 iter      adam lr 0.01
sds_type fixed              sds_t_range [25, 75]   use_gt_contact_map False
keypoint2d          [0.02,  0.02]      init_pose               [200.0, 200.0]
diffusion_prior_pose[100.0, 100.0]     diffusion_prior_transl  [10000.0, 10000.0]
diffusion_prior_shape[10.0,   0.0]     hhc_contact_general     [0.0,   10.0]
```

拟合 458 条，`2026-08-25 13:49:33 → 17:07:13` = **3h18m**（约 26s/张）。

**关于 `CKPT_CFG` 为什么是 `cfg/` 下的一份副本**：它与
`demo/diffusion/training/interx6_chi3d_cond_bev/config.yaml` **逐行相同，仅 `batch_size: 512 → 1`**。
[main.py:160-163](../../../../llib/methods/hhcs_optimization/main.py#L160-L163) 会拿这个
`batch_size` 去给 SMPL-X 分配缓冲区，而拟合一次只处理一张图——直接传训练 config 会白占约 8GB 显存，
GPU 上有别的任务时会 OOM。这份副本只为省显存，不影响任何数值。

---

## 4. n 是怎么从 504 变成 429 的

```
504   s03 的 126 个标注动作 × 4 机位
 ↓    chi3d_items_for_eval：要求 openpose / vitpose / vitposeplus / bev
      四路检测对两人都非 −1（evaluation/utils.py:26-58）
440   评测循环实际遍历的 (动作, 机位) 组合
 ↓    再要求拟合结果文件存在（缺的打印 ITEM MISSING，共 11 条）
429   两次评测实际用的 n
```

**11 条 `ITEM MISSING` 的来源已查实**：全部来自三个序列 `s03 Grab 10` / `Push 3` / `Posing 2`，
而这三个正是 [chi3d.py:327-330](../../../../llib/data/preprocess/chi3d.py#L327-L330)
`frame_ids` 上游 bug 命中的序列。实跑同一段算术确认：

```
s03 Grab 10    fr_id=81  start_fr=81
s03 Push 3     fr_id=44  start_fr=48
s03 Posing 2   fr_id=53  start_fr=56
```

`start_fr >= fr_id` 时第一段 `arange` 为空、第二段的 `[1:]` 又切掉 `fr_id`，兜底分支因第二段非空而不触发
→ 标注的接触帧根本没进 `frame_ids`，整个序列一帧不剩，于是也从未进 `val_optimization.pkl`，拟合无产出。

3 × 4 = 12 而只打印 11，是因为 `Posing 2 / 60457274` 在更前面的四通道检测过滤就被丢掉了，
压根没进 `ITEMS_TO_EVAL`。

**另有一个容易混淆的口径**：拟合阶段跑的是 **458 条**（`val_optimization.pkl` 3648 条 → 458），
ep323 和官方各产出 458 个结果 `.pkl`，两边完全相同。458（拟合覆盖）和 429（评测使用）
是两个不同的数，不要混用——本文写作过程中曾两次算错 n（先用 493，又用 458），
正确做法是直接执行 `chi3d_items_for_eval` 复现脚本自身的取样路径。

---

## 5. 协议偏差（报数时必须一并说明）

**继承 [RESULTS.md](RESULTS.md) 的全部 5 条**：OpenPose 通道是 ViTPose 检测的副本 /
BEV 是本地重跑的 / 只抽了接触帧 / `SMPLX_to_J14.pkl` 取自 SMPLer-X 公开镜像 /
BEV-init 行与论文 BEV 行量纲不同。以下三条是本报告独有的：

6. **checkpoint 是在 s03 上选的——模型选择泄漏。**
   s03 从未进入训练集（train = s02 + s04），所以不是训练泄漏；但第 3.D 节的 checkpoint 扫描
   用的正是 s03，而第 1 节又在 s03 上报数，形式上是在测试集上选点。
   缓解证据是 3.D 那组统计量：epoch ≥164 的 9 个 checkpoint 在两个指标上都落在一个标准误内，
   选择几乎没携带信息。但这一条必须写出来，不能省。

7. **s03 同时是训练时的验证集**（`val_diffusion.pkl`，458 条）。
   它每 3 个 epoch 被评一次并写进 TensorBoard，但 `val_names: ['chi3d', 'interx']` 的最后一个是
   interx，按 [diffusion_trainer.py:252-309](../../../../llib/training/diffusion_trainer.py#L252-L309)
   只有最后一个驱动 best-checkpoint 与 LR 退火——所以 s03 **没有**影响训练轨迹。

8. **本文的评测是准同域测试，不是域外测试。**
   训练集里的 CHI3D s02/s04 与评测的 s03 是同一个实验室、同一套 4 机位、同样的采集流程；
   s03 是没见过的**受试者**，不是没见过的**域**。模型训练时吃了 911 条 CHI3D 样本
   （每 batch 20%、每 1000 步过 112 遍），学会了这个实验室的成像分布。

   > **这一条决定了本文的数字能推出什么、不能推出什么。** 同一个 ep323 模型在 FlickrCI3D
   > 野外照片上的下游接触评测里胜率只有 49.6%（二项检验 z = −0.09，等同于「不用扩散先验」），
   > 而官方权重是 65.4%（z = +3.56）。**「在 CHI3D 上追平官方」不蕴含「在真实照片上可用」**，
   > 引用本文数字时请连同这一句一起引用。

---

## 7. 补充实验：这个落差是「接触能力」还是「域」

上面第 1 节测的是关节（PA-MPJPE），V6 测的是体表接触（dist_on_gt）——两个评测在**指标**和**域**
两个轴上同时不同，所以单凭它们无法判断 ep323 到底差在哪。`chi3d_eval.py:226-232` 本来有接触指标，
但那段是注释掉的，所以 CHI3D 上从未测过接触。

补上缺的那一格：在 CHI3D s03 上用**与 V6 完全相同的指标定义**
（`dist_on_gt = ContactMap.get_full_heatmap(v0,v1)[gt_cmap].mean()`，标注接触区域对的实际距离）
重新评一遍两个 run 已有的 458 条拟合结果，无需重新拟合。

**CHI3D s03，n=429（与第 1 节同一批样本）：**

| run | dist_on_gt 均值 | 中位 | pcc@0.10 | pcc@0.20 |
|---|---:|---:|---:|---:|
| BUDDI 官方权重 | 195.8 | 128.4 | 0.383 | 0.703 |
| **BUDDI ep323** | **196.1** | **126.7** | **0.399** | 0.673 |
| BUDDI (gen.)（不优化，参照） | 237.8 | 178.6 | 0.214 | 0.566 |

逐图对比：ep323 胜过官方 **210/429 = 49.0%**（二项检验 z = −0.43），**统计上打平**。

三个测点连起来：

| | 域 | 指标 | ep323 vs 官方 |
|---|---|---|---|
| 第 1 节 | CHI3D 实验室 | 关节 PA-MPJPE | 打平（70.6 vs 69.9） |
| 本节 | CHI3D 实验室 | **接触 dist_on_gt** | 打平（196.1 vs 195.8） |
| V6 | FlickrCI3D 野外 | 接触 dist_on_gt | 惨败（胜率 49.6% vs 65.4%） |

**指标轴被控制住之后结论是清楚的：同一个接触指标，实验室里打平、野外里失效。
落差来自域，不是来自接触能力。**

因此对本轮模型的准确评价是：**在它见过的域里它是一个真实可用的先验**——用纯合成渲染 +
911 条 CHI3D 就在关节和接触两项上都追平了用 60% 野外照片训练的官方权重。唯一不能做的是把它
当官方权重的替代品用在野外照片上。

> 读数注意：官方在 CHI3D 上是 195.8、在 Flickr 上是 182.8，**这两个数不能跨数据集比**——
> 两边的接触标注密度和难度分布完全不同。只有同一数据集内部的对比有意义。

---

## 8. 各段的重跑入口

每段都可独立重跑，产物存在即跳过。

| 段 | 入口 |
|---|---|
| A1 抽帧 + 切分 | `python llib/data/preprocess/utils/process_interx.py`（`--split-only` 只重切分） |
| A2 六视角 + BEV | `python llib/data/preprocess/utils/process_interx_bev.py`（读 `--in-fn`，写 `--out-fn`） |
| A3 穿模清单 | `python llib/data/preprocess/utils/check_interx_penetration.py --exclude-threshold 0.08` |
| B1 CHI3D 预处理 | `SUBJECTS="s02 s04" ./datasets/scripts/CHI3D/repro/run_all.sh prep` |
| B2 训练用 pkl | `python datasets/scripts/CHI3D/repro/s08_make_diffusion_pkl.py` |
| B3 门禁 | `python datasets/scripts/CHI3D/repro/s09_verify_diffusion_pkl.py` |
| C 训练 | 见 3.C |
| D checkpoint 选择 | 见 3.D |
| E 拟合 + 评测 | 见 3.E |

前置条件（与 RESULTS.md 相同，`run_all.sh` 已代劳）：`conda activate buddi`、
`export PYTHONPATH=$PWD`、`export ESSENTIALS_HOME=$PWD/essentials`、
`mkdir -p llib/methods/hhcs_optimization/evaluation/temp`。

只重跑评测（拟合结果已在盘上）：

```bash
BASE_FOLDER=demo/optimization/chi3d_interx6 ./datasets/scripts/CHI3D/repro/run_all.sh eval buddi
```

应当打印 `est_pa_mpjpe_h0: 49.1` / `est_pa_mpjpe_h1: 47.5` / `est_pa_mpjpe_h0h1: 70.6`。
