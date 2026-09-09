# BUDDI 自训模型评测结论

两个基准、三个权重的横向对比。所有数字均为本机实测，评测代码与配置在各表下注明。

---

## 0. 被评测的三个权重

| 代号 | checkpoint | 训练数据 |
|---|---|---|
| **官方** | `essentials/buddi/buddi_cond_bev.pt` | 论文配比：FlickrCI3D 伪真值 60% / CHI3D 20% / Hi4D 20% |
| **camfix**<br>（第二代，单视角） | `demo/diffusion/training/interx_cond_bev_camfix/checkpoints/`<br>`2026_08_16-15_59_09__0000003509__0000000167__26.58.pt` | InterX 100%，**单视角**、stride 20、无穿模过滤 |
| **interx6**<br>（第三代，六视角） | `demo/diffusion/training/interx6_chi3d_cond_bev/checkpoints/`<br>`2026_08_24-23_11_43__0000000323__0000000463__32.32.pt` | InterX **六视角** 80% / CHI3D 20%，stride 40、8cm 穿模过滤 |

三者在下游全部走**同一套优化流程**（`buddi_cond_bev.yaml`，`max_iters [100,100]`、
`sds_t_range [25,75]`、`keypoint2d 0.02` / `diffusion_prior_pose 100` / `diffusion_prior_transl 10000`），
**唯一变量是权重文件**。

---

## 1. CHI3D（pair s03）—— 论文 Table 3

PA-MPJPE 单位 mm，越低越好；PCC 单位 %，越高越好（半径 r 与论文 Table 1 的 5/10/15/20 同刻度）。

| 行 | 论文<br>PER PERSON / JOINT | 复现<br>PER PERSON | 复现<br>JOINT | PCC@5 | PCC@10 | PCC@15 | PCC@20 | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| BEV (raw SMPL) | 50 / 52 · 96 | 51.1 / 53.1 | 97.4 | — | — | — | — | 440 |
| BUDDI (gen.) · 官方权重 | 53 / 53 · 80 | 53.0 / 52.2 | 82.6 | 8.1 | 21.4 | 39.5 | 56.6 | 429 |
| **BUDDI (gen.) · interx6 ep323** | — | **49.9 / 51.8** | **80.6** | 7.2 | 23.5 | 39.9 | 53.9 | 429 |
| **BUDDI 官方权重** | 48 / 47 · 68 | **50.3 / 47.1** | **69.9** | **13.5** | 38.3 | 56.7 | **70.3** | 429 |
| **BUDDI · interx6 ep323** | — | **49.1 / 47.5** | **70.6** | **13.6** | **39.9** | **57.7** | 67.3 | 429 |
| **BUDDI · camfix ep3509** | — | 58.5 / 59.0 | **99.7** | 8.8 | 30.2 | 48.7 | 60.7 | 429 |

**读数**

- **上一代单视角模型（camfix）在实验室域是有害的**：JOINT PA-MPJPE **99.7**，比它起步的 BEV
  初始化 **97.4 还差 2.3mm**；PER PERSON 58.5 / 59.0 也显著劣于 BEV 的 51.1 / 53.1。
  两代自训模型之间相差 **29.1mm**（99.7 → 70.6），是本轮最显著的单一改善。
  **但两次训练之间同时变动了五项设置**（六视角、混入 CHI3D、8cm 穿模过滤、按演员组重切分、
  LR 退火），因此这 29.1mm 应归于该组合而非其中任何单独一项。
- **PA-MPJPE 上 interx6 与官方权重打平**：JOINT 差 0.7mm（1%），PER PERSON 一列略优（49.1 vs 50.3）
  一列略差（47.5 vs 47.1）。429 个样本上都在噪声内。
- **PCC 上同样打平**：@5 / @10 / @15 三档 interx6 略高，@20 略低。逐图对比 interx6 胜过官方
  210/429 = **49.0%**，二项检验 z = −0.43。
- **先验贡献与图像证据贡献现在可以拆开**（JOINT PA-MPJPE，`gen.` 行是只生成不优化）：

  ```
                  BEV 初始化  →  纯生成   →  生成 + 优化
  官方权重           97.4    →   82.6    →   69.9      先验 −14.8mm，图像证据 −12.7mm
  interx6 ep323      97.4    →   80.6    →   70.6      先验 −16.8mm，图像证据 −10.0mm
  ```

  两个先验各自贡献的量级相当，interx6 在纯生成这一段甚至略多。**`gen.` 行连关键点都没看**，
  所以这是「联合分布先验确实学到了东西」的一个不依赖优化流程的独立证据。

**口径说明（引用时必须一并说明）**

1. **论文 Table 3 没有 PCC 列**，只有 PER PERSON / JOINT PA-MPJPE。上表的 PCC 是本报告新增，
   用的是与第 2 节 FlickrCI3D 评测**完全相同**的定义
   （`ContactMap.get_full_heatmap(v0,v1)[gt_contact_map]`，取标注接触区域对的实际距离），
   所以两个基准之间的 PCC 可以横向比较趋势，但**不能与论文 Table 3 对照**（那里没有这一列）。
2. **BEV 行没有 PCC**：BEV 输出是 SMPL（6890 顶点），而 `ContactMap` 的区域→顶点映射是
   SMPL-X（10475 顶点），不做 SMPL→SMPL-X 转换算不了。上游代码里那句注释
   （`chi3d_eval.py:226`「not implemented for BEV because bev estimate is SMPL」）说的就是这件事。
3. **n 的两个口径**：BEV 行 440 是 `chi3d_items_for_eval` 的输出；其余行 429 是再扣掉 11 条
   拟合结果缺失的样本（那 11 条来自 `s03 Grab 10` / `Push 3` / `Posing 2` 三个序列，
   是 `chi3d.py:327-330` 的上游 `frame_ids` bug 导致其接触帧从未进入数据集）。
4. **`BUDDI (gen.)` 两行都是每张图单次随机抽样。** 扩散生成从 `torch.randn` 起步，重跑不会得到
   相同数字。两行的 `dbs`（采样 batch size）都是 512、都取第 0 个样本，协议一致、对比公平，
   但**不要把 1–2mm 的差异当作模型能力差异**——那个量级在采样噪声里。相比之下 `BUDDI` 行由
   200 步优化收敛决定，重跑稳得多。
   用该 run **最新**的 checkpoint（epoch 725）重跑同一行得到 49.8 / 51.5 / **80.5**、
   PCC 7.2 / 22.2 / 40.0 / 54.2，与 epoch 323 全在噪声内——**本轮结论对 checkpoint 选择不敏感**。
5. 官方权重那一行是与 interx6 **同日、同一份代码**重算的，数值与
   [RESULTS.md](datasets/scripts/CHI3D/repro/RESULTS.md) 一字不差，可证两次评测样本集相同。
6. 其余协议偏差（OpenPose 通道是 ViTPose 副本、BEV 本地重跑、`SMPLX_to_J14.pkl` 取自第三方镜像等）
   见 [RESULTS.md](datasets/scripts/CHI3D/repro/RESULTS.md)；自训模型特有的三条
   （checkpoint 在 s03 上选、s03 同时是训练时的 val、CHI3D 是准同域测试）见
   [RESULTS_interx6.md](datasets/scripts/CHI3D/repro/RESULTS_interx6.md) 第 5 节。

---

## 2. FlickrCI3D Signatures —— 野外真实照片的接触评测

133 张公共交集图片，接触真值来自 `interaction_contact_signature.json`。
`dist_on_gt` = 标注中「应该接触」的区域对在重建结果上的实际距离（mm，越低越好）。

| run | 权重 | PCC@5 | PCC@10 | PCC@15 | PCC@20 | dist_on_gt<br>均值 / 中位 | 逐图胜过<br>baseline | z |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **baseline** | 不用扩散先验 | 9.6 | 26.8 | 41.9 | 51.6 | 255.5 / 204.6 | — | — |
| **官方权重** | `buddi_cond_bev.pt` | **15.9** | **37.0** | **53.2** | **66.5** | **182.8 / 160.6** | **87/133 = 65.4%** | **+3.56** |
| **camfix ep3509** | `...__0000003509__..._26.58.pt` | 9.5 | 26.2 | 40.5 | 51.0 | 249.1 / 207.8 | 66/133 = 49.6% | −0.09 |
| **interx6 ep323** | `...__0000000323__..._32.32.pt` | 9.7 | 28.7 | 41.0 | 52.5 | 245.9 / 203.7 | 66/133 = 49.6% | −0.09 |
| *论文 Table 1 的 BUDDI 行（参考）* | — | *19* | *44* | *62* | *73* | — | — | — |

**读数**

- **官方权重在四个半径上全面高出 baseline 约 10–15 个百分点**，逐图胜率 65.4%（z=+3.56，显著）。
- **两个自训模型在聚合上都与 baseline 无法区分**：PCC 四档全部落在 baseline 的 ±2pp 内，
  逐图胜率 49.6%（z=−0.09，掷硬币）。
- **但聚合掩盖了两个方向相反的效果。** 按 `dist_on_gt` 分档（133 张）：

  | | <100mm 优秀 | 100–200 尚可 | 200–400 明显不对 | **>400 灾难** |
  |---|---:|---:|---:|---:|
  | 无先验 | 31 | 34 | 46 | **22** |
  | 官方 | **39** | **51** | 35 | **8** |
  | interx6 ep323 | 23 | 41 | 54 | **15** |

  **ep323 把灾难从 22 降到 15（−32%），同时把优秀从 31 降到 23**，两者在均值上抵消。
  灾难正是翻可视化时最先被注意到的东西，所以它**看着**比 baseline 好。官方权重是两头同时改善，
  所以聚合指标才干净地赢。「用顶端精度换更少灾难」是真实效果，但不等于「有用的先验」。
- **先验改动的大部分在图像叠加里看不见。** 相对无先验拟合，两人相对摆位的改动量：
  官方 深度 200.1mm vs 图像平面 89.6mm；interx6 227.7mm vs 124.9mm。四个 run 的
  `keypoint2d` 权重相同、投影被钉死，所以叠加视图本来就看着差不多——分歧在叠加图无法呈现的那个轴上。
- **穿模的代价照付**：baseline 22.6mm，官方 34.3，interx6 34.3，camfix 33.2。
  两个自训模型付出了与官方相同的互穿代价，却没换到收益。

**口径说明**

- 论文 Table 1 那行**仅供量级参照，不是严格对照**：论文评的是全测试集、真值是 Flickr fits；
  本表评的是 133 张有接触标注的子集、真值是接触标注本身。我们复现的官方权重
  （15.9/37.0/53.2/66.5）比论文（19/44/62/73）系统性低几个点，与 CHI3D 上复现值略差于论文的
  情况一致。
- 论文 Table 1 图注写的是「radius r mm」，但数值只有按 **cm** 理解才与实现对得上
  （代码里 `pcc_x = np.arange(0, 1.0, 0.05)`，距离单位是米）。上表按 cm 对齐。
- 四个 run 的配置逐项一致，唯一变量是先验；`baseline` 的 `use_diffusion: False`，
  扩散模型根本没加载。

---

## 3. 结论

**① 在实验室域（CHI3D），第三代自训模型 interx6 与官方权重相当——关节和接触两项都是。**

JOINT PA-MPJPE 70.6 vs 69.9，PCC 四档互有高低，逐图胜率 49.0%（z=−0.43）。
考虑到 interx6 的训练数据是**纯合成灰模渲染 + 911 条 CHI3D**，而官方用了 60% 野外真实照片，
这个结果说明六视角改造和混入 CHI3D 是有效的。

**② 在野外域（FlickrCI3D），两个自训模型在聚合指标上都等同于「不用先验」。**

PCC 四档全部落在 baseline 的 ±2pp 内，逐图胜率 49.6%。第三代相对第二代**没有可测量的改善**。
误差分布并不完全相同（ep323 灾难 22→15、优秀 31→23，均值抵消），所以它翻图时看着比 baseline 好；
但「用顶端精度换更少灾难」不等于有用的先验——官方权重是两头同时改善。

**③ 这个落差来自域，不是接触能力。**

同一个接触指标（`dist_on_gt` / PCC），interx6 在 CHI3D 上与官方打平、在 FlickrCI3D 上失效。
指标轴被控制住之后，剩下的唯一变量就是域。根因是训练数据里**一张野外照片都没有**：
InterX 是合成渲染，CHI3D 是实验室。而 cond_bev 学的是 `p(真实姿态 | BEV 估计)`——
一个纠错模型，它必须见过「BEV 在那类图像上会犯什么错」。实测 BEV 在真实 Flickr 照片上
硬失败率 **0/184 = 0%**，在 InterX 合成渲染上是 **18.3%**，两者的错误分布不是一回事。

**④ 因此本轮模型的定位是**：在见过的域内是一个真实可用的先验；
**不能**当官方权重的替代品用在野外照片上——那条路上它不仅无收益，还会主动造成失败
（实测它把姿态拽离 2D 关键点证据的幅度是官方的 2.6 倍，而优化流程的权重比
`keypoint2d 0.02` : `diffusion_prior_pose 100` 是围绕官方先验调的）。

**⑤ 下一步**：补野外训练数据（FlickrCI3D 伪真值，本地已有 10,631 张原图 + 标注）是唯一能同时
修掉「BEV 错误分布不匹配」「交互构型分布不对」「betas 退化」三条的路径。
零成本的先行判别实验是把 `diffusion_prior_pose` 从 100 下扫到 30/10 重跑这 133 张，
判断先验是「方向对但过度自信」还是「模态本身错」。

---

## 附：复现命令

```bash
conda activate buddi && export PYTHONPATH=$PWD ESSENTIALS_HOME=$PWD/essentials
mkdir -p llib/methods/hhcs_optimization/evaluation/temp
R=datasets/scripts/CHI3D/repro

# CHI3D Table 3 —— 官方权重（三行基线）
BASE_FOLDER=demo/optimization/chi3d_table3_reproduce $R/run_all.sh fit  buddi_gen
BASE_FOLDER=demo/optimization/chi3d_table3_reproduce $R/run_all.sh eval buddi_gen
python $R/s07_eval_bev_raw.py                     # BEV (raw SMPL) 行，无需拟合

# CHI3D Table 3 —— 自训权重的 BUDDI 行（换 CKPT / CKPT_CFG 即可）
CKPT=demo/diffusion/training/interx6_chi3d_cond_bev/checkpoints/2026_08_24-23_11_43__0000000323__0000000463__32.32.pt \
CKPT_CFG=$R/cfg/interx6_chi3d_ep323_bs1.yaml \
BASE_FOLDER=demo/optimization/chi3d_interx6 $R/run_all.sh fit buddi
BASE_FOLDER=demo/optimization/chi3d_interx6 $R/run_all.sh eval buddi

# CHI3D Table 3 —— 自训权重的 BUDDI (gen.) 行
#   注意 CKPT_CFG 用训练 config 本身（batch_size 512），不用 bs1 副本：
#   官方 gen. 行的 dbs 就是 512，采样协议必须对齐（见下方说明）
CKPT=demo/diffusion/training/interx6_chi3d_cond_bev/checkpoints/2026_08_24-23_11_43__0000000323__0000000463__32.32.pt \
CKPT_CFG=demo/diffusion/training/interx6_chi3d_cond_bev/config.yaml \
BASE_FOLDER=demo/optimization/chi3d_interx6_gen $R/run_all.sh fit buddi_gen
BASE_FOLDER=demo/optimization/chi3d_interx6_gen $R/run_all.sh eval buddi_gen

# CHI3D 的 PCC 列（chi3d_eval.py 里这段是注释掉的，另用脚本算）
#   见 scratchpad/chi3d_contact_eval.py

# FlickrCI3D 接触评测
python llib/methods/hhcs_optimization/evaluation/flickrci3ds_contact_eval.py ...
#   四方对比结果在 demo/optimization/flickr_contact_eval/comparison_v6/
```

**关于 `CKPT_CFG` 的两种用法**：

- **`BUDDI` 行用 `batch_size: 1` 的副本。** `main.py:160-163` 拿它给 SMPL-X 分配缓冲区，
  训练配置写的 512 会白占约 8GB 显存。仓库里已备好两份：
  `$R/cfg/interx6_chi3d_ep323_bs1.yaml` 和 `$R/cfg/interx_camfix_ep3509_bs1.yaml`，
  它们与各自的训练 config **逐行相同，仅 `batch_size` 一行不同**。该行的最终结果由 200 步优化
  收敛决定，`dbs` 只影响初始化那一次采样。
- **`BUDDI (gen.)` 行必须用 `batch_size: 512`（即训练 config 本身）。** 这一行的结果**就是**
  那次采样，而 `sample_from_model` 用 `cfg.batch_size` 生成 `dbs` 个样本、
  `fit_module.py:339-347` 只取第 0 个。查实官方 gen. 行用的 `essentials/buddi/buddi_cond_bev.yaml`
  的 `batch_size` 是 **512**，所以自训权重也必须用 512 才是同一采样协议。
  （严格说 512 取 [0] 与 1 取 [0] 是同分布 iid 抽样、统计等价，但既然显存够就按官方口径来。）

相关文档：
[RESULTS.md](datasets/scripts/CHI3D/repro/RESULTS.md)（论文 Table 3 复现） ·
[RESULTS_interx6.md](datasets/scripts/CHI3D/repro/RESULTS_interx6.md)（interx6 完整链路） ·
[demo/optimization/README.md](demo/optimization/README.md)（各评测目录说明）
