# 实验结果

`demo/optimization/` 下各次优化实验的**量化结果**。渲染图和逐样本拟合结果
（合计 19 GB）没有保留，这里只存汇总指标、配置和对比报告——不到 1 MB，
所以直接进 git，不走 Releases。

每个实验目录下：

- `results.pkl` — 完整指标，含逐动作（Hug / Kick / Handshake …）和逐 subject 拆分
- `config.yaml` — 该次运行的完整配置，含所用的 checkpoint 路径

## CHI3D 验证集

指标是 PA-MPJPE（mm，越小越好）。`h0h1` 是把两人当作一个整体做 Procrustes 对齐后
的误差——衡量**相对姿态关系**是否正确，是这个任务真正关心的量；`h0` / `h1` 是各自
单独对齐的误差，反映单人姿态质量。

| 实验 | n | PA-MPJPE (h0h1) | h0 | h1 | 所用 checkpoint |
|---|---|---|---|---|---|
| `chi3d_table3_reproduce/fit_buddi_chi3d_val` | 429 | **69.9** | 50.3 | 47.1 | `buddi_cond_bev.pt` |
| `chi3d_interx6/fit_buddi_chi3d_val` | 429 | **70.6** | 49.1 | 47.5 | `2026_08_24-23_11_43__0000000323__0000000463__32.32.pt` |
| `chi3d_interx6_gen_ep725/fit_buddi_gen_chi3d_val` | 429 | **80.5** | 49.8 | 51.5 | `2026_08_25-13_06_16__0000000725__0000000463__31.95.pt` |
| `chi3d_interx6_gen/fit_buddi_gen_chi3d_val` | 429 | **80.6** | 49.9 | 51.8 | `2026_08_24-23_11_43__0000000323__0000000463__32.32.pt` |
| `chi3d_table3_reproduce/fit_buddi_gen_chi3d_val` | 429 | **82.6** | 53.0 | 52.2 | `buddi_cond_bev.pt` |
| `chi3d_table3_reproduce/eval_bev_raw` | 440 | **97.4** | 51.1 | 53.1 | `—` |
| `chi3d_camfix/fit_buddi_chi3d_val` | 429 | **99.7** | 58.5 | 59.0 | `2026_08_16-15_59_09__0000003509__0000000167__26.58.pt` |

`eval_bev_raw` 是未经优化的 BEV 原始输出，作为下界参考。

### 几点观察

**在 Inter-X 六视角 + CHI3D 上训练的模型（`chi3d_interx6`，70.6）基本追平了官方
预训练权重的复现结果（`chi3d_table3_reproduce`，69.9）**，差距 0.7 mm。这是这批
实验里最有价值的一条。

**验证损失低不代表下游效果好，而且跨 run 的损失完全不可比。** `chi3d_camfix`
用的 checkpoint 验证损失 26.58，是所有实验里最低的，但 PA-MPJPE 99.7 反而最差——
甚至差于未优化的 BEV 原始输出（97.4，注意该行样本数是 440 而非 429，
两者并非严格同一子集，但差距不足以解释这个量级的反转）。相比之下 `chi3d_interx6` 用的 checkpoint
损失 32.32，明显更高，下游却好得多。所以**挑 checkpoint 不能只看文件名末尾那个数**。

同一 run 内部则趋势一致但差异很小：`interx6` 的 ep323（损失 32.32）与 ep725
（损失 31.95）下游分别是 80.6 和 80.5，几乎无差别。

## FlickrCI3D 接触评估

`flickr_contact_eval/comparison/` 和 `comparison_v6/` 各含：

- `report.md` — 可读报告，含覆盖率和指标表
- `metrics.csv` / `metrics.json` — 汇总指标
- `per_image.csv` — 逐图结果

评估的是接触区域预测（IoU、F-score、精确率、召回率）以及 `dist_on_gt_mm`、
`penetration_mm`。`comparison_v6` 是较新的一版，跑在 150 张图上（133 张四个 run
都有结果）。

## 复现这些数字需要注意

**这些结果所用的 checkpoint，和 [Release v0.1-interx-ckpt](https://github.com/graces6pink/buddi/releases/tag/v0.1-interx-ckpt)
里发布的不是同一批。** Release 发布的是各 run 验证损失最低的那个；而这些实验跑的
时候用的是当时手头的 checkpoint。对应关系见上表"所用 checkpoint"列。

两个 Release 的分工：

- **[v0.1-interx-ckpt](https://github.com/graces6pink/buddi/releases/tag/v0.1-interx-ckpt)**
  —— 各 run 验证损失最低的 checkpoint。想直接拿来用或续训，用这个。
- **[v0.2-experiment-ckpts](https://github.com/graces6pink/buddi/releases/tag/v0.2-experiment-ckpts)**
  —— 上表这些结果**实际用的** checkpoint。想复现上面的数字，用这个。

具体差异：

| run | v0.1 发布的 | 实验实际用的（在 v0.2） |
|---|---|---|
| `interx6_chi3d_cond_bev` | ep686 (31.87) | ep146 (34.02)、ep323 (32.32)、ep725 (31.95) |
| `interx_cond_bev_camfix` | ep3609 (26.16) | ep2249 (27.02)、ep3509 (26.58) |
| `interx_cond_bev` | 未发布 | ep1299 (33.78)、ep1499 (34.66) —— **已从磁盘丢失，无法发布** |

下载复现用的权重：

```bash
cd $BUDDI_ROOT
BASE=https://github.com/graces6pink/buddi/releases/download/v0.2-experiment-ckpts
for a in interx6_chi3d_cond_bev interx_cond_bev_camfix; do
    curl -L -O "$BASE/$a-experiment-ckpts.tar.gz"
    tar xzf "$a-experiment-ckpts.tar.gz" -C demo/diffusion/training/
done
curl -L -O "$BASE/SHA256SUMS" && sha256sum -c SHA256SUMS
```

上表中每个用到自训练权重的实验，其 checkpoint 现在都能从 v0.2 获取。
其余用的是官方预训练权重 `essentials/buddi/buddi_cond_bev.pt`，需按
[DATA.md](../documentation/DATA.md) 自行获取，因许可证限制无法转发。

## 两个需要留意的坑

**`flickr_contact_eval/baseline-nodiff` 这个名字有误导性。** 它的扩散先验损失权重
（`diffusion_prior_pose=[100,100]`、`diffusion_prior_transl=[10000,10000]` 等）与
`buddi-official` **逐项完全相同**，并没有关掉扩散先验。两者唯一的差别是 checkpoint
文件名不同（`buddi_cond_bev_checkpoint.pt` vs `buddi_cond_bev.pt`），而前者已不在
本机、无从核对。所以 report 里 `baseline-nodiff` 那一行**不应被当作"无扩散先验"的
对照组**来解读。

**`interx_cond_bev` 这个 run 的权重已彻底丢失**（本机只剩 summaries 和 config），
因此 `demo/optimization/buddi_cond_bev_mine_live` 的结果永久不可复现。
