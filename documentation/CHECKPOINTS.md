# 训练好的模型权重

本 fork 在 Inter-X 上训练的模型权重，通过
[GitHub Releases](https://github.com/graces6pink/buddi/releases/tag/v0.1-interx-ckpt)
分发，不在 git 仓库里（避免每次 clone 都拖几百 MB）。

## 有哪些

每个 run 取验证损失（checkpoint 文件名末段的数字，即 `total_loss`，越小越好）
最低的那个 checkpoint。四个 run 都已收敛——最优值都落在训练末期。

| Run | 训练数据 / 条件 | epoch | val loss | 下载 |
|---|---|---|---|---|
| `interx_uncond` | Inter-X，无条件 | 1799 | **12.26** | `interx_uncond.tar.gz` |
| `interx_uncond_clean` | Inter-X（清洗过），无条件 | 1899 | **16.09** | `interx_uncond_clean.tar.gz` |
| `interx_cond_bev_camfix` | Inter-X，BEV 条件（相机修正后） | 3609 | **26.16** | `interx_cond_bev_camfix.tar.gz` |
| `interx6_chi3d_cond_bev` | Inter-X 六视角 + CHI3D，BEV 条件 | 686 | **31.87** | `interx6_chi3d_cond_bev.tar.gz` |

> 不同 run 的 loss 数值**不能横向比较**——训练数据和损失项配置都不同。同一列里
> 只有跟自己的历史比才有意义（各 run 的起始 loss 分别是 90.47 / 62.99 / 187.48 / 298.72）。

每个压缩包里是一个 `.pt` 加同 run 的 `config.yaml`。**两个都要**：加载权重时需要配套的
配置，光有 `.pt` 用不了。

模型本体 3.5M 参数；文件有 28–30 MB 是因为一并存了 optimizer 状态，所以这些
checkpoint 既能推理也能直接续训。

## 下载与校验

```bash
cd $BUDDI_ROOT
TAG=v0.1-interx-ckpt
BASE=https://github.com/graces6pink/buddi/releases/download/$TAG

mkdir -p demo/diffusion/training
for run in interx_uncond interx_uncond_clean interx_cond_bev_camfix interx6_chi3d_cond_bev; do
    curl -L -o "$run.tar.gz" "$BASE/$run.tar.gz"
    tar xzf "$run.tar.gz" -C demo/diffusion/training/
done

# 校验（可选，需在删除 tar.gz 之前做）
curl -L -o SHA256SUMS "$BASE/SHA256SUMS" && sha256sum -c SHA256SUMS
```

解压后目录结构与训练时一致：

```
demo/diffusion/training/
├── interx_uncond/
│   ├── config.yaml
│   ├── checkpoints/
│   │   └── 2026_08_02-09_48_26__0000001799__0000000167__12.26.pt
├── ...
```

保持这个结构很重要——续训依赖它（见下）。

## 怎么用

**无条件采样**（`--checkpoint-name` 必须是绝对路径，原因见
[USAGE_zh.md](./USAGE_zh.md#1-无条件采样生成--渲染成-gif)）：

```bash
python llib/methods/hhc_diffusion/evaluation/sample.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml \
  --output-folder demo/diffusion/samples/uncond_best \
  --checkpoint-name $(pwd)/demo/diffusion/training/interx_uncond/checkpoints/2026_08_02-09_48_26__0000001799__0000000167__12.26.pt \
  --num-samples 16 --batch_size 16 --max-t 1000 --skip-steps 20 --save-vis
```

**用 BEV 条件模型跑优化**（`_cfg` 指向同 run 的 `config.yaml`）：

```bash
python llib/methods/hhcs_optimization/main.py \
  --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml \
  --exp-opts logging.base_folder=demo/optimization/interx6_live \
             logging.run=fit_interx6 \
             datasets.train_names=['demo'] datasets.train_composition=[1.0] \
             datasets.demo.original_data_folder=demo/data \
             datasets.demo.image_folder=images_live \
             datasets.demo.bev_folder=bev_live \
             datasets.demo.vitpose_folder=vitpose_live \
             datasets.demo.openpose_folder=none \
             model.optimization.pretrained_diffusion_model_ckpt=demo/diffusion/training/interx6_chi3d_cond_bev/checkpoints/2026_08_25-11_44_23__0000000686__0000000463__31.87.pt \
             model.optimization.pretrained_diffusion_model_cfg=demo/diffusion/training/interx6_chi3d_cond_bev/config.yaml
```

**续训**：没有"指定 checkpoint"的参数——恢复是**自动**的。用原来的
`logging.base_folder` 和 `logging.run` 重跑训练即可，`Logger` 会扫描该 run 的
`checkpoints/` 目录并接上最后一个（[logger.py:132](../llib/logging/logger.py#L132)
的 `get_latest_checkpoint`，按文件名排序取最后一个；文件名以时间戳打头，所以
等价于取时间最新的那个），optimizer 状态一并恢复：

```bash
python llib/methods/hhc_diffusion/main.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml \
  --exp-opts logging.base_folder=demo/diffusion/training logging.run=interx_uncond \
             datasets.train_names=[interx] datasets.train_composition=[1.0]
```

所以解压出来的 `<run>/checkpoints/` 这层目录不能改名或压平，否则扫不到、会从头训。
另外注意它取的是**最新**而非**最优**——本 Release 每个 run 只有一个 checkpoint，
不存在歧义；但如果你把多个 checkpoint 放进同一目录，续训接的是时间最新的那个。

## 另一个 Release：复现结果用的权重

本页这四个是**按验证损失挑的最优 checkpoint**，适合直接使用或续训。但
[results/](../results/) 里那些实验跑的时候用的不是它们——那批权重单独发布在
[v0.2-experiment-ckpts](https://github.com/graces6pink/buddi/releases/tag/v0.2-experiment-ckpts)。
要复现 `results/` 里的数字，请用 v0.2；对应关系见
[results/README.md](../results/README.md)。

顺带一提，`results/` 的数据显示**验证损失并不能可靠地排序 checkpoint 的下游表现**：
损失最低的那个（camfix ep3509，26.58）PA-MPJPE 反而最差。所以本页的"最优"仅指
验证损失最低，不等于下游效果最好。

## 没有发布的部分

本机训练目录共 24 GB、657 个 checkpoint，这里只挑了 4 个最优的。中间 checkpoint、
tensorboard summaries、以及 `demo/optimization/` 下 19 GB 的优化实验输出都没有发布。
需要某个特定 epoch 的权重，请开 issue。

`interx_cond_bev` 这个 run 的 checkpoint 已不在本机（只剩 summaries 和 config），
因此无法发布。
