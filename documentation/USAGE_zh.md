# 常用操作速查（中文）

本 fork 在实际使用中整理的命令备忘。上游英文文档见 [README](../README.md)，
环境相关见 [ENVIRONMENT.md](./ENVIRONMENT.md)，数据准备见 [DATA.md](./DATA.md)。

下文所有命令都假设已经执行过这四行，`$BUDDI_ROOT` 即仓库根目录：

```bash
conda activate buddi
cd $BUDDI_ROOT
export PYTHONPATH=$(pwd)
export PYOPENGL_PLATFORM=egl    # pyrender 无头渲染必须设，不设会报错
```

---

## 1. 无条件采样（生成 + 渲染成 GIF）

```bash
python llib/methods/hhc_diffusion/evaluation/sample.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml \
  --output-folder demo/diffusion/samples/<输出文件夹名> \
  --checkpoint-name <某个 .pt 的绝对路径> \
  --num-samples 16 --batch_size 16 \
  --max-t 1000 --skip-steps 20 \
  --save-vis
```

> **`--checkpoint-name` 必须写具体某个 checkpoint 文件的绝对路径，不能用 `latest`。**
> 脚本里"自动找最新 checkpoint"的逻辑是根据配置文件路径去推断目录的，跟自定义的
> `demo/diffusion/training/<run名>/` 对不上，推断不到实际位置。每次手动指定文件最省事，例如：
>
> ```
> --checkpoint-name $BUDDI_ROOT/demo/diffusion/training/interx_uncond/checkpoints/2026_07_30-05_42_36__0000000129__0000000167__18.96.pt
> ```

用预训练权重的话：

```bash
python llib/methods/hhc_diffusion/evaluation/sample.py \
  --exp-cfg essentials/buddi/buddi_unconditional.yaml \
  --output-folder demo/diffusion/samples/ \
  --checkpoint-name essentials/buddi/buddi_unconditional.pt \
  --max-images-render=100 --num-samples 100 \
  --max-t 1000 --skip-steps 10 --log-steps=100 --save-vis
```

### 参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `--num-samples` | 生成多少个双人姿态样本 | 看多样性调大（如 64），纯展示 16 够用 |
| `--batch_size` | 一批处理多少个 | 显存够就跟 `--num-samples` 一样 |
| `--max-t` | 从多"重"的噪声开始去噪 | 1000 = 从纯噪声开始，标准做法，一般不用改 |
| `--skip-steps` | 去噪步长 | 越大越快、细节略糙。展示用 20，追质量调到 5~10 |
| `--render-width` / `--render-height` | 渲染尺寸 | 默认 200×256，想清晰可改 400×512 |
| `--render-floor` | 加地面网格 | 想要空间感可以加 |
| `--max-images-render` | **不用管** | `--save-vis` 是另一套渲染逻辑，这个参数给脚本里目前未启用的另一段代码用，保持默认 0 |

### 输出

结果在 `<输出文件夹>/generate_1000_20_v0/`：

- `renders/*.gif` — 360° 旋转的可视化 GIF
- `x_starts_smplx.pkl` — 生成的原始 SMPL-X 参数（数值，非图片）

---

## 2. 生成一批再挑最亲密/最疏远的

`sample_and_rank.py` 是本 fork 新增的脚本：先生成一个候选池，再按两人距离排序取前 k 个。

```bash
python llib/methods/hhc_diffusion/evaluation/sample_and_rank.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml \
  --checkpoint-name <某个 .pt 的绝对路径> \
  --output-folder demo/diffusion/samples/<输出文件夹名> \
  --num-samples 64 --top-k 8 --rank closest --export-glb
```

| 参数 | 作用 |
|---|---|
| `--checkpoint-name` | 必填，`.pt` 的绝对路径 |
| `--num-samples` | 候选池多大。越大越可能挑到更极端的结果，也越慢 |
| `--top-k` | 最终渲染几个 |
| `--rank` | `closest`（最亲密，默认）或 `farthest`（最疏远） |
| `--skip-steps` | 去噪步长，展示用默认 20 |
| `--export-glb` | 额外导出 `.glb`，可用 Blender / VS Code 的 3D 插件交互查看 |
| `--seed` | 固定住可复现，不设则每次随机 |

结果在 `<输出文件夹>/top8_closest/`，文件名形如 `rankXX_distX.XX.gif`（按距离从近到远编号），
以及对应的 `.glb`。

---

## 3. 在自己的图片上跑完整流程

放图片到 `demo/data/images_live/`，然后依次跑三步。

**第 1 步：ViTPose 2D 关键点**

```bash
python llib/utils/keypoints/vitpose_model.py \
  --image_folder demo/data/images_live \
  --out_folder demo/data/vitpose_live
```

**第 2 步：BEV 粗略 3D 初始化**

```bash
for image in demo/data/images_live/*; do
    image_name=$(basename "$image")
    bev -i "$image" -o demo/data/bev_live/$image_name
done
```

> 这两步都是遍历整个文件夹、**覆盖**已有的同名输出（`vitpose_model.py` 用
> `os.makedirs(out_folder, exist_ok=True)`，不会跳过已处理的图）。所以新增图片后
> 直接对整个文件夹重跑即可，不用单独摘出新图；重复生成的内容一致，覆盖没问题。

**第 3 步：BUDDI 优化**

用官方预训练权重：

```bash
python llib/methods/hhcs_optimization/main.py \
  --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml \
  --exp-opts logging.base_folder=demo/optimization/buddi_cond_bev_demo_live \
             logging.run=fit_buddi_cond_bev_pretrained \
             datasets.train_names=['demo'] datasets.train_composition=[1.0] \
             datasets.demo.original_data_folder=demo/data \
             datasets.demo.image_folder=images_live \
             datasets.demo.bev_folder=bev_live \
             datasets.demo.vitpose_folder=vitpose_live \
             datasets.demo.openpose_folder=none \
             model.optimization.pretrained_diffusion_model_ckpt=essentials/buddi/buddi_cond_bev.pt \
             model.optimization.pretrained_diffusion_model_cfg=essentials/buddi/buddi_cond_bev.yaml
```

换成自己在 Inter-X 上训的权重，只改最后两行（注意 cfg 指向该 run 的 `config.yaml`）：

```bash
             model.optimization.pretrained_diffusion_model_ckpt=demo/diffusion/training/interx_cond_bev/checkpoints/<某个>.pt \
             model.optimization.pretrained_diffusion_model_cfg=demo/diffusion/training/interx_cond_bev/config.yaml
```

---

## 4. 训练

BEV 条件模型，在 Inter-X 上：

```bash
python llib/methods/hhc_diffusion/main.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml \
  --exp-opts logging.base_folder=demo/diffusion/training logging.run=<新run名> \
             datasets.train_names=[interx] datasets.train_composition=[1.0] \
             datasets.val_names=[interx] \
             datasets.interx.processed_data_folder=datasets/processed/InterX \
             datasets.interx.allow_missing_bev=True \
             datasets.augmentation.use=True \
             model.regressor.losses.pseudogt_v2v.weight=[1000.0] \
             logging.logger=tensorboard
```

> `allow_missing_bev=True` 会保留 BEV 检测失败的样本（`bev_smplx_*` 全零）。
> 无条件训练时无所谓（根本不读 `bev_*`），但**做 BEV 条件训练时建议设成 `False`**，
> 否则全零占位会被当成真实引导信号喂进去。各配置项含义见
> [DATA.md 的 Inter-X 章节](./DATA.md#relevant-config-options)。

---

## 5. 检查 Inter-X 数据

```bash
# 指定某一帧
python llib/data/preprocess/utils/inspect_interx_frame.py --imgname G001T000A000R000_0_289

# 指定 take + 帧序号
python llib/data/preprocess/utils/inspect_interx_frame.py --take-id G001T000A000R004 --frame-index 2

# 随机抽（优先抽已被阶段 2 处理过的帧）
python llib/data/preprocess/utils/inspect_interx_frame.py --random --pick-seed 3

# 同一帧换个相机角度再看
python llib/data/preprocess/utils/inspect_interx_frame.py --imgname <帧名> --new-camera --seed 1

# 额外单独存每一格
python llib/data/preprocess/utils/inspect_interx_frame.py --random --save-panels
```

预处理本身、穿模检查、统计工具见 [DATA.md 的 Inter-X 章节](./DATA.md#inter-x)。
