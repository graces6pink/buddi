# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目简介

BUDDI（**Bud**dies **Di**ffusion model）是一个学习「两人近距离社交互动」联合分布的扩散模型，直接生成两个人的 SMPL-X 人体参数。论文：*Generative Proxemics: A Prior for 3D Social Interaction from Images* (CVPR 2024)。仓库里包含三个串联的子系统（更详细的架构笔记见 [documentation/CUSTOM_TRAINING_GUIDE.md](documentation/CUSTOM_TRAINING_GUIDE.md)，建议先读）：

1. **Flickr Fits（伪真值生成）** — `llib/methods/hhcs_optimization/`：用优化方法（BEV 初始化 + 2D 关键点 + 可选接触标注）为图片中两人拟合 SMPL-X 参数，产出训练 BUDDI 所需的伪真值。
2. **BUDDI 扩散模型训练** — `llib/methods/hhc_diffusion/`：在伪真值（以及 CHI3D/Hi4D 的 mocap 真值）上训练 diffusion transformer，学习两人姿态的联合分布。
3. **Optimization with BUDDI（推理）** — 同样是 `llib/methods/hhcs_optimization/`（同一份代码，不同配置）：把训练好的 BUDDI 当先验（类 DreamFusion 的 SDS loss），对新图片做优化拟合，不需要接触真值标注。`demo.sh` 跑的就是这一步。

## 环境搭建

```bash
./install_conda_env.sh     # 创建 conda 环境 hhcenv39，装 pytorch/pytorch3d/detectron2/mmcv/ViTPose 等
./fetch_data.sh             # 下载 essentials（BUDDI 预训练权重、priors、接触区域定义等）
./fetch_bodymodels.sh       # 下载 SMPL-X/SMPL/SMIL 身体模型（需要 SMPL-X 和 SMPL 官网账号）
./install_thirdparty.sh     # 安装 BEV(ROMP) / ViTPose 子模块，转换 body model 到 BEV 格式
```

环境变量：`export PYTHONPATH=<repo_root>`（脚本里都是这样设置的，见 [demo.sh](demo.sh)）。

Python 环境是 conda 环境 `buddi`（注意：`install_conda_env.sh` 脚本里默认创建的环境名是 `hhcenv39`，但本项目实际使用的环境是 `buddi`，python 3.9, pytorch 1.9.1, cuda 10.2, pytorch3d）。仓库没有 pytest/setup.py/pyproject.toml，也没有测试套件——验证改动的方式是实际跑一遍对应阶段的 pipeline（见下方常用命令）并检查输出。

## 常用命令

所有入口脚本都用同一套参数模式：`--exp-cfg <yaml路径> [<yaml路径> ...] --exp-opts key=value ...`。`--exp-opts` 可以覆盖 yaml 里任意嵌套字段，不需要为了改个别参数就去改 yaml 文件本身（配置合并逻辑见 [llib/defaults/main.py](llib/defaults/main.py) 的 `merge()`）。

```bash
# 跑demo：ViTPose + BEV 出初始化，然后用预训练 BUDDI 做优化推理
./demo.sh

# 无条件采样（从预训练 BUDDI 生成随机样本）
python llib/methods/hhc_diffusion/evaluation/sample.py --exp-cfg essentials/buddi/buddi_unconditional.yaml \
  --output-folder demo/diffusion/samples/ --checkpoint-name essentials/buddi/buddi_unconditional.pt \
  --max-images-render=100 --num-samples 100 --max-t 1000 --skip-steps 10 --log-steps=100 --save-vis

# ① 生成伪真值（Flickr Fits，需要接触标注）
python llib/methods/hhcs_optimization/main.py --exp-cfg llib/methods/hhcs_optimization/configs/flickr_fits.yaml \
  --exp-opts logging.base_folder=demo/optimization logging.run=flickr_fits \
             datasets.train_names=['flickrci3ds'] datasets.train_composition=[1.0] \
             datasets.val_names=[] datasets.test_names=[]
# 无接触标注则用 heuristic_01/02/03.yaml 代替 flickr_fits.yaml（use_gt_contact_map: False）

# ② 训练 BUDDI（BEV 条件版，README 推荐）
python llib/methods/hhc_diffusion/main.py --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml \
  --exp-opts logging.base_folder=demo/diffusion/training logging.run=buddi_cond_bev \
             datasets.augmentation.use=True model.regressor.losses.pseudogt_v2v.weight=[1000.0] \
             logging.logger='tensorboard'

# ③ 用训练好的 BUDDI 做优化推理（对新图片跑 SDS 优化）
python llib/methods/hhcs_optimization/main.py --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev.yaml \
  --exp-opts logging.base_folder=demo/optimization/buddi_cond_bev logging.run=fit_buddi_cond_bev_flickrci3ds \
             datasets.train_names=['flickrci3ds'] datasets.train_composition=[1.0] \
             datasets.val_names=[] datasets.test_names=[] \
             model.optimization.pretrained_diffusion_model_ckpt=essentials/buddi/buddi_cond_bev.pt \
             model.optimization.pretrained_diffusion_model_cfg=essentials/buddi/buddi_cond_bev.yaml

# 评估：与伪真值 / 接触标注对比
python llib/methods/hhcs_optimization/evaluation/flickrci3ds_eval.py \
  --exp-cfg llib/methods/hhcs_optimization/evaluation/flickrci3ds_eval.yaml \
  -gt <base_folder>/fit_pseudogt_flickrci3ds_test -p <base_folder>/<run_folder> --flickrci3ds-split test
python llib/methods/hhcs_optimization/evaluation/chi3d_eval.py ...
python llib/methods/hhcs_optimization/evaluation/hi4d_eval.py ...
```

优化脚本（`hhcs_optimization/main.py`）支持 `--cluster_pid` / `--cluster_bs` 来把数据集切成 batch 分布到集群跑，也用于只处理部分数据做调试。

## 架构要点

### 配置系统（OmegaConf structured config）

所有配置都是 dataclass 定义的（分散在 `llib/defaults/*/`），通过 `llib/defaults/main.py` 的 `Config` dataclass 组合成顶层结构（`body_model` / `datasets` / `camera` / `model` / `training` / `evaluation` / `visualization` / `logging`）。运行时 `merge(cmd_args, default_config)` 依次：取 dataclass 默认值 → 用 `--exp-cfg` 指定的 yaml 覆盖 → 用 `--exp-opts` 的命令行 key=value 覆盖。新增配置字段时优先去改 `llib/defaults/` 下对应的 dataclass，而不是直接在 yaml 里加没有 schema 支持的字段。

### 两套独立的数据加载逻辑

- [llib/data/single_optimization.py](llib/data/single_optimization.py) — 优化/推理用（①③阶段），按需读原始图片，认识 `flickrci3ds*` / `chi3d` / `demo` / `hi4d`。
- [llib/data/single.py](llib/data/single.py) — 扩散训练用（②阶段），读预处理好的 `.pkl`，认识 `flickrci3ds*` / `chi3d` / `hi4d`，**不认识 `demo`**。
- [llib/data/collective.py](llib/data/collective.py) — ②训练阶段多数据集混合采样（按 `train_composition` 权重）。
- [llib/data/build.py](llib/data/build.py) — 组装入口：`build_optimization_datasets()` 给①③用（每个 split 最多一个数据集，见其中的 `assert`），`build_datasets()` 给②用（支持多数据集混合）。

数据集名字（`dataset_name`）在这两套加载逻辑里都是**硬编码的 if/elif 分支**，不是自动发现的——接入新数据集需要在两边（如果两个阶段都要用）各加一个分支。新数据集的字段定义在 [llib/defaults/datasets/datasets.py](llib/defaults/datasets/datasets.py)，注册到总配置在 [llib/defaults/datasets/main.py](llib/defaults/datasets/main.py)。

### 数据预处理层

`llib/data/preprocess/` 下每个类负责把"一张图 + 标注"转成训练/优化用的样本：
- [demo.py](llib/data/preprocess/demo.py) — 处理任意新图片的通用类（image + BEV + ViTPose [+ OpenPose]），两人配对基于 2D bbox IoU。
- [flickrci3d_signatures_contacts.py](llib/data/preprocess/flickrci3d_signatures_contacts.py) — FlickrCI3D 专用，强依赖 `interaction_contact_signature.json`（文件必须存在，即使内容为空）。
- [chi3d.py](llib/data/preprocess/chi3d.py) / [hi4d.py](llib/data/preprocess/hi4d.py) — CHI3D / Hi4D 专用，这两个数据集自带 mocap 真值，不走优化拟合这条路，只需离线预处理成 `processed.pkl` 直接作为②训练的真值源。
- `preprocess/utils/` 下是离线批处理脚本（`process_flickrci3ds.py` / `process_chi3d.py` / `process_hi4d.py`）和 `ShapeConverter`（SMPL↔SMPL-X betas/scale 转换，儿童体型走 SMIL：`bev_smpl_scale > 0.8` 会被当作儿童）。

### 训练/优化模块

- [llib/methods/hhc_diffusion/train_module.py](llib/methods/hhc_diffusion/train_module.py) + [llib/training/diffusion_trainer.py](llib/training/diffusion_trainer.py) — ②扩散模型训练循环。
- [llib/methods/hhcs_optimization/fit_module.py](llib/methods/hhcs_optimization/fit_module.py)（`HHCSOpti`）— ①③共用的优化拟合模块，接受 body model / camera / criterion / 可选的预训练 diffusion 先验。
- [llib/models/regressors/buddi.py](llib/models/regressors/buddi.py) — BUDDI 的核心 diffusion transformer 架构。
- [llib/models/diffusion/](llib/models/diffusion/) — 基于 [openai/guided-diffusion](https://github.com/openai/guided-diffusion) 改造的高斯扩散过程实现（`gaussian_diffusion.py` / `respace.py` / `resample.py`）。
- 损失函数：`hhc_diffusion/loss_module.py`（训练用）和 `hhcs_optimization/loss_module.py`（优化用）分别定义各自阶段的 loss 组合。

### 完整数据流水线

```
原始图片 ──► ViTPose(2D关键点) / BEV(粗略3D初始化) / [可选]OpenPose
                │
                ▼
① hhcs_optimization/main.py (flickr_fits.yaml 或 heuristic_0X.yaml)
   keypoint loss + pose prior + [可选]接触 loss ──► 伪真值 (pseudo-GT) .pkl
                │
                ▼
② hhc_diffusion/main.py (config_buddi_v02[_cond_bev].yaml)
   训练 diffusion transformer ──► essentials/buddi/*.pt (+*.yaml)
                │
                ▼
③ hhcs_optimization/main.py (buddi_cond_bev*.yaml, use_diffusion=True)
   用 BUDDI 做 SDS 先验优化 ──► 最终双人姿态
```

CHI3D/Hi4D 走另一条支线：自带 mocap 真值，跳过①，预处理后直接混入②的训练数据（`train_composition` 控制混合比例）。

### 输出与日志

[llib/logging/logger.py](llib/logging/logger.py) 的 `Logger(cfg)` 负责在 `logging.base_folder/logging.run` 下创建输出目录、保存合并后的 config、管理 checkpoint。支持 tensorboard/wandb（`logging.logger` 字段）。

### 第三方依赖（git submodule，`third-party/`）

- **ROMP**（提供 `bev` 命令行工具）— 粗略 3D 初始化。
- **ViTPose** — 2D 关键点检测，`pip install -v -e third-party/ViTPose/` 安装。
- **imar_vision_datasets_tools** — CI3D 官方工具，通过符号链接挂到 `essentials/imar_vision_datasets_tools`。

### 评测流水线（三个 eval 脚本共用的前置条件）

`llib/methods/hhcs_optimization/evaluation/` 下的 `chi3d_eval.py` / `flickrci3ds_eval.py` / `hi4d_eval.py` 有一批**开箱即错**的前置条件，全部是 import 期或收尾期崩溃，且报错信息与病因无关：

1. **缺失的关节回归器**。以下两个文件被 import 期硬引用，但**官方 `fetch_data.sh` 下载的 essentials.zip 里根本没有**（已实测确认，包内 `body_model_utils/` 只有 `smplx_faces.pt` / `smplx_inner_mouth_bounds.pkl` / `lowres_smplx.pkl` / `smpl_to_smplx.pkl`）：
   - `essentials/body_model_utils/joint_regressors/SMPLX_to_J14.pkl` (14, 10475)
   - `essentials/body_model_utils/joint_regressors/J_regressor_h36m.npy` (17, 6890)

   引用它们的确切位置是 [chi3d_eval.py:33-39](llib/methods/hhcs_optimization/evaluation/chi3d_eval.py#L33-L39)、[flickrci3ds_eval.py:29-33](llib/methods/hhcs_optimization/evaluation/flickrci3ds_eval.py#L29-L33)、[evaluation/utils.py:16-22](llib/methods/hhcs_optimization/evaluation/utils.py#L16-L22)（后者用硬编码相对路径，因此必须从仓库根目录运行）。
   用 `datasets/scripts/CHI3D/repro/s00_fix_essentials.py` 补齐：`J_regressor_h36m.npy` 从仓库内 `third-party/ROMP/smpl_model_data/` 拷贝，`SMPLX_to_J14.pkl` 从 SMPLer-X 的 HuggingFace 镜像下载。
   **不要用该脚本的 `--method reconstruct` 分支产出的数值报结果** —— 实测与标准回归器差 ~31mm 关节位置 / ~23mm PA-MPJPE。

2. **`ESSENTIALS_HOME` 必须 export**。[chi3d_eval.py:28](llib/methods/hhcs_optimization/evaluation/chi3d_eval.py#L28) 是 `os.environ['ESSENTIALS_HOME']`，未设置直接 KeyError。`fetch_data.sh:5` 里那行 export 是注释掉的。

3. **`llib/methods/hhcs_optimization/evaluation/temp/` 必须先建**。[evaluation/utils.py:367](llib/methods/hhcs_optimization/evaluation/utils.py#L367) 的 `ResultLogger.topkl()` 无条件往这里写 `result.pkl`，目录不存在会在**打印完指标之后**崩溃。

4. **README 的 `--eval-split test` 是错的**（[README.md:143](README.md#L143)）。[chi3d_eval.py:60](llib/methods/hhcs_optimization/evaluation/chi3d_eval.py#L60) 的 argparse 只接受 `train|val`；论文 Table 3 的 s03 对应 `--eval-split val`。

5. **`buddi_cond_bev.yaml` 里的 ckpt 路径指向不存在的文件**（[:30-31](llib/methods/hhcs_optimization/configs/buddi_cond_bev.yaml#L30-L31) 写的是 `buddi_cond_bev_checkpoint.pt` / `_checkpoint_config.yaml`，实际文件名是 `buddi_cond_bev.pt` / `buddi_cond_bev.yaml`）。每次跑优化都要用 `--exp-opts` 覆盖这两项，`demo.sh:49` 就是这么做的。

指标含义：`est_pa_mpjpe_h0` / `est_pa_mpjpe_h1` 是论文的 PER PERSON 两列，`est_pa_mpjpe_h0h1` 是 JOINT 列（把两人 14 个关节拼成 28 个后整体做 Procrustes）。三者都定义在 [llib/defaults/training/eval.py:61-64](llib/defaults/training/eval.py#L61-L64)，实现是 [llib/utils/metrics/points.py](llib/utils/metrics/points.py) 的 `PointError` + [llib/utils/metrics/alignment.py](llib/utils/metrics/alignment.py) 的 `ProcrustesAlignment`（相似变换 sR+t，含尺度）。

## 已知坑（详见 CUSTOM_TRAINING_GUIDE.md 第 9 节）

- `demo` 数据集只能用于①③（优化/推理），不能用于②（训练）——`single.py` 没有 `demo` 分支。
- 单次优化 batch（①③）最多一个 train / 一个 val / 一个 test 数据集；②训练阶段才支持多数据集混合。
- 走"没有接触标注、只有原始图片"路线时，Detectron2 在两人紧密接触的照片上容易把两人识别成一个人，检测器只找到 1 个人时该图会被**静默跳过**；同场景下 BEV 的检测通常更稳。
- `has_gt_smpl_pose` / `has_pgt_smpl_pose` 这两个 `DatasetFeatures` flag 在现有 CHI3D/HI4D 配置里并未设置，其在训练 loss 中的实际作用尚未验证清楚，接入新数据集时不要照抄。

### 预处理阶段的静默失败（实测踩过）

- **BEV 的 npz 文件名逐数据集不同，且是唯一带 `os.path.exists` guard 的输入**。CHI3D 要双下划线 `{name}__2_0.08.npz`（[process_chi3d.py:204](llib/data/preprocess/utils/process_chi3d.py#L204)），FlickrCI3D 要单下划线 `{name}_0.08.npz`（[process_flickrci3ds.py:161](llib/data/preprocess/utils/process_flickrci3ds.py#L161)）。双下划线来自 `bev -m video` 目录模式（[bev/main.py:301](third-party/ROMP/simple_romp/bev/main.py#L301) 的 prefix 加上 [romp/utils.py:63](third-party/ROMP/simple_romp/romp/utils.py#L63) 再补一个 `_`），单下划线来自逐图模式。
  名字不对**不会报错**，只会静默填 `bev_params_template` 的全零参数，最终表现为所有 `bev_human_idx == -1`、评测集为空，然后 [chi3d_eval.py:277](llib/methods/hhcs_optimization/evaluation/chi3d_eval.py#L277) 抛一个与病因无关的 `NameError: thres`。
  video 模式默认不做时序平滑（`--temporal_optimize` 是 `store_true`），所以对无关的独立图片也安全，可以跑 video 模式后批量改名。

- **另外三个关键点源没有 guard**，缺文件直接 FileNotFoundError：[process_chi3d.py:233-235](llib/data/preprocess/utils/process_chi3d.py#L233-L235) 无条件读 openpose / vitpose / vitposeplus。而 [evaluation/utils.py:44-50](llib/methods/hhcs_optimization/evaluation/utils.py#L44-L50) 的 `chi3d_items_for_eval` 会把四个 `*_human_idx` 任一为 −1 的样本**整条丢弃**，所以四个通道都必须非空才有评测样本。

- **`use_hands: False` 时只有 25 个身体关键点进入 loss**。[single_optimization.py:55-65](llib/data/single_optimization.py#L55-L65) 的 `kpts_idxs = np.arange(0,25)`，手/脸/轮廓全被切掉；[single_optimization.py:225-236](llib/data/single_optimization.py#L225-L236) 选的是 `item['vitpose']`。因此 **vitposeplus 通道被要求必须存在，但其数据从不进入任何 loss**。

- **`correspondance.py` 的 CHI3D 分支跑完不落盘**（[:217-247](llib/data/preprocess/utils/correspondance.py#L217-L247) 只填 `all_output` 就结束；只有 Hi4D 分支有 `pickle.dump`）。FlickrCI3D 分支则会落盘（`.process_folder()` 默认 `save_output=True`）。

- **`train_val_split.npz` 仓库里没有任何脚本生成**，但 [chi3d.py:59-61](llib/data/preprocess/chi3d.py#L59-L61) 和 [chi3d_eval.py:48](llib/methods/hhcs_optimization/evaluation/chi3d_eval.py#L48) 都要读。它在 `datasets/processed/CHI3D/train/` 下（不是 [DATA.md:172-175](documentation/DATA.md) 画的 CHI3D 根目录，文档是错的），内含 `train` / `val` 两个 subject 名数组。

- **`pcl_pcl_pairwise_distance` 的 `use_cuda` 默认 True**（[distance.py:25](llib/utils/threed/distance.py#L25)），索引张量建在 CUDA 上；[chi3d_eval.py:215](llib/methods/hhcs_optimization/evaluation/chi3d_eval.py#L215) 传的是 `.cpu()` 张量，会 RuntimeError。已改为跟随输入张量设备。

## 其他文档

- [documentation/INSTALL.md](documentation/INSTALL.md) — 详细安装步骤。
- [documentation/DATA.md](documentation/DATA.md) — FlickrCI3D / CHI3D / Hi4D 三个数据集的下载与目录结构约定（`datasets/original/<Dataset>/`、`datasets/processed/<Dataset>/`）。
- [documentation/CUSTOM_TRAINING_GUIDE.md](documentation/CUSTOM_TRAINING_GUIDE.md) — 接入自定义数据集重新训练的完整指南（三条路径：伪装成 FlickrCI3D / 新增数据集类型 / 已有 SMPL-X 真值如 Inter-X），比本文件更详细。

## 论文复现脚本（本仓库新增，未提交上游）

作者发布的 auxiliary 数据（`datasets/processed/` 里的 BEV/关键点/`processed.pkl`）在 [DATA.md](documentation/DATA.md) 里 CHI3D 和 Hi4D 两项标着 "coming soon"，实际是死链。以下脚本从 `datasets/original/` 重建整条链：

- `datasets/scripts/CHI3D/repro/` — CHI3D Table 3 复现，已跑通。入口 `run_all.sh {prep|fit|eval}`，
  阶段脚本 `s00`–`s07` 各自可独立重跑（都做了 exists 跳过）。`s07_eval_bev_raw.py` 补完了作者未写完的
  raw-SMPL BEV 评测分支（`chi3d_bev_verts_from_processed_data` 在 [evaluation/utils.py:109](llib/methods/hhcs_optimization/evaluation/utils.py#L109) 定义但全仓库无人调用），
  无需跑拟合。结果与偏差见 [datasets/scripts/CHI3D/repro/RESULTS.md](datasets/scripts/CHI3D/repro/RESULTS.md)。
- `datasets/scripts/FlickrCI3D/repro/` — FlickrCI3D 的对应工作，进行中。

复现 CHI3D Table 3 的实测结果（论文 → 复现）：BEV 50/52/96 → 51.1/53.1/97.4；BUDDI (gen.) 53/53/80 → 53.0/52.2/82.6；BUDDI 48/47/68 → 50.3/47.1/69.9。

已知的协议偏差（报数时必须说明）：OpenPose 通道是 ViTPose 检测的副本（本机未装 OpenPose）；BEV 是本地重跑的；`SMPLX_to_J14.pkl` 来自第三方镜像而非作者原件。
