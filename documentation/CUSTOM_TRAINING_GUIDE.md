# BUDDI 项目上手指南 —— 为「自己数据集重训练」做准备

> 本文档基于对代码库的实际阅读整理（而非只读 README），目的是让你在最短时间内建立起
> 关于 BUDDI 整个 pipeline 的正确心智模型，并清楚地知道：如果要换成自己的数据集重新
> 跑一遍「预处理 → 伪真值拟合 → 扩散模型训练 → 优化推理」，具体要改哪些文件、卡点在哪里。

---

## 1. 项目是什么

BUDDI（**Bud**dies **Di**ffusion model）是一个学习「两人近距离社交互动」联合分布的扩散模型，
直接生成两个人的 SMPL-X 参数。仓库里实际上包含 **三个相对独立但串联的子系统**：

| 子系统 | 目录 | 作用 |
|---|---|---|
| ① Flickr Fits（伪真值生成） | `llib/methods/hhcs_optimization/` | 用优化的方法，结合 BEV 初始化 + 2D 关键点 + （可选）接触标注，为图片中的两人拟合出 SMPL-X 参数，产出 **训练 BUDDI 所需的伪真值（pseudo-GT）** |
| ② BUDDI 扩散模型训练 | `llib/methods/hhc_diffusion/` | 在①产出的伪真值（以及 CHI3D/Hi4D 的 mocap 真值）上训练 diffusion transformer，学习两人姿态的联合分布 |
| ③ Optimization with BUDDI（推理） | `llib/methods/hhcs_optimization/`（同一份代码，配置不同） | 把训练好的 BUDDI 当作 **先验**（通过类似 DreamFusion 的 SDS loss），对新图片做优化拟合，不再需要接触真值标注 |

`demo.sh` 跑的就是 ③：ViTPose 出 2D 关键点 → BEV 出粗略 3D 初始化 → 用训练好的 BUDDI 做优化。

**你要做的「用自己数据集重新训练」，本质上是把 ① 和 ② 重新跑一遍**，③ 只是最终验证效果的手段。

---

## 2. 代码地图（真正需要认识的文件）

```
llib/
├── data/
│   ├── build.py                     # 组装 Dataset 的入口（区分「优化」和「扩散训练」两套）
│   ├── single_optimization.py       # ③ 优化推理用的 dataset（按需读原始图片）
│   ├── single.py                    # ② 扩散训练用的 dataset（读预处理好的 .pkl）
│   ├── collective.py                # 多数据集混合采样（training composition）
│   └── preprocess/
│       ├── demo.py                  # 处理"任意新图片"的通用类（image+bev+vitpose[+openpose]）
│       ├── flickrci3d_signatures_contacts.py  # FlickrCI3D 专用（依赖 interaction_contact_signature.json）
│       ├── chi3d.py / hi4d.py       # CHI3D / Hi4D 专用
│       └── utils/process_*.py       # 离线批处理脚本，生成 processed.pkl / correspondence.pkl
├── defaults/datasets/datasets.py    # ⭐ 所有数据集的 dataclass 配置定义（新增数据集从这里改）
├── methods/
│   ├── hhcs_optimization/
│   │   ├── main.py                  # ①③ 共用入口
│   │   └── configs/                 # flickr_fits.yaml(①) / heuristic_*.yaml(①,无需接触标注)
│   │                                 # buddi_cond_bev*.yaml(③) / buddi.yaml
│   └── hhc_diffusion/
│       ├── main.py                  # ② 训练入口
│       ├── evaluation/sample.py     # 无条件采样
│       └── configs/                 # config_buddi_v02.yaml(无条件) / config_buddi_v02_cond_bev.yaml(BEV条件)
essentials/                          # SMPL-X等body model、BUDDI预训练权重、先验、接触区域定义
datasets/original/, datasets/processed/  # 数据集的软链接约定目录
documentation/DATA.md                # 现有三个数据集(FlickrCI3D/CHI3D/Hi4D)的下载与目录结构
```

关键认知：**`single_optimization.py`（优化/推理用）和 `single.py`（扩散训练用）是两套独立的
数据加载逻辑**，数据集名字（`dataset_name`）在两边都是**硬编码的 if/elif 分支**，不是自动发现的：

- `single_optimization.py` 里认识：`flickrci3ds*`、`chi3d`、`demo`、`hi4d`
- `single.py` 里认识：`flickrci3ds*`、`chi3d`、`hi4d`（**不认识 `demo`！**）

这是接入自己数据集时最容易踩的坑，见第 4 节。

---

## 3. 完整数据流水线

```
原始图片
   │
   ├─► ViTPose (llib/utils/keypoints/vitpose_model.py)  → 2D 关键点(全身/带手部面部)
   ├─► BEV (third-party ROMP, 命令行 `bev`)              → 粗略 3D 初始化 (SMPL betas/pose/scale/cam)
   └─► [可选] OpenPose                                   → 2D 关键点(备用/更准)
                │
                ▼
   ① llib/methods/hhcs_optimization/main.py
      (flickr_fits.yaml 或 heuristic_0X.yaml，use_gt_contact_map 可开关)
      —— 用 keypoint 2D loss + pose prior + (可选)接触 loss，优化出每人的 SMPL-X 参数
                │
                ▼
        伪真值 (pseudo-GT) .pkl，存到 processed_data_folder/pseudogt_folder
                │
                ▼
   ② llib/methods/hhc_diffusion/main.py
      (config_buddi_v02.yaml 无条件 / config_buddi_v02_cond_bev.yaml BEV条件)
      —— 训练 diffusion transformer 学习两人联合分布
                │
                ▼
        essentials/buddi/*.pt (+ *.yaml) 训练好的 BUDDI 权重
                │
                ▼
   ③ llib/methods/hhcs_optimization/main.py
      (buddi_cond_bev*.yaml, use_diffusion=True, 加载②的 ckpt)
      —— 用 BUDDI 作为先验，对新图片做 SDS 优化，得到最终双人姿态
```

CHI3D / Hi4D 走的是另一条支线：它们本身有 mocap 真值，不需要①的优化拟合，只需各自的
`process_chi3d.py` / `process_hi4d.py` 预处理成 `processed.pkl`，直接作为②训练里的真值数据源
（与 Flickr 的伪真值混合，见 `config_buddi_v02_cond_bev.yaml` 里 `train_composition: [0.2, 0.2, 0.6]`）。

---

## 4. 用自己的数据集重新训练：先问自己一个问题

**你的数据有没有精确的 SMPL-X 真值（比如多相机 mocap、Inter-X 这类视频抽帧+SMPL-X 参数的数据集）？**

- **没有，只有原始图片**（网络图片、自己拍的照片）→ 走 **路径 A**：需要 ViTPose/OpenPose 出 2D 关键点，
  再靠优化拟合出伪真值（pseudo-GT），本质是"用 2D 信息反推 3D"。
- **有，图片对应的 SMPL-X 参数已经是准确值**（比如 Inter-X）→ 直接走 **路径 C**（见下），
  **完全不需要 ViTPose / OpenPose / Detectron2 这套 2D 检测流程**，因为你不需要"反推"3D，
  你已经有 3D 了。这也是 CHI3D / Hi4D 在本仓库里实际走的路。

这个判断很重要：本文档早期版本容易让人误以为"自己数据集训练=必须先跑通2D关键点检测"，
但那只在**没有真值、需要靠优化生成伪真值**的场景下成立（路径 A）。如果你手上已经有真值，
2D 检测这一整套（包括我们踩过的 Detectron2 把两人识别成一个人的坑）跟你没有关系，直接跳到路径 C。

### 路径 A（推荐，改动最小）—— 把自己的数据「伪装成」FlickrCI3D 格式

这是工程量最小的路线，因为 `flickrci3ds` 这个 dataset_name 在②的训练代码（`single.py`）里
已经是现成支持的，不用碰任何训练代码。

1. **收集数据**：图片按 `datasets/original/<YourDataset>/train/images/*.png` 存放。
2. **生成 2D 关键点**：跑 ViTPose（`llib/utils/keypoints/vitpose_model.py`，`demo.sh` 里有示例调用）。
   如果没有 OpenPose，`Demo`/`FlickrCI3D_Signatures` 预处理类都支持 ViTPose-only（会自动兜底）。
3. **生成 BEV 初始化**：对每张图跑 `bev -i <img> -o <out>/<img>_0.08.npz`（`demo.sh` 里的 for 循环）。
4. **（可选）接触标注**：如果你能标出两人的接触区域（`interaction_contact_signature.json`，
   见 `llib/data/preprocess/flickrci3d_signatures_contacts.py` 里的格式），就能用
   `flickr_fits.yaml`（`use_gt_contact_map` 可选）获得更准的伪真值；**没有标注也没关系**，
   用 `heuristic_01/02/03.yaml`（`use_gt_contact_map: False`）走纯优化路线一样能出伪真值。
5. **跑①生成伪真值**：仿照 README「Flickr Fits」一节的命令，把
   `datasets.train_names=['flickrci3ds']` 指向你自己在 `llib/defaults/datasets/datasets.py` 里
   新增/复用的 `FlickrCI3D_Signatures` dataclass（把 `original_data_folder` /
   `processed_data_folder` 指到你的目录即可，不需要改代码，只需要在命令行 `--exp-opts` 里覆盖，
   或者复制一份 dataclass 改个名字，比如 `MyDataset`，再在 `single_optimization.py` /
   `single.py` 里各加一个 `elif self.dataset_name == 'mydataset': ... FlickrCI3D_Signatures(...)`
   ——因为预处理类本身是通用的，只是目录参数不同）。
6. **跑②训练 BUDDI**：`datasets.train_names=['mydataset']`（或复用改名后的 `flickrci3ds`），
   `pseudogt_folder` 指向第5步的输出目录。

> 关键点：`flickrci3d_signatures_contacts.py` 目前**强制读取**
> `original_data_folder/<split>/interaction_contact_signature.json`（哪怕内容可以为空标注，
> 但文件必须存在且可解析）。如果完全没有接触标注，最简单的做法是造一个「空」的同格式 json
> （每张图对应一个空的 `person_ids`/`smplx_signature` 结构），或者直接照抄 `Demo` 类
> （`llib/data/preprocess/demo.py`，`has_gt_contact_annotation=False` 时不读这个文件）改造出一个
> 不依赖标注文件的新预处理类，见路径 B。

### 路径 B（更彻底，适合数据形态差异较大）—— 新增一个数据集类型

1. 在 `llib/defaults/datasets/datasets.py` 新增一个 `@dataclass MyDataset`，参照 `Demo` 或
   `FlickrCI3D_Signatures` 填字段（`original_data_folder`、`image_folder`、`bev_folder`、
   `vitpose_folder`、`features: DatasetFeatures(...)`）。
2. 在 `llib/defaults/datasets/main.py`（组合所有数据集到总 config）里注册这个新 dataclass。
3. 写 `llib/data/preprocess/mydataset.py`，可以直接复制 `demo.py` 或
   `flickrci3d_signatures_contacts.py` 改造（保留 `process_bev`/`load_single_image` 这类通用逻辑，
   改数据读取和输出 schema）。
4. 在 `llib/data/single_optimization.py` 和 `llib/data/single.py` 的 `load_data()` 里各加一个
   `elif self.dataset_name == 'mydataset':` 分支，指向新预处理类。
5. 其余（①跑优化生成伪真值、②跑训练）与路径 A 相同。

路径 B 工作量更大，但当你的数据不是「网络图片+接触标注」这种形态（比如你本来就有多相机
mocap，类似 CHI3D/Hi4D），可以参照 `chi3d.py`/`hi4d.py` 和
`datasets/scripts/CHI3D/`、`datasets/scripts/Hi4D/` 下的离线处理脚本来写。

### 路径 C（已有精确 SMPL-X 真值，如 Inter-X）—— 跳过①，直接产出训练用的真值数据

这是路径 B 的一个特化场景，单独展开是因为它**跳过了整个"生成伪真值"环节**，数据流变成：

```
Inter-X 等已有 SMPL-X 真值的数据
   │
   └─► 写一个 process_interx.py（参照 chi3d.py / hi4d.py）
              把每帧的 SMPL-X 参数 + （可选）相机参数，转换成 processed.pkl
                │
                ▼
   ② llib/methods/hhc_diffusion/main.py —— 直接用这份真值训练扩散模型
```

**完全不需要**：ViTPose、Detectron2、OpenPose——因为这一整套都是为了"从 2D 图像反推 3D 姿态"，
而你已经有 3D 真值了，不需要反推。

**唯一可能还需要的：BEV**——只有当你想训练 README 推荐的「BEV 条件版」扩散模型
（`config_buddi_v02_cond_bev.yaml`，训练时让模型学习"如何把 BEV 的粗糙估计纠正成正确姿态"）
才需要跑 `bev` 命令行工具产出初始化。如果训练无条件版本（`config_buddi_v02.yaml`），连 BEV 都不需要。
好在 BEV 这个环节我们实测过很稳定（连续几张密切接触图片都能正确识别出2个人），不是需要担心的对象。

具体步骤：
1. 参照 `chi3d.py`（`llib/data/preprocess/chi3d.py`）的 `__init__`/`load()` 结构写 `interx.py`，
   把 Inter-X 每帧的 `global_orient`/`body_pose`/`betas`/`transl`（或 Inter-X 自己的字段名，
   需要先确认）读出来，按两人一组拼装。
2. 在 `llib/defaults/datasets/datasets.py` 里新增一个 `@dataclass InterX`，参照 `CHI3D`/`HI4D` 的字段。
3. 在 `llib/data/single.py` 的 `load_data()`（第131-148行附近）加一个
   `elif self.dataset_name == 'interx':` 分支，指向 `InterX` 预处理类——**注意 `single_optimization.py`
   如果也要支持这个数据集名字，要单独加一遍，两边是各自独立的 if/elif，见第2节"关键认知"**。
4. **`DatasetFeatures` 里 `has_gt_smpl_pose`/`has_gt_smpl_shape` 这两个 flag 怎么设，需要先验证**——
   见下方"待确认事项"，不要想当然直接抄。

> #### ⚠️ 待确认事项：`has_gt_smpl_pose` / `has_pgt_smpl_pose` 的实际语义
> 直觉上，Inter-X 这种有精确 mocap 真值的数据，应该在 `DatasetFeatures` 里设
> `has_gt_smpl_pose=True, has_gt_smpl_shape=True`。但翻了一下现有代码，**CHI3D 和 HI4D 自己的
> `features` 配置(`datasets.py` 里 `CHI3D`/`HI4D` 这两个 dataclass)也没有设置这两个 flag**
> （只设了 `has_dhhc_sig=True, has_op_kpts=True`），尽管它们本身就是 mocap 级别的真值数据。
> 说明这套 flag 体系在训练 loss（`llib/training/diffusion_trainer.py` 或相关 loss 代码）里
> 具体怎么被消费、是否真的影响监督权重，还没有验证清楚。**接入 Inter-X 前建议先追一遍这几个
> flag 在训练代码里的实际作用，再决定怎么配置**，避免照抄一个实际不起作用/或语义不是预期的配置。

### 三条路径的分界线

- 只有单目图片、没有 3D 真值、可能有/没有接触标注 → **路径 A**（伪装成 FlickrCI3D）。
- 有多相机 / mocap / 视频，格式跟 CHI3D/Hi4D 类似（图片+需要额外处理才能拿到真值）→ **路径 B**。
- 已经是现成的、精确的 SMPL-X 参数（比如 Inter-X 抽帧数据）→ **路径 C**（路径B的特化版，跳过①）。

---

## 5. 环境与依赖（一次性）

```bash
./install_conda_env.sh     # conda 环境 + pytorch/pytorch3d/detectron2/mmcv/ViTPose 等
./fetch_data.sh             # 下载 essentials（BUDDI 预训练权重、priors、接触区域定义等）
./fetch_bodymodels.sh       # 下载 SMPL-X/SMPL/SMIL 身体模型（需要官网账号）
./install_thirdparty.sh     # 安装 BEV / ViTPose，转换 body model 到 BEV 格式
```

细节见 [`documentation/INSTALL.md`](./INSTALL.md)。注意 SMPL-X/SMPL/SMIL **不是 MIT 协议**，
需要在对应官网注册账号。

---

## 6. 现成数据集的目录约定（供参考/对齐自己数据集结构）

```
datasets/original/<Dataset>/          # 原始图片 + 官方标注（软链接到真实存储位置）
datasets/processed/<Dataset>/         # 你/官方生成的辅助数据：bev/、vitpose/、openpose/、
                                       # correspondence.pkl、processed.pkl、pseudogt/summaries/
```

详见 [`documentation/DATA.md`](./DATA.md)（FlickrCI3D / CHI3D / Hi4D 三个数据集各自的具体目录树）。

---

## 7. 关键配置文件速查

| 配置文件 | 阶段 | 关键字段 |
|---|---|---|
| `llib/methods/hhcs_optimization/configs/flickr_fits.yaml` | ①生成伪真值（用接触标注） | `use_gt_contact_map` |
| `llib/methods/hhcs_optimization/configs/heuristic_01/02/03.yaml` | ①生成伪真值（无接触标注） | `use_gt_contact_map: False` |
| `llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml` | ②训练（无条件） | `datasets.train_names`, `train_composition` |
| `llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml` | ②训练（BEV条件，README推荐） | 同上 + `model.regressor.experiment.guidance_params` |
| `llib/methods/hhcs_optimization/configs/buddi_cond_bev*.yaml` | ③用训练好的BUDDI做优化推理 | `model.optimization.pretrained_diffusion_model_ckpt/cfg` |

所有命令行都支持 `--exp-opts key=value ...` 直接覆盖 yaml 里的任意字段（README 里的例子全是这个用法），
不需要为了改路径就去改 yaml 文件本身。

---

## 8. 命令速查表

```bash
# ① 生成伪真值（Flickr Fits，训练集）
python llib/methods/hhcs_optimization/main.py \
  --exp-cfg llib/methods/hhcs_optimization/configs/flickr_fits.yaml \
  --exp-opts logging.base_folder=demo/optimization logging.run=flickr_fits \
             datasets.train_names=['flickrci3ds'] datasets.train_composition=[1.0] \
             datasets.val_names=[] datasets.test_names=[]

# ② 训练 BUDDI（BEV条件版本，README推荐）
python llib/methods/hhc_diffusion/main.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml \
  --exp-opts logging.base_folder=demo/diffusion/training logging.run=buddi_cond_bev \
             datasets.augmentation.use=True \
             model.regressor.losses.pseudogt_v2v.weight=[1000.0] \
             logging.logger='tensorboard'

# ③ 用训练好的 BUDDI 做优化推理（对齐 demo.sh 的写法，指向你自己的 ckpt）
python llib/methods/hhcs_optimization/main.py \
  --exp-cfg llib/methods/hhcs_optimization/configs/buddi_cond_bev_demo.yaml \
  --exp-opts logging.base_folder=demo/optimization/my_run \
             datasets.train_names=['demo'] datasets.train_composition=[1.0] \
             datasets.demo.original_data_folder=<your_data_folder> \
             model.optimization.pretrained_diffusion_model_ckpt=<your_ckpt.pt> \
             model.optimization.pretrained_diffusion_model_cfg=<your_ckpt.yaml>

# 评估：与伪真值/接触标注对比
python llib/methods/hhcs_optimization/evaluation/flickrci3ds_eval.py \
  --exp-cfg llib/methods/hhcs_optimization/evaluation/flickrci3ds_eval.yaml \
  -gt <base_folder>/fit_pseudogt_flickrci3ds_test -p <base_folder>/<run_folder> \
  --flickrci3ds-split test
```

---

## 9. 常见坑

1. **`demo` 数据集只能用于③（优化推理），不能直接用于②（扩散训练）**——`single.py` 的
   `load_data()` 没有 `demo` 分支，硬塞会直接 `NotImplementedError`。训练一定要走
   `flickrci3ds`/`chi3d`/`hi4d` 或你新增的分支。
2. **`interaction_contact_signature.json` 是强依赖**（路径 A），文件不存在会直接报错，不是「优雅降级」。
3. **SMPL→SMPL-X 的 betas/scale 转换**依赖 `ShapeConverter`（`llib/data/preprocess/utils/shape_converter.py`）
   和 `essentials/body_model_utils/smpl_to_smplx.pkl`，这一步是 BEV(SMPL) → SMPL-X pipeline 里
   容易被忽略但必需的环节。
4. **`bev_smpl_scale > 0.8` 会被当作儿童**，走 SMIL 转换分支（`shape_converter_smil`），如果你的数据
   里有儿童/矮小成人，注意这个阈值可能造成误判。
5. **单个优化 batch 最多一个 train / 一个 val / 一个 test 数据集**（`build_optimization_datasets`
   里有 `assert len(...) <= 1`），扩散训练阶段（`build_datasets`/`CollectivDataset`）才支持多数据集混合。
6. **两人配对逻辑基于 2D bbox IoU**（`llib/data/preprocess/demo.py` 的 `load_single_image`），
   如果你的图片里同时有 3 个以上互相接触的人，目前的 pipeline 只按两两重叠的 pair 处理，需要自己
   决定怎么拆分成 pair。
7. **路径 A 专属坑：Detectron2 在"两人紧密接触"的照片上经常把两人识别成一个人**——`demo.py`
   的配对逻辑依赖 ViTPose(没装OpenPose时)/OpenPose 检测出的人体框数量，如果检测器只找到1个人，
   这张图会被**静默跳过**（不报错、不警告），不会进入优化流程。实测：只要两人拥抱/贴得很近，
   Detectron2 几乎必现这个问题，NMS/置信度阈值调不动，因为区域提议阶段就没生成第二个人的候选框。
   BEV 的检测在同样场景下反而一直是准的——如果走路径 A 且数据集里有大量密切接触图片，
   优先考虑"用 BEV 的检测结果代替 Detectron2 给 ViTPose 提供框"，而不是死磕装 OpenPose。
   **注：如果走路径 C(已有真值)，这一条完全不适用，2D 检测这一层压根用不上。**
8. **`has_gt_smpl_pose`/`has_pgt_smpl_pose` 这两个 flag 在现有代码里的设置不完全符合直觉**——
   见路径 C 里的"待确认事项"，CHI3D/HI4D 都没设置这两个 flag，说明它们在训练 loss 里的实际作用
   需要单独确认，不要照抄。

---

## 10. 建议阅读顺序（吸收本文档之后）

1. 先跑通 `demo.sh`（③），确认环境和预训练权重没问题，同时借此观察 `demo/optimization/` 输出
   长什么样——这就是①最终要产出的同类数据格式的直觉参考（走路径 C 的话，这一步主要是为了
   验证环境/预训练权重没问题，不代表你自己训练时也要走 ViTPose/BEV 这条线）。
2. **走路径 A**：精读 `llib/data/preprocess/flickrci3d_signatures_contacts.py`，这是理解
   「一张图 + 标注 → 训练样本」全过程最关键的一个文件；然后用 5-10 张图跑通路径 A 的第 1-5 步，
   确认能生成伪真值再扩大规模。
   **走路径 C**：精读 `llib/data/preprocess/chi3d.py`（或 `hi4d.py`），这是理解「已有真值 →
   训练样本」全过程最关键的文件；然后追一遍 `has_gt_smpl_pose` 等 flag 在训练 loss 里的作用
   （见第9节坑8），确认清楚再写 `interx.py`。
3. 跑②训练前，先看一眼 `llib/training/diffusion_trainer.py` 和 `logging.logger`（wandb/tensorboard）
   配置，决定要不要接自己的实验跟踪账号。
