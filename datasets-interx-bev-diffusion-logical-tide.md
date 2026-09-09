# 计划：单帧全链路检查脚本 `inspect_interx_frame.py`

## Context

阶段② 的 5 小时重跑正在进行（写这份计划时 1200/4735 take）。跑完之后需要有办法**对任意单独一帧做端到端体检**：从 take 里抽出来的那一帧真值，到随机相机渲染出的合成照片，到实际喂给 BEV 的数组，到 BEV 吐出来的原始结果，到 SMPL→SMPL-X 转换，再到最终写回样本的 `bev_smplx_*` / `pgt_smplx_*_cam`——每一步的**图和数值都要能看到**。

现有工具只覆盖了片段：`vis_interx_bev.py` 只比最终的 pgt vs bev，`--calibrate` 只出聚合统计，中间的渲染图、BEV 输入数组、BEV 原始输出全都是黑盒。坐标系那个 bug 之所以拖到现在才发现，正是因为中间态没人看得见。

**目标**：一条命令，指定一帧，产出一张多面板拼图 + 一份可读的参数转储 + 一份 npz，让每一步都能自己复核。

## 关键设计原则

脚本**必须调用生产代码本身**——`InterXBevProcessor.render_sample` / `match_detections` / `bev_to_smplx` / `pgt_to_camera_frame`，不另写平行实现。否则验证的是这个脚本，不是流水线。

## 改动 1：`process_interx_bev.py::render_sample` 加可选参数

签名改成 `render_sample(self, sample, cam_override=None)`。传入 `{'pitch','yaw','roll','dist'}` 时跳过 `sample_camera_params(self.rng)` 那一行，直接用给定值；其余逻辑一字不动，默认 `None` 时行为完全不变。

用途：每帧的相机是从 `RandomState(0)` 按序抽的，事后单帧抽不出来；但新代码已经把 `render_cam_euler` / `render_cam_dist` 存进样本了，有了这个参数就能**精确重放跑批当时那一帧的那张图**，并把重算的 `pgt_*_cam` 跟已落库的那份逐位对比。

（正在跑的任务不受影响：Python 在 import 时已把源码读进内存；即使中途崩溃续跑，新参数向后兼容。）

## 改动 2：新文件 `llib/data/preprocess/utils/inspect_interx_frame.py`

### CLI

```
--processed-data-folder  默认 datasets/processed/InterX
选帧（三选一）：--imgname <str> | --take-id <str> [--frame-index N] | --random [--pick-seed N]
相机模式：     --replay（默认，样本里有 render_cam_* 时）| --new-camera [--seed N]
--out-dir      默认 datasets/processed/InterX/diagnostics/frame_inspect
--save-panels  额外单独存每一格
```

样本里没有 `render_cam_*`（该 take 还没被重跑到）时，自动退回 `--new-camera` 并明确打印提示。

### 拼图面板（2×4）

| # | 内容 | 验证什么 |
|---|---|---|
| 1 | 世界系真值，正面视角 | 阶段①抽出来的这一帧本身合不合理 |
| 2 | 世界系真值，侧面视角 | 两人前后关系、有没有明显穿模 |
| 3 | 随机相机渲染出的 RGB 512×512 | 这就是"合成照片" |
| 4 | 实际传给 BEV 的 BGR 数组（**故意按原样显示**） | 通道看起来是错的才对——确认我们传的确实是 BGR |
| 5 | BEV 内部 `img_preprocess` 之后的图 | 应与面板 3 **逐像素相同**（脚本里断言） |
| 6 | `pj2d_org` 关键点叠在输入图上，两人不同色，标注置信度与我方 `our_screen_x` | BEV 有没有找对人、配对对不对 |
| 7 | BEV 自己的 `rendered_image` | 白送的——`render_mesh` 默认 True，跑批时本来就在算 |
| 8 | 对齐检查：`pgt_*_cam`（蓝/红）与 `bev_*`（绿/黄）共用一台相机叠加 | 坐标系到底对没对齐 |

面板 3/5 之间那个像素级断言是重点：BGR↔RGB 翻转如果哪天写反了，是典型的静默 bug。

### 文本转储（`<imgname>_params.txt`，按链路分 6 段）

- **STEP 0 抽样**：`imgname` 解码成 take/片段号/帧号；样本全部 key 的 shape/dtype；`information_missing`；有则打印 `render_cam_*`
- **STEP 1 世界系真值**：逐人 `global_orient` / `transl` / `betas` / `scale` 完整数值，`body_pose` 前几个关节 + 范数统计；两根关节距离（对照阶段①的 `dist_thresh=0.6`）
- **STEP 2 相机**：`pitch/yaw/roll/dist`、`center`、`radius`、`base_dist`、R 矩阵、T，以及推导出的 `A = F·Rᵀ` 和 `b = F·(T − center@R)`
- **STEP 3 BEV 输入**：渲染输出的 dtype/值域 → uint8 → BGR 的每一步形状与值域；BEV 内部的 `image_pad_info`（证明 padding 和 resize 对 512×512 都是 no-op）；像素级相等断言的结果
- **STEP 4 BEV 输出**：所有 key 的 shape；检测数、`center_confs`；逐检测的 `smpl_thetas[:3]`（global orient）、`smpl_betas`(10+scale)、`cam`、`cam_trans`、`pj2d_org` 的 x 范围；`match_detections` 的代价矩阵与最终配对
- **STEP 5 SMPL→SMPL-X**：逐人走 SMIL 还是 SMPLA 分支（`smpl_scale > 0.8`）、输入 betas(11) → 输出 betas(10)+scale、`root_trans`、最终 `transl`
- **STEP 6 落库与校验**：将写回的 `bev_smplx_*` 与 `pgt_smplx_*_cam` 数值；对齐指标（逐人朝向残差角、两人偏移向量夹角，并与世界系版本对照）；**重放模式下额外输出"重算 `_cam` vs 已落库 `_cam`"的最大绝对误差**（应在 1e-6 量级）

### npz 转储

所有中间量（含 R/T/center/A/b、BEV 原始输出、转换前后参数）存一份 `<imgname>_params.npz`，方便自己另算。

## 复用清单（不新写）

- `InterXBevProcessor` 的四个方法 + `sample_camera_params` / `BEV_FRAME_FIX` / `BEV_FOV` / `IMAGE_SIZE` — [process_interx_bev.py](llib/data/preprocess/utils/process_interx_bev.py)
- `build_body_model` / `make_renderer` / `render_pose` — [vis_interx_bev.py](llib/data/preprocess/utils/vis_interx_bev.py)
- `img_preprocess` — [llib/models/regressors/bev/utils.py](llib/models/regressors/bev/utils.py)（复现 BEV 内部预处理，与 BEV 走同一个函数）
- `Pytorch3dRenderer` / `PerspectiveCamera`

## 验证方法

1. 挑一帧已重跑完的 take，`--replay` 跑一遍：STEP 6 的"重算 vs 已落库"误差应 < 1e-5；面板 3 与面板 5 断言通过。
2. 同一帧再跑 `--new-camera --seed 1`：图应该换个角度，但 STEP 6 的对齐指标仍在同量级（~11°）。
3. 挑一帧 `information_missing=True` 的：脚本应优雅处理，明确指出是在"BEV 检测数 != 2"这一步失败，前 5 个面板照常出图。
4. 肉眼看面板 6（BEV 找对人没）和面板 8（两组网格是否重合）。

## 注意事项

- 脚本会自己加载一份 BEV 模型（约 2–3GB 显存），与正在跑的 5 小时任务共用 GPU。24GB 卡上另一位用户占 9.3GB，够用。
- `processed.pkl` 正在被原地重写。`atomic_save` 保证读到的永远是完整文件，但内容可能处于"部分 take 已更新"的中间态——没有 `render_cam_*` 的帧会自动走 `--new-camera`。

---

# 附：全链路追踪报告（前一轮产出，供对照）

# Inter-X → BEV → BUDDI 扩散训练：全链路数据流水线追踪报告

> 本文是**讲解 + 追踪手册**，不是待实施的改动计划。目标：让你能拿着它逐个文件、逐个函数对着代码读一遍，并且每一步都能自己跑命令验证中间结果是否合理。
> 文中所有数字分两类，已明确标注：**【实测】**= 我刚在这台机器上跑真实数据得到的；**【读码】**= 从代码/配置直接读出来的事实。

---

## Context

你的 Inter-X 格式双人 mocap 数据要用来重训 BUDDI 扩散模型。目前仓库里已经有一套完整实现（`EXECUTION_PLAN.md` 记录了历史），并且已经跑出了三个训练 run。你现在要做的是**亲自把整条流水线追一遍、检查中间结果是否合理**。本报告把这条链路拆成 6 个阶段，给出每个阶段的代码入口、数据形状、实际产物、以及我实测发现的**若干个真实问题**（其中一个是会影响 BEV 条件版训练正确性的坐标系问题，见第八节 A）。

---

## 零、全景图

```
datasets/original/InterX/motions/<take>/{P1,P2}.npz        ← 原始 mocap（8026 个 take）
   │
   │  ① llib/data/preprocess/utils/process_interx.py
   │     互动分段(根平移距离) → 抽帧 → 字段映射 → pgt_smplx_*，bev_smplx_* 填零
   ▼
datasets/processed/InterX/processed.pkl                     ← {take_id: [sample,...]}，95,448 帧
datasets/processed/InterX/train_val_split.npz               ← 按 take 划分 4262/237/236
   │
   │  ② llib/data/preprocess/utils/process_interx_bev.py
   │     随机相机渲染 512×512 → BEV 推理 → 检测配对 → SMPL→SMPL-X → 就地写回 bev_smplx_*
   ▼
processed.pkl（同一个文件，被原地更新）                      ← 71,109 帧拿到真实 BEV，24,339 帧失败
   │
   │  ②' 数据体检（不改 processed.pkl）
   │     check_interx_penetration.py → diagnostics/penetration_report.pkl + exclude_list.pkl
   │     depenetrate_interx.py（另存 processed_depenetrated.pkl）
   │     vis_interx_bev.py（pgt vs bev 并排渲染抽查）
   ▼
   │  ③ llib/data/preprocess/interx.py::InterX.load()
   │     按 split 取 take → 展平 → 缓存 {split}_diffusion.pkl → 过滤(缺BEV/穿模) → overfit截断
   ▼
datasets/processed/InterX/{train,val}_diffusion.pkl          ← 85,714 / 4,863 帧（未过滤前）
   │
   │  ④ llib/data/single.py::SingleDataset.get_single_item()
   │     人物排序(世界系 x) + swap 增广 → 张量化 → 'pgt_*' / 'bev_*' 键
   │     llib/data/collective.py + PartitionSampler → 多数据集混采（这里只有一个）
   ▼
batch: {pgt_global_orient[B,2,3], pgt_body_pose[B,2,63], pgt_betas[B,2,10],
        pgt_scale[B,2,1], pgt_transl[B,2,3], bev_*(同构), action, ...}
   │
   │  ⑤ llib/methods/hhc_diffusion/train_module.py
   │     preprocess_batch → cast_smpl（轴角→6D，相对平移）→ split_humans → 8 个 token
   ▼
   │  ⑥ 加噪 q_sample(t) → BUDDI Transformer 预测 x_start → SMPL 前向 → loss
   ▼
demo/diffusion/training/<run>/checkpoints/*.pt
```

---

## 一、阶段①：`process_interx.py`（原始 mocap → 训练样本）

**入口**：[llib/data/preprocess/utils/process_interx.py](llib/data/preprocess/utils/process_interx.py)

### 1.1 找 take
`find_takes()` (L29) 递归 `os.walk` 整个 `motions/`，**只要一个目录里同时有 `P1.npz` 和 `P2.npz` 就算一个 take**，排序后返回。【实测】8026 个 take。

### 1.2 读单人 `load_person()` (L38)
Inter-X npz → 4 个字段，其余丢弃：

| npz 字段 | 形状 | 处理 | 输出键 |
|---|---|---|---|
| `root_orient` | `[T,3]` | 原样 float32 | `global_orient` |
| `pose_body` | `[T,21,3]` | `reshape(T,63)` | `body_pose` |
| `betas` | `[1,10]` | `np.tile` 广播到 `[T,10]` | `betas` |
| `trans` | `[T,3]` | 原样 | `transl` |
| `pose_lhand`/`pose_rhand` | `[T,15,3]` | **丢弃**（BUDDI 不建模手） | — |
| `gender` | 标量字符串 | **丢弃** | — |

⚠️ **gender 丢弃是一个已知近似误差**：Inter-X 的 betas 是用 gender-specific SMPL-X 拟合的，这里全部塞进 `gender: neutral` 模型（见 `config_buddi_v02*.yaml` 的 `body_model.smplx.init.gender`），体型不严格等价。目前没解决。

### 1.3 互动分段 `detect_interaction_segments()` (L49)
1. `dist[t] = ||transl1[t] - transl2[t]||`（**根关节≈骨盆**间距，不是身体表面距离）
2. `medfilt(dist, 7)` 去抖动（序列长度 <7 时跳过）
3. `is_close = dist < dist_thresh`，连续 True 区间 = 候选片段
4. 丢弃 `< min_len(15)` 帧的片段
5. 前后各 pad 8 帧

**正式跑批用的是 `dist_thresh=0.6`**（不是代码默认的 1.2！默认值是 argparse 里的 `--dist-thresh 1.2`，实际跑批时命令行传了 0.6，目的是筛"大面积接触"）。这一点很关键：你如果直接跑默认参数会得到完全不同的数据集。

**已知局限**：根关节距离对"骨盆分开、只有手接触"的动作偏保守，会漏掉边界帧。

### 1.4 抽帧与样本组装 `process_take()` (L84)
片段内 `range(s, e, stride)`，正式跑批 `stride=20`。

每帧组装成一个 dict（P1 在 index 0，P2 在 index 1）：

```
imgname                  'G001T000A000R000_0_289'      # take_id + 片段号 + 帧号，全局唯一
imgpath                  'InterX/G001T000A000R000_0_289.png'   # 假路径，文件不存在
img_height/img_width     900 / 900                     # 占位，只进 gen_target 不参与计算
pgt_smplx_global_orient  (2,3)  float32
pgt_smplx_body_pose      (2,63) float32
pgt_smplx_betas          (2,10) float32
pgt_smplx_transl         (2,3)  float32
pgt_smplx_scale          (2,1)  float32  ← 全 0（成人）
bev_smplx_global_orient  (2,3)  ← 阶段①全 0 占位
bev_smplx_body_pose      (2,63) ← 全 0
bev_smplx_betas          (2,10) ← 全 0
bev_smplx_transl         (2,3)  ← 全 0
bev_smplx_scale          (2,)   ← 全 0，注意是 (2,) 不是 (2,1)
information_missing      False
take_id                  'G001T000A000R000'
```

【读码】**`imgpath` 里必须含子串 `'InterX'`**——`single.py::get_single_item()` 是靠 `imgpath` 子串匹配来分发数据集分支的（L452-461 和 L473-539），不是靠 `dataset_name`。这是全仓库最容易踩的隐式契约。

【读码】**`pgt_smplx_scale` 是 `(2,1)` 而 `bev_smplx_scale` 是 `(2,)`**，形状故意不一致：`single.py` 里 pgt 直接用 `[idxs]`，bev 用 `[idxs][:,None]`（L538），两边最终都是 `(2,1)`。这是照抄 FlickrCI3D 分支的历史包袱，新写预处理时极易踩坑。

### 1.5 划分 `make_split()` (L128)
**按 take 划分**（不是按帧），`seed=0`，90%/5%/5%。按 take 划分是对的——同一动作相邻帧高度相似，按帧划分会造成 train/val 泄漏。

### 1.6 【实测】阶段①产物核对

```
takes in processed.pkl: 4735 / 8026    （3291 个 take 一帧都没检出互动）
total frames: 95448
split: train=4262 val=237 test=236 （take 数）
pgt_smplx_scale 唯一值: [0.]
```

**合理性判断**：4735/8026 ≈ 59% 的 take 有可用数据，在 `dist_thresh=0.6` 这个很严的阈值下是合理的。平均每个 take 20 帧样本。

### 1.7 配套体检工具
[diagnose_interx.py](llib/data/preprocess/utils/diagnose_interx.py)：**跑正式预处理前**用它扫一批 take，输出根平移距离分布 + 一整格 `(dist_thresh, stride)` 组合分别能留下多少帧/片段/take，用来选参数。产物 `diagnostics/interx_distance_diagnostics.png`、`G001T000A000R000_distance_curve.png` 已经在仓库里。

---

## 二、阶段②：`process_interx_bev.py`（合成图 → 真实 BEV 估计）

**入口**：[llib/data/preprocess/utils/process_interx_bev.py](llib/data/preprocess/utils/process_interx_bev.py)

这一步做的事情是：**没有真实照片，就用 mocap 真值渲染一张合成照片，再拿 BEV 去"检测"它**，从而给每一帧配上一个 BEV 估计，作为 BEV 条件版训练的条件输入。

### 2.1 初始化 `InterXBevProcessor.__init__` (L59)
- `smplx.create(..., gender='neutral', num_betas=10, batch_size=2)` —— 渲染用的干净 SMPL-X（**注意不是训练用的那个带 `scale` 的 smplxa**）
- `PerspectiveCamera(afov_horizontal=60°, image_size=512×512, R=0, T=0)`
- `Pytorch3dRenderer(cameras=camera.cameras, 512, 512)` —— **共享同一个 pytorch3d camera 对象**（[renderer.py:46](llib/visualization/renderer.py#L46) `self.cameras = cameras`），所以 `update_camera_pose()` 改的 R/T 对后面 `transform_points_screen` 也生效。这一点是屏幕 x 匹配能成立的前提，我核对过了。
- `BEV(bev_settings(['-i','dummy','--calc_smpl']))` —— **模型常驻进程内**。对比 `demo.sh` 走 `bev` CLI 每张图重载一次模型（95,448 帧要 ~80 小时），常驻后单帧 ~74ms。
- 两个 `ShapeConverter`：`smpla→smplxa`（成人）、`smil→smplxa`（儿童），跟 [hi4d.py::process_bev](llib/data/preprocess/hi4d.py) 同一条路径。

### 2.2 随机相机 `sample_camera_params()` (L39)

| 参数 | 分布 | 意图 |
|---|---|---|
| yaw | `U(-150°, 150°)` | 排除正后方 ±30° |
| pitch | `N(5°,12°)` 截断 `[-20°,35°]` | 模拟持机人平视/略俯拍 |
| roll | `N(0°,3°)` 截断 `[-8°,8°]` | 手持抖动 |
| dist | `base_dist × U(0.85,1.25)` | 取景松紧 |

`base_dist = radius / sin(FOV/2) × 1.3`，radius 是两人合并网格相对质心的最大半径 → 保证两人总在画面内。

### 2.3 渲染 `render_sample()` (L89)
1. SMPL-X 前向得 `verts [2,V,3]`
2. **减去两人合并质心** `verts_c = verts - center` ← 注意：绝对世界位置在这一步被丢掉了
3. `update_camera_pose(pitch,yaw,roll,0,0,dist)` → `render()` → RGB float
4. `×255 → uint8 → [...,::-1]` 转 BGR（BEV/cv2 要 BGR）
5. `transform_points_screen(verts_c.mean(1))` 拿到两人质心的屏幕 x → `our_screen_x [2]`

### 2.4 检测配对 `match_detections()` (L123)
BEV 输出的两个检测顺序不保证对应 P1/P2。用 `pj2d_org[:,:,0].mean(axis=1)`（每个检测的 2D 关键点平均 x）跟 `our_screen_x` 做 2×2 匈牙利（退化成比较 identity vs swapped 两种代价和）。**这保证了 `bev_smplx_*[i]` 和 `pgt_smplx_*[i]` 指的是同一个人**——这一点很重要，后面 `single.py` 用同一套 `idxs` 索引两组字段时靠的就是它。

### 2.5 SMPL → SMPL-X `bev_to_smplx()` (L131)
逐检测：
- `smpl_betas_scale = bev['smpl_betas'][d]`，前 10 维是 betas，最后一维是 **scale（儿童程度）**
- `smpl_scale > 0.8` → 走 SMIL 转换器（当儿童），否则走 smpla 转换器
- `body_pose = smpl_thetas[3:][:63]`（SMPL 的 23 关节截成 SMPL-X 的 21 关节，直接截断是原仓库的做法）
- `transl = -root_trans + cam_trans`：BEV 给的是相机平移，减去该 betas 下的根关节位置，换算成 SMPL-X 的 `transl`

### 2.6 失败处理 `process_sample()` (L183)
`bev_out is None` 或 `cam_trans.shape[0] != 2` → **`information_missing=True`，样本保留不删**，交给下游 `allow_missing_bev` 决定。

### 2.7 断点续跑
`atomic_save()`（写 `.tmp` 再 `os.replace`）+ `processed_bev_progress.pkl` 记录已完成 take，每 200 个 take 存一次盘。全部跑完删掉 progress 文件。

### 2.8 【实测】阶段②产物核对

```
information_missing: 24339 / 95448 = 25.5%
bev_smplx_transl 全零的样本数: 24339   ← 与 information_missing 完全一致 ✔
拿到真实 BEV 的样本: 71,109
```
两个数字完全对上，说明"失败即保持全零占位"的不变量成立，没有半成品样本。

**25.5% 失败率合理吗？** 合理但值得注意：失败模式全是"两人被误检成 1 个人"。这是"大面积接触"数据本身的必然副作用——两人贴得越紧，2D 投影上越难分开。它带来的**选择偏差**是真实的：被 BEV 条件版训练用到的 71,109 帧，系统性地偏向"接触没那么极端"的姿态。

---

## 三、阶段②'：数据体检（三个独立工具）

### 3.1 穿模检测 `check_interx_penetration.py`
**入口**：[llib/data/preprocess/utils/check_interx_penetration.py](llib/data/preprocess/utils/check_interx_penetration.py)

- 算法：`winding_numbers(v1, t2_lowres) >= 0.99` 判断 A 的顶点是否在 B 的闭合网格内部；对内部顶点算到对方表面的最近距离，取最大值 = 穿透深度。查询顶点保持全分辨率 10,475，目标网格降到 `lowres=500`。
- **刻意不调用 `MaxIntersection.forward_batch()`**：那里 [contact.py:190](llib/utils/metrics/contact.py#L190) 有裸 `except: import ipdb; ipdb.set_trace()`，无人值守跑 9.5 万帧一旦命中会静默挂死。这里复用它的安全 helper（`prep_mesh`/`close_mouth`/`to_lowres`），自己重写打分循环（`score_batch()` L55）。
- 每帧记录 `min_dist / max_v1_in_v2 / mean_v1_in_v2 / max_v2_in_v1 / mean_v2_in_v1 / max_penetration / has_penetration`
- 排除清单存的是 **`imgname` 字符串集合**（不是 `take_id+下标`），因为 `imgname` 全局唯一、能挺过缓存展平重载
- 有 report 缓存：换阈值重出清单不用重跑 GPU

**【实测】全量 95,448 帧的真实分布**（我刚从 `penetration_report.pkl` 读出来的）：

```
有穿模的帧: 75.5%
穿透深度分位数（含 0 值）: p50=3.62cm  p75=5.89cm  p90=8.14cm  p95=9.43cm  p99=11.27cm  max=15.61cm
不同阈值排除比例:
  0.5cm → 排除 75.3%      3cm → 排除 57.2%
  1cm   → 排除 73.9%      5cm → 排除 33.2%
  2cm   → 排除 66.5%      8cm → 排除 10.7%
```

⚠️ 这跟 `EXECUTION_PLAN.md` 里记的 50-take 小规模结果（64.9%，p50=2.07cm）**明显偏乐观**——全量数据的穿模比全量更普遍也更深。以本报告的数字为准。

⚠️ **当前磁盘上的排除清单是用阈值 0 生成的**：【实测】`penetration_exclude_list.pkl` 有 **72,060** 条 = 95,448 × 75.49%，正好等于 `has_penetration` 的帧数。也就是说"只要有任何穿模就排除"。这个选择很激进（见第八节 B）。

### 3.2 去穿模 `depenetrate_interx.py`
**入口**：[llib/data/preprocess/utils/depenetrate_interx.py](llib/data/preprocess/utils/depenetrate_interx.py)

对被标记穿模的帧做逐帧 Adam 优化：
`loss = reg·(||b1-β1||² + ||b2-β2||²) + pen·L_winding`
- 优化 `transl + global_orient + body_pose`，**`betas` 全程冻结**（防止靠"缩小身体"来消除穿模）
- `L_winding` 复用 [llib/losses/contact.py::GeneralContactLoss](llib/losses/contact.py)（跟检测同一套绕数机制，这里当可微目标用）
- 干净帧原样拷贝
- **写新文件 `processed_depenetrated.pkl`，绝不动 `processed.pkl`**

【实测】目前只跑过 10-take 试验：`diagnostics/depenetration_report_test10.pkl` + `processed_test10.pkl`，**没有全量产物**。这条路线目前是搁置状态。

### 3.3 抽查可视化 `vis_interx_bev.py`
[vis_interx_bev.py](llib/data/preprocess/utils/vis_interx_bev.py)：把同一个样本的 `pgt_smplx_*` 和 `bev_smplx_*` 并排渲染，人物 0=蓝、1=红。**两边各自独立居中、独立自动变焦**——因为 pgt 在 mocap 世界系、bev 在合成相机系，绝对位置根本不可比，只能比"每个人的姿态"和"两人相对布局"。这句话在代码 docstring 里写得很清楚，也正是第八节 A 那个问题的源头。产物在 `diagnostics/bev_vis/`（目前 2 张）。

---

## 四、阶段③：训练侧数据加载

### 4.1 `InterX` 加载类
**入口**：[llib/data/preprocess/interx.py](llib/data/preprocess/interx.py)

`__init__`：读 `train_val_split.npz[split]` 拿 take id 列表，读整个 `processed.pkl`（163MB，**每个 split 都完整读一遍**）。

`load()` 顺序（这个顺序很重要）：
1. 若 `{split}_diffusion.pkl` 存在且 `load_from_scratch=False` → **直接吃缓存**
2. 否则按 take id 展平 `processed[take_id]` 成一个大 list，写缓存
3. `allow_missing_information=False` → 丢 `information_missing=True` 的
4. `filter_penetration=True` → 按 `imgname` 减去排除清单；**清单不存在直接 `FileNotFoundError`**（不会静默用未过滤数据训练，这个设计是对的）
5. `overfit` → 截前 N 条

⚠️ **缓存陷阱**：第 1 步的缓存是在过滤之前，所以切 `allow_missing_bev` / `filter_penetration` **不需要删缓存**；但是如果你**重跑了阶段①或②改动了 `processed.pkl`，必须手工删掉 `{train,val}_diffusion.pkl`**，否则会拿旧数据训练且毫无提示。`process_interx_bev.py` 结尾专门打了一行提醒。

【实测】当前缓存是好的：`train_diffusion.pkl` 里 63,738 条有非零 bev 字段，说明缓存是在 BEV 阶段跑完之后重建的。

### 4.2 `SingleDataset`
**入口**：[llib/data/single.py](llib/data/single.py)

`load_data()` (L121)：`dataset_name == 'interx'` 分支 (L149-158) → `InterX(**dataset_cfg, split, body_model_type).load(processed_fn_ext='_diffusion.pkl', allow_missing_information=self.dataset_cfg.allow_missing_bev)`。
注意 `filter_penetration` 是通过 `**self.dataset_cfg` 自动透传的（dataclass 里加字段即可，不用改 `single.py`）。

`get_single_item()` (L333) 逐步：
1. `image_processing.load_image=False` → `input_image=[0.0]`，**不读图**（本来也没图）
2. `swap = self.augm_params_threed()`：`split=='train' and augmentation.use` 时以 `augmentation.swap=0.5` 概率置 1
   ⚠️ **`datasets.augmentation.use=True` 在扩散训练里只启用这一个 swap**。`_augm_params()`（mirror/noise/rotation/scale）在 L388 是**注释掉的**，完全没被调用。
3. `action_to_class_id.json` 不存在 → `action=-1, action_name=''`
4. **人物排序**：`h0id = 0 if transl_param[0][0] <= transl_param[1][0] else 1`，即按 **transl 的 x 分量**排。对 InterX 用的是 `pgt_smplx_transl`，也就是 **mocap 世界系的 x**。然后 `swap` 会再翻一次。
   ⚠️ 对 FlickrCI3D/CHI3D/Hi4D，这个 x 是**相机系**的 x，语义是"画面左边的人排第一"；对 InterX 它是世界系 x，跟画面左右无关，是个**任意但确定**的规范化。不算 bug（同一套 `idxs` 同时索引 pgt 和 bev，人物身份始终一致），但它没起到原本"左右规范化"的作用。
5. InterX 分支 (L526-539) 组装 `human_target`，键名从 `pgt_smplx_*` / `bev_smplx_*` 改成 `pgt_*` / `bev_*`
6. `to_tensors()` 递归转张量

`set_feature_vec()` (L98)：把 `DatasetFeatures` 的布尔广播成 `[N]` 数组塞进每个 batch。InterX 设了 `has_gt_smpl_pose/shape=True`——⚠️ 这两个 flag 在扩散训练里**当前没有任何消费方**（`train_module` 从不读），照抄进新数据集时不要指望它有作用。

### 4.3 多数据集混采
[llib/data/collective.py](llib/data/collective.py) 的 `CollectivDataset` 把多个 `SingleDataset` 串起来，`PartitionSampler` 按 `train_composition` 决定**每个 batch 内**各数据集占多少条（`round(partition × batch_size)`，凑不齐 batch_size 就从第一个数据集补/减）。
`__len__` 是所有数据集长度之和，`num_batches = sum(lengths) / batch_size`。这里只有 interx 一个、composition=[1.0]，所以退化成普通打乱。

[llib/data/build.py::build_datasets](llib/data/build.py#L71)：train 建一个 `CollectivDataset`；**val 是每个数据集单独建一个**，存成 `{name: ds}` 字典，方便分数据集报验证指标。

### 4.4 【实测】各配置下的真实数据量

| 配置 | train | val |
|---|---|---|
| 缓存原始（无过滤） | 85,714 | 4,863 |
| `allow_missing_bev=False` | **63,738** | 3,650 |
| `filter_penetration=True`（阈值 0） | **21,053** | 1,287 |
| 两个都开 | **17,069** | 1,038 |

交叉验证：checkpoint 文件名里的 batch_idx 就是每 epoch 的 step 数——`interx_uncond` 是 167（85714/512=167.4 ✔）、`interx_uncond_clean` 是 41（21053/512=41.1 ✔）、`interx_cond_bev` 是 124（63738/512=124.5 ✔）。**三个 run 的实际数据量与上表完全对得上。**

---

## 五、阶段④：`train_module.py` 里的参数变换

**入口**：[llib/methods/hhc_diffusion/train_module.py](llib/methods/hhc_diffusion/train_module.py)

`single_training_step()` (L1064) 的执行顺序值得注意——**guidance 先算，target 后算**：

```python
guidance_params = self.get_guidance_params(batch)          # 先用 bev_* 造条件
batch = self.cast_smpl(self.preprocess_batch(batch))       # 再把 pgt_* 变换成 target
target_params = self.get_gt_params(batch)
target_smpls  = self.get_smpl(target_params)
t, weights = self.schedule_sampler.sample(self.bs, "cuda") # 均匀采 t ∈ [0,1000)
diffusion_output = self.diffuse_denoise(x=target_params, y=guidance_params, t=t)
```

### 5.1 `preprocess_batch()` (L283)
`prefix = exp_cfg.in_data`（配置里是 `pgt`）：
```
orient  = batch['pgt_global_orient']                                  [B,2,3]
pose    = batch['pgt_body_pose']                                      [B,2,63]
shape   = cat(batch['pgt_betas'], batch['pgt_scale'], axis=2)         [B,2,11]
transl  = batch['pgt_transl']                                         [B,2,3]
action  = batch['action'][:,None].float()                             [B,1]  (=-1，未用)
```
`get_guidance_params` 里以 `in_data='bev'` 再走一遍同样的逻辑。

### 5.2 `cast_smpl()` (L404) —— 关键变换
配置：`rotrep=sixd`、`relative_orient=false`、`relative_transl=true`

- **orient**：`relative=False` → `cam_rotation=None`，**不做任何坐标系归一化**，只是轴角 → 6D 旋转表示，`[B,2,3] → [B,2,6]`
- **pose**：`[B,2,63] → [B,2,21,3] → [B,2,126]`（21 个关节各 6D）
- **transl**：`relative=True` 且 `cam_rotation is None` → 走 else 分支：
  ```python
  transl[:,1,:] -= transl[:,0,:]
  transl[:,0,:]  = 0.0
  ```
  即 **h0 的平移被清零，h1 存的是相对 h0 的偏移向量**。绝对位置信息被完全丢弃，模型只建模"两人相对布局"。
  ⚠️ 这是**原地操作**，直接改 batch 张量。训练里每个 batch 是新的所以没事；`single_validation_step` 用了 `clone=True` 规避。
- **shape**：不变，`[B,2,11]`

`relative_orient=false` 意味着**模型要建模两个人的绝对朝向**（在各自数据集自己的坐标系里），而不是只建模相对朝向。`unit_rotation`（绕 X 转 π）只在 `relative_orient=True` 时才用得上，当前配置下是死代码。

### 5.3 `get_guidance_params()` (L453)
- `is_full_bev_guidance = 'bev' in guidance_params`
- 无条件版 `guidance_params=[]` → 直接 `pass`，返回空 dict。**此时 `bev_*` 字段虽然被 dataloader 读进了 batch，但一行也不会被用到**——这就是阶段①敢全填零的依据。
- BEV 条件版 → `split_humans(cast_smpl(preprocess_batch(batch, in_data='bev')))`，得到 8 个键：`orient_h0/pose_h0/shape_h0/transl_h0/orient_h1/...`
- **classifier-free guidance 的随机丢弃**：每个样本抽 `all_or_none ~ U(0,1)`
  - `≤ guidance_all_nc(0.2)` → `noise_chance=1.0`（整条全部置 0 → 无条件）
  - `(0.2, 0.9]` → `noise_chance=0.0`（条件完整保留）
  - 剩下 10% → `noise_chance = guidance_param_nc(0.5)`，逐参数各 50% 概率置 0
  - 置的是 `null_value = 0.0`（不是随机噪声，尽管变量名叫 nc = noise chance）

### 5.4 `split_humans()` / token 布局
`token_setup: H0PH1P` → **每人 4 个 token**（orient/pose/shape/transl），共 8 个目标 token。
（另一种 `H0H1` 会把每人 4 个参数拼成 1 个 146 维 token，共 2 个，当前没用。）

---

## 六、阶段⑤：模型与扩散过程

### 6.1 BUDDI Transformer
**入口**：[llib/models/regressors/buddi.py](llib/models/regressors/buddi.py)

配置只给了 `dim=152, depth=6, heads=8, mlp_dim=500, dropout=0.1, use_human_embedding=true, use_param_embedding=true`，**`embed_guidance_config` / `embed_target_config` 用的是函数签名里的默认字典**（L117-128）——`build_buddi(model_cfg.diffusion_transformer)` 只是 `BUDDI(**cfg)`，配置里没这两个键就吃默认值。

`forward()` (L228) 的 token 流：
```
目标 token  ×8 : embed_input_{orient_h0,pose_h0,shape_h0,transl_h0, ...}  (6/126/11/3 → 152)
时间 token  ×1 : TimestepEmbedder(sinusoidal → MLP)
条件 token  ×8 : embed_guidance_{同名键}                                   (BEV 条件版才有)
        ↓ 每个 token 加上 human_embedding_h{0,1} + param_embedding_{orient,pose,shape,transl}
        ↓ concat → [B, 9 或 17, 152]
        ↓ nn.TransformerEncoder(depth=6, heads=8)
        ↓ unembed_input_{k}(xx[:, idx])  → 回到 6/126/11/3 维
```

几个**必须知道的架构事实**：
1. **完全没有位置编码**（`use_positional_encoding` / `use_positional_embedding` 都是 false）。token 的身份**只靠**"用哪个线性层投影" + "加了哪个 human/param embedding"来区分。Transformer 本身对 token 顺序是置换不变的。
2. **条件 token 和目标 token 用的是同名的键**（都叫 `orient_h0` 等）。因为 `share_linear_layers=False`，走的是 `embed_guidance_orient_h0` 这套**独立权重**，所以两者能区分开——但它们加的 human/param embedding 是**同一个**。区分目标与条件的唯一信号就是那对线性层的权重差异。
3. 默认 config 里的 `orient_bev_h0 / pose_bev_h0 / ...`、`human_h0/h1`、`action`、`contact`(5625维) 这些 embedder **全部被创建但永远不会被调用**——是死参数，白占显存和 optimizer state（`contact` 那个是 5625×152 ≈ 85 万参数）。

### 6.2 扩散过程
配置 `model.diffusion`：`steps=1000, noise_schedule=cosine, model_mean_type=start_x, model_var_type=fixed_large, loss_type=custom`

`diffuse_denoise()` (L984)：
1. 对 8 个参数各自 `input_noise = randn_like`，`q_sample(v, t, noise)` 加噪
2. `split_humans` → 喂给 transformer，`timesteps = diffusion._scale_timesteps(t)`（`rescale_timesteps=false` 所以是恒等）
3. `model_mean_type == 'start_x'` → 模型**直接预测 x_0**，`denoised_tokens = pred`
4. `concat_humans` → `get_smpl()` 前向出网格

⚠️ 注意加噪是**在 6D 旋转 / 相对平移 / betas+scale 这个混合空间里各向同性地加**的：`orient` 6 维、`pose` 126 维、`shape` 11 维、`transl` 3 维，量纲完全不同（6D 旋转分量 ~O(1)，transl ~O(0.5m)，betas ~O(0.5)），但共用同一个噪声 schedule。这是原版 BUDDI 的设计，不是这次改的，但在你判断"loss 数值合不合理"时要记得。

### 6.3 Loss
**入口**：[llib/methods/hhc_diffusion/loss_module.py](llib/methods/hhc_diffusion/loss_module.py)

`start_x` 模式下损失**不在噪声空间算，而在 SMPL 空间算**（`criterion(est_smpl=denoised_smpls, tar_smpl=target_smpls, ...)`）：

| loss | 计算 | 权重（uncond / cond_bev 实跑） |
|---|---|---|
| `pseudogt_pose` | body_pose 和 global_orient 各转旋转矩阵后 MSE，`.sum((1,2,3)).mean()` | 10.0 / 10.0 |
| `pseudogt_shape` | betas L2 + `(scale差)²` | 0.1 / 0.1 |
| `pseudogt_transl` | transl L2 | 1.0 / 1.0 |
| `pseudogt_v2v` | 顶点欧氏距离平方 `.sum(-1).mean(-1).mean()` | 1.0 / **1000.0** |
| 其余（contact/cmap/prior/j2j…） | 权重 0，未启用 | — |

⚠️ **loss 里完全没有 t 相关的加权**——t=999 的样本和 t=1 的样本贡献同权重。
⚠️ **`interx_uncond` 和 `interx_cond_bev` 的 v2v 权重差 1000 倍，两个 run 的 loss 数值不可直接比较**（uncond ckpt 指标 ~12.8，cond_bev ~33.8，这个差异大部分来自权重而非质量）。
✅ NaN 处理已经修好：原来是 `ipdb.set_trace()`（无人值守会挂死），现在是打印 per-loss 明细后 `raise RuntimeError`（L265-272）。

### 6.4 训练循环
**入口**：[llib/training/diffusion_trainer.py](llib/training/diffusion_trainer.py)

`train_one_epoch()` (L282)：每个 epoch 新建 `PartitionSampler` + `DataLoader`（`shuffle=False`，打乱在 sampler 里做）→ 前向 → `optimizer.zero_grad()` → `backward()` → `clip_grad_norm_(1.0, error_if_nonfinite=True)` → `step()`。

- **单卡、无 AMP、无梯度累积**，`batch_size=512` 是真实 batch
- `summaries_freq` / `checkpoint_freq` 的单位是 **epoch 不是 step**：`checkpoint_steps = ckpt_freq × steps_per_epoch`（[diffusion_trainer.py:160](llib/training/diffusion_trainer.py#L160)）。配置里的 `100.0` = **每 100 个 epoch** 存一次盘并验证一次，约合 16,700 步 / 5 小时。这也解释了已有 run 的 checkpoint 为什么是 1399/1499/1599 这种间隔
- `main.py` 开了 `torch.autograd.set_detect_anomaly(True)` ⚠️ 这会**显著拖慢训练**，是调试残留
- `max_epochs=5000`，**没有 early stopping**
- ✅ `validate()` **已修复**（原来只跑 `batch_idx == 0` 那一个 batch）：现在遍历整个验证集累积指标，并用固定种子 42 包住（前后保存/恢复 torch+numpy+cuda RNG 状态）。昂贵的两个采样循环和 tensorboard 渲染仍只在第一个 batch 跑，`evaluation.max_val_batches` 可限流（默认 -1 = 全量）。
  实测（interx_cond_bev epoch 1599 checkpoint，同一验证集）：旧方案 5 次重复 16.615~17.100（std 0.178，变异 1.05%），新方案 3 次全部 16.515（std 0）；逐 batch 指标在 15.2~19.1 之间，batch 0 读数 17.1 而全集真值 16.5。单次验证耗时 19.65s → 31.9s（不带采样的批 1.53s/批），相对"每 100 epoch 才验证一次"可忽略
- checkpoint 命名：`{日期}__{epoch:010d}__{batch_idx:010d}__{val_metric:.2f}.pt`

---

## 七、当前三个 run 的真实状态

| run | guidance | allow_missing_bev | filter_penetration | 训练帧数 | 最新 ckpt |
|---|---|---|---|---|---|
| `interx_uncond` | `[]` | 默认 True | 默认 False | 85,714 | epoch 1999, metric 12.81（08-02） |
| `interx_uncond_clean` | `[]` | True | **True** | 21,053 | epoch 1999, metric 17.84（08-05） |
| `interx_cond_bev` | `['bev']` | **False** | False | 63,738 | epoch 1599, metric 33.80（08-09） |

【实测】`interx_cond_bev` 的 screen 会话 (`SCREEN -dmS interx_cond_bev`) **还在**，但最新 checkpoint 停在 08-09，`train_resume.log` 是空的。开跑时**没有**加 `filter_penetration=True`（跟 `EXECUTION_PLAN.md` 第五节里那条"待办"命令不一致），所以它训练的是**未做穿模清洗**的数据。

---

## 八、我在追踪过程中发现的问题（按严重程度排序）

### A. 🔴 BEV 条件与训练目标不在同一坐标系 —— 会让 BEV 条件版训练在朝向/布局上学不到东西

**机制**：
- `target`（`pgt_smplx_*`）是 **Inter-X mocap 世界坐标系**下的参数，且 `relative_orient=false` 意味着这个绝对朝向被原样送进模型。
- `guidance`（`bev_smplx_*`）是 BEV 在**每一帧各自随机采样的合成相机坐标系**下的估计。
- 两者之间差一个**逐样本随机**的相机旋转（yaw 在 ±150° 内均匀）。
- `relative_transl=true` 只是把 h1 的平移变成相对 h0 的偏移向量——**偏移向量的方向仍然是坐标系相关的**，所以这一步不能挽救什么。

对比：原版 BUDDI 里，FlickrCI3D 的 pgt 是在相机系里拟合的，CHI3D 用 `global_orient_cam`/`transl_cam`，Hi4D 用 `global_orient_cam_smplx`/`transl_cam_smplx`——**target 和 BEV 条件共享同一个相机系**。这条 InterX 支线是唯一一个没有做 `_cam` 变换的。

**【实测】证据**（71,109 个有真实 BEV 的样本里随机抽 4000）：

```
两人平移偏移向量 (t1-t0) 的方向夹角  pgt vs bev :
    mean=98.6°   p10=31°   p50=101°   p90=162°      ← 近似均匀分布 = 两个无关坐标系
人物0 绝对朝向差 |R_bev⁻¹R_pgt|      : mean=172°, p10=163°, p90=179°
    （恒接近 180° 是因为"绕X翻180° ∘ 任意绕上轴 yaw"的合成角恒为 180°，
      这正好印证了"翻转 + 随机 yaw"的关系，而不是印证对齐）

作为对照，坐标系无关的量确实是对上的：
两人相对朝向角之差 |pgt - bev|       : p50=7.9°, p90=25.2°      ← BEV 姿态估计本身是准的
两人距离 ||t1-t0||  pgt vs bev       : 0.474m vs 0.538m, 相关系数 0.455
```

**结论**：BEV 估计的**姿态质量本身没问题**（相对朝向中位数只差 8°），问题纯粹出在**没有把 pgt 变换到渲染相机系**（或反过来）。模型拿到的 `orient_h0/transl_h1` 条件与目标之间没有确定的函数关系，只能学到边缘分布，条件信号在这两个 token 上基本作废。而阶段③（用 BUDDI 做 SDS 先验优化真实照片）恰恰依赖"条件与目标共享相机系"这个假设。

**修法方向**（不在本报告范围内实施）：在 `process_interx_bev.py::render_sample()` 里已经知道相机的 R/T，顺手把 `pgt_smplx_global_orient/transl` 也变换到该相机系，另存成 `pgt_smplx_global_orient_cam` / `pgt_smplx_transl_cam`，然后在 `single.py` 的 InterX 分支里改用 `_cam` 版本作为 target（跟 CHI3D/Hi4D 分支的做法完全一致）。注意这会让 target 依赖阶段②，所以无条件版要么保留旧字段、要么一起切。

### B. 🟠 穿模排除清单用的是阈值 0，等于"有任何穿模就扔"
【实测】排除 72,060/95,448 = 75.5%，训练集只剩 21,053 帧。SMPL-X 是刚性网格、不模拟衣物和皮肤压缩，真实接触天然带几厘米浅穿插——按 0 阈值过滤会把大量**正常的接触姿态**一起扔掉，而"大面积接触"正是你专门筛出来的目标。`EXECUTION_PLAN.md` 里原本的计划是"看全量分布后由你选阈值"，实际落地成了 0。参考实测分布，5cm 排除 33.2%、8cm 排除 10.7%，是更值得考虑的档位。

### C. 🟠 BEV 的 betas 条件几乎不含信息
【实测】
```
pgt betas 逐维 std: [0.73 0.51 0.53 0.07 0.07 0.09 0.04 0.04 0.04 0.05]
bev betas 逐维 std: [0.36 0.16 0.46 0.48 0.17 0.36 0.16 0.31 0.18 0.25]
betas 第 0 维相关系数: 0.139
```
两点：(1) Inter-X 的 betas 从第 4 维起基本退化（std<0.1），体型多样性很低；(2) BEV 从合成灰模图估出来的 betas 跟真值几乎不相关。`shape` token 的条件信号基本是噪声。

### D. ✅ 25.5% 的 BEV 失败带来系统性选择偏差 —— 已修复，改用 `allow_missing_bev=True`
失败模式全是"两人被识别成一个人"，所以 `allow_missing_bev=False` 筛掉的不是随机样本，而是**接触最紧密的那批**。

但直接把开关打开是**不安全**的：缺 BEV 的样本 `bev_*` 是全零，而全零**过不了** `cast_smpl`——零轴角转 6D 得到 `[1,0,0,0,1,0]`，即**单位旋转**，一个看起来完全合法的假朝向；零 shape token 则是平均体型。这等于给 25% 的样本喂假条件。已实测确认：未打补丁时 missing 行的 `orient_h0` 正是 `[1,0,0,0,1,0]`。

修法（已实施）：
- [llib/data/single.py](llib/data/single.py) 的 `gen_target` 里加 `information_missing`（`item.get(...)`，其他数据集自动为 False）
- [train_module.py::get_guidance_params](llib/methods/hhc_diffusion/train_module.py) 在构造完 BEV guidance 后，把 `information_missing` 的行整条置为 `null_value`(0)，让这些样本以**无条件样本**参与训练——`null_value` 正是 classifier-free guidance 掩码用的值，模型本来就认识（`guidance_all_nc=0.2` 每步都在这么做）
- 顺带修了 `Trainer.expand_batch` 的 `torch.zeros` 没带 `dtype`（bool 字段会让 `torch.cat` 直接报错）

实测：val 首个 batch 512 条里 166 条 missing，其 guidance 全部精确为 0，非 missing 行保持非零。

### E. 🟡 gender 信息被丢弃
Inter-X 每人带 gender 标注，预处理直接丢掉，betas 被塞进 neutral 模型。已在 `EXECUTION_PLAN.md` 记录，未解决。

### F. 🟡 若干"看起来在做、其实没做"的东西
- `datasets.augmentation.use=True` 在这条训练路径上**只启用 person swap**，mirror/noise/rotation/scale 的代码 (`_augm_params`) 是注释掉的
- `DatasetFeatures.has_gt_smpl_pose/shape=True` 没有任何消费方
- `imgpath` / `img_height=900` / `action=-1` 全是占位，不参与计算
- `main.py` 的 `torch.autograd.set_detect_anomaly(True)` 是调试残留，白白拖慢训练
- 模型里 `orient_bev_h0` 等 8 个 embedder + `human_h0/h1` + `action` + `contact`(5625×152) 全是死参数
- `validate()` 只评估验证集第一个 batch

---

## 九、你亲自追踪的验证清单

按顺序做，每一步都给了"该看到什么"。所有命令在仓库根目录、`conda activate buddi`、`export PYTHONPATH=$(pwd)` 下跑。

### ① 原始数据
```bash
python -c "
import numpy as np
d=np.load('datasets/original/InterX/motions/G001T000A000R000/P1.npz',allow_pickle=True)
for k in d.files: print(k, getattr(d[k],'shape',None), d[k].dtype)
print('gender', d['gender'])"
```
→ 应看到 `pose_body (T,21,3)`、`betas (1,10)`、`root_orient (T,3)`、`trans (T,3)`。

### ② 分段阈值是否合理
```bash
python llib/data/preprocess/utils/diagnose_interx.py --sample-takes 200
```
→ 看根平移距离分布，以及 `dist_thresh=0.6, stride=20` 这一格能留下多少帧；对照实际的 4735 take / 95,448 帧。

### ③ 阶段①产物结构
```bash
python - <<'EOF'
import pickle,numpy as np,itertools
p=pickle.load(open('datasets/processed/InterX/processed.pkl','rb'))
print('takes',len(p),'frames',sum(len(v) for v in p.values()))
s=next(iter(p.values()))[0]
for k,v in s.items(): print(k, getattr(v,'shape',v))
EOF
```
→ 对照本报告 1.4 节的字段表。

### ④ 阶段②：BEV 是否真的写进去了 / 失败率
```bash
python - <<'EOF'
import pickle,numpy as np,itertools
p=pickle.load(open('datasets/processed/InterX/processed.pkl','rb'))
S=list(itertools.chain.from_iterable(p.values()))
miss=sum(s['information_missing'] for s in S)
zero=sum(np.abs(s['bev_smplx_transl']).sum()==0 for s in S)
print(f'{miss}/{len(S)} missing ({miss/len(S)*100:.1f}%), bev全零 {zero}')
EOF
```
→ 两个数应**完全相等**（当前 24,339）。不相等说明有半成品样本。

### ⑤ 肉眼查 BEV 质量（最重要的一步）
```bash
python llib/data/preprocess/utils/vis_interx_bev.py --num-samples 12 \
  --out-dir datasets/processed/InterX/diagnostics/bev_vis
```
→ 左右并排看 pgt vs bev。**注意**：因为两边各自独立居中变焦，你只该比较"每个人的姿态像不像"和"两人相对布局像不像"，**不要**指望朝向一致——朝向不一致正是第八节 A 描述的问题，这张图本身就是最直观的证据。

### ⑥ 亲手复现坐标系问题（第八节 A 的证据）
```bash
python - <<'EOF'
import pickle,numpy as np,itertools
from scipy.spatial.transform import Rotation as R
p=pickle.load(open('datasets/processed/InterX/processed.pkl','rb'))
S=[s for s in itertools.chain.from_iterable(p.values()) if not s['information_missing']]
rng=np.random.RandomState(0); idx=rng.choice(len(S),2000,replace=False)
ang=[]
for i in idx:
    s=S[i]
    tp=s['pgt_smplx_transl'][1]-s['pgt_smplx_transl'][0]
    tb=s['bev_smplx_transl'][1]-s['bev_smplx_transl'][0]
    c=np.dot(tp,tb)/(np.linalg.norm(tp)*np.linalg.norm(tb)+1e-9)
    ang.append(np.degrees(np.arccos(np.clip(c,-1,1))))
ang=np.array(ang)
print('两人偏移向量夹角 pgt vs bev: mean=%.1f p50=%.1f'%(ang.mean(),np.percentile(ang,50)))
EOF
```
→ 若共享坐标系应集中在 0° 附近；实际 p50 ≈ 101°。

### ⑦ 穿模分布与阈值选择
```bash
python llib/data/preprocess/utils/check_interx_penetration.py \
  --processed-data-folder datasets/processed/InterX
```
→ 会**复用缓存的 report**（几秒出结果，不重跑 GPU），打印本报告 3.1 节那套分位数和各阈值排除比例，并出 `diagnostics/penetration_depth_hist.png`。想换阈值：加 `--exclude-threshold 0.05`（米）重生成清单。

### ⑧ 训练侧真实拿到多少数据
```bash
python - <<'EOF'
import pickle,numpy as np
P='datasets/processed/InterX'
ex=pickle.load(open(P+'/diagnostics/penetration_exclude_list.pkl','rb'))
for sp in ['train','val']:
    d=pickle.load(open(f'{P}/{sp}_diffusion.pkl','rb'))
    miss=sum(x['information_missing'] for x in d)
    pen=sum(x['imgname'] in ex for x in d)
    print(sp,'total',len(d),'| allow_missing_bev=False→',len(d)-miss,'| filter_penetration=True→',len(d)-pen)
EOF
```
→ 对照第四节 4.4 的表；再对照 `ls demo/diffusion/training/<run>/checkpoints/` 里文件名的第三段（每 epoch step 数 × 512 ≈ 训练帧数）。

### ⑨ 单个 batch 走到模型输入前长什么样
```bash
python - <<'EOF'
import sys; sys.path.insert(0,'llib/methods/hhc_diffusion')
from llib.defaults.main import config as dc, merge
import argparse
class A: exp_cfgs=['llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml']; exp_opts=[
 "datasets.train_names=['interx']","datasets.train_composition=[1.0]","datasets.val_names=[]",
 "datasets.interx.processed_data_folder=datasets/processed/InterX","datasets.interx.allow_missing_bev=False"]
cfg=merge(A(),dc)
from llib.data.build import build_datasets
tr,_=build_datasets(datasets_cfg=cfg.datasets, body_model_type='smplx', build_val=False)
b=tr[0]
for k,v in b.items():
    print(f'{k:22s}', tuple(v.shape) if hasattr(v,'shape') else v)
EOF
```
→ 应看到 `pgt_global_orient (2,3)` / `pgt_body_pose (2,63)` / `pgt_scale (2,1)` / `bev_scale (2,1)` 等；确认 pgt 和 bev 的 scale **都是 (2,1)**。

### ⑩ 训练曲线
```bash
tensorboard --logdir demo/diffusion/training
```
→ 三个 run 一起看。记住 `interx_cond_bev` 的 v2v 权重是 1000，**跟另外两个的 loss 绝对值不可比**；要比就看 `train/pseudogt_pose_losses_*` 这类单项。

---

## 十、给你读代码的建议顺序

1. `process_interx.py`（180 行，最简单，先建立数据形状的直觉）
2. `interx.py` + `single.py::load_data/get_single_item` 的 InterX 分支（看清楚字段怎么改名、人物怎么排序）
3. `process_interx_bev.py`（渲染 → BEV → 配对 → 转换四步，配合 ⑤ ⑥ 两个验证一起看）
4. `train_module.py::single_training_step → preprocess_batch → cast_smpl → get_guidance_params`（参数变换的全部真相都在这 200 行里）
5. `buddi.py::forward`（token 布局）
6. `loss_module.py::forward` + `diffusion_trainer.py::train_one_epoch`

读到第 3 和第 4 步时会自然撞上第八节 A 那个坐标系问题——`cast_smpl` 里 `relative_orient=false` 那条 `RR = None` 分支，和 `process_interx_bev.py` 里 `verts_c = verts - center` 之后再摆随机相机，这两处放在一起看就清楚了。
