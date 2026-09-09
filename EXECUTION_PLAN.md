# 用自有 Inter-X 格式 SMPL-X 数据重训练 BUDDI 扩散模型 —— 执行计划与进度记录

> 本文件是这次重训练工作从头到尾的**执行计划 + 真实进度记录**（不是纯设计稿——每个阶段标注了"计划"还是"已验证/已完成"）。原始文件是 plan mode 下产出的规划文档，现在挪到仓库根目录，后续每个阶段的真实结果都会追加更新在这里，方便随时回看全局进度。

## 当前状态速览（最新）

| 阶段 | 状态 |
|---|---|
| ① 无条件版训练数据预处理（自动分段+抽帧，`dist_thresh=0.6,stride=20`） | ✅ 已完成，95,448 帧 |
| ① 训练代码接入（`InterX` dataset class + `single.py` 分支） | ✅ 已完成 |
| 追加①：训练可视化/健壮性 bug 修复（4 处） | ✅ 已完成 |
| 无条件版训练（`interx_uncond`） | 🟡 后台跑着（用户自己盯 tensorboard） |
| 追加②：BEV 条件版训练数据生成（随机相机渲染 + 真实 BEV 推理） | ✅ 已完成，71,109/95,448 成功 |
| 追加③：穿模检测 + 数据清洗 | 🟡 全量扫描进行中（见下方"阶段③执行更新"） |
| BEV 条件版训练 | ⏸️ 未开始，命令已就绪，等穿模清洗定阈值后再跑 |

---

## Context

用户的数据是 **Inter-X 格式**的双人交互动作捕捉数据（已用 `/home/yuhong/workspace/buddi/interxtest/{P1,P2}.npz` 实际读取验证），规模「几万帧以上」。目标是重训 BUDDI 扩散模型（阶段②），最终想用于阶段③（对新照片做 SDS 优化推理，需要 BEV 条件版）。

**已经确认的决策（不再讨论）**：
- 先不碰 BEV/渲染/摆相机，**先把无条件版（unconditional）训练流程用真值数据跑通、看效果**，BEV 条件版留到之后。
- 2D 关键点、接触标注这两块本来这条训练代码也用不上，不用管。
- 用户的原始数据是"一个 take 里混杂互动和非互动片段"，需要**自动检测互动片段**、**自动抽帧**；自动摆相机是为将来 BEV 阶段预留的设计，本阶段不实现。

### 数据实测结果（直接读取 `interxtest/P1.npz` / `P2.npz` 得到）

```
P1.npz: pose_body (824,21,3)  pose_lhand (824,15,3)  pose_rhand (824,15,3)
        betas (1,10)  root_orient (824,3)  trans (824,3)  gender ()  # 'female'
P2.npz: 同结构                                                        # 'male'
```
这是标准 Inter-X 导出格式：`pose_body` = SMPL-X 21 个身体关节的轴角（展平后就是 `body_pose` 的 63 维）、`root_orient` = `global_orient`、`trans` = `transl`、`betas` 每人每个 take 只有 1 行（体型不随时间变，需要广播到每一帧）。`pose_lhand`/`pose_rhand` 是完整 15 关节手部姿态——**BUDDI 当前训练只用 body_pose/global_orient/betas/transl，不需要手部姿态**，可以直接不读或忽略。`gender` 是逐人标注的性别（这条数据 P1=female, P2=male）——**BUDDI 的 body model 固定用 `gender: neutral`**（各训练配置里都是这个值），gender-specific 拟合出的 betas 放到 neutral 模型下不是严格等价的，这是一个已知的近似误差，先如实记录，不在本计划里解决。

抽样验证过一个直接可行的"互动检测"信号：**两人 `trans`（根关节平移）之间的欧氏距离**。这条 824 帧序列里该距离从 1.63m 逐渐降到 0.34m（实测数据，见下），说明用这个距离做阈值分段是可行的、不需要一开始就上复杂的关节级 forward kinematics。

---

## 一、原始数据组织形式（现状，不用改）

```
datasets/original/InterX/motions/
├── G001T000A000R000/
│   ├── P1.npz
│   └── P2.npz
├── G001T000A000R001/
│   ├── P1.npz
│   └── P2.npz
...（用户已有的完整目录树，脚本直接递归扫描 motions/ 下所有叶子文件夹）
```

每个叶子文件夹（"一个 take"）代表一段连续的双人动作捕捉，**从头到尾可能混杂着互动和非互动的部分**，需要下一节的自动分段逻辑处理。不需要用户手动整理/裁剪/搬动这些文件，脚本直接对着现有目录树跑。

---

## 二、预处理流程（`llib/data/preprocess/utils/process_interx.py`）

### 2.1 自动检测互动片段

对每个 take：
1. 读 `P1.npz`/`P2.npz` 的 `trans`（各 `[T,3]`），逐帧算欧氏距离 `dist[t] = ||trans1[t] - trans2[t]||`。
2. 做一次中值滤波去抖动，避免阈值附近来回抖动导致片段被切碎。
3. 用距离阈值判断"互动中"，取连续满足阈值的帧区间为一个候选片段（最终标定为 `dist_thresh=0.6`，见下）。
4. 过滤掉太短的片段（`min_len=15`），前后各留 `pad=8` 帧 padding，保留互动前后的过渡姿态。
5. 一个 take 可能产出 0 个、1 个或多个互动片段。

**已知局限**：`trans` 距离是根关节（约等于骨盆）间距，对"骨盆分开但手部接触"的动作可能偏保守漏检边界帧。

### 2.2 自动抽帧

对每个检测出的互动片段，按固定步长抽帧（最终标定为 `stride=20`，见下）。

### 2.3 字段映射（Inter-X → BUDDI 训练用的 `pgt_smplx_*`）

| Inter-X 字段 | 处理 | 映射到 |
|---|---|---|
| `pose_body` `[T,21,3]` | reshape 成 `[T,63]` | `pgt_smplx_body_pose` |
| `root_orient` `[T,3]` | 原样 | `pgt_smplx_global_orient` |
| `trans` `[T,3]` | 原样 | `pgt_smplx_transl` |
| `betas` `[1,10]` | 广播到 `[T,10]` | `pgt_smplx_betas` |
| （无） | 填零 `[T,1]` | `pgt_smplx_scale` |
| `pose_lhand`/`pose_rhand`/`gender` | 不使用 | — |

每帧组装成 P1/P2 堆叠的双人样本（`[2, ...]`），`bev_smplx_*` 阶段①先全部填零占位，阶段②（见下）再换成真实值。存成 `datasets/processed/InterX/processed.pkl`。

**✅ 已完成的正式跑批**：`dist_thresh=0.6, stride=20`（用户明确要"大面积接触"的数据，收窄了阈值）→ 4735/8026 个 take 有可用数据，5659 个互动片段，**95,448 条训练样本**；按 take 划分 train=4262/val=237/test=236（90%/5%/5%，`make_split`，`seed=0`，避免同一动作相邻帧的 train/val 泄漏）。

---

## 三、代码里确认过的关键事实

1. 训练监督目标只用 `pgt_*`（SMPL-X 真值参数），2D 关键点、接触标注当前训练代码完全用不上。
2. `llib/data/single.py::get_single_item()` 不管 `guidance_params` 怎么配，都无条件地取 `bev_*` 5 个字段拼进 `human_target`，缺了会 `KeyError`，但 `guidance_params=[]`（无条件版）时这些字段读进 batch 后根本不会被用到。**结论：无条件版阶段 `bev_*` 填零占位即可。**
3. `get_single_item` 的数据集分支靠 `item['imgpath']` 里的子串（`'FlickrCI3D'`/`'CHI3D'`/`'Hi4D'`/`'InterX'`）区分，不是靠 `dataset_name`。
4. `CollectivDataset.__len__` 是所有训练数据集长度之和，`train_composition` 只决定 batch 内比例。
5. 训练纯单卡、无 AMP、无梯度累积，`batch_size=512` 是每步真实 batch。模型很小（Transformer `dim=152, depth=6, heads=8`），瓶颈大概率在 CPU dataloader。
6. `max_epochs: 5000`，训练循环没有 early stopping，靠 `checkpoint_freq` 自己盯着验证 loss 决定何时停。

---

## 四、代码改动清单（阶段①，已完成）

| # | 文件 | 改动 |
|---|---|---|
| 1 | `llib/data/preprocess/utils/process_interx.py` | 离线预处理脚本：自动分段 → 自动抽帧 → 字段映射 → 存 `processed.pkl` |
| 2 | `llib/data/preprocess/interx.py` | 生产版加载类 `InterX`，`load()` 参照 `hi4d.py` 结构 |
| 3 | `llib/defaults/datasets/datasets.py` | 新增 `@dataclass class InterX` |
| 4 | `llib/defaults/datasets/main.py` | `Datasets` 类里加 `interx: InterX = InterX()` |
| 5 | `llib/data/single.py::load_data()` | 新增 `elif self.dataset_name == 'interx':` 分支 |
| 6 | `llib/data/single.py::get_single_item()` | 两处 `elif 'InterX' in item['imgpath']:` 分支，字段映射照抄 FlickrCI3D 分支 |

不需要碰：`llib/data/single_optimization.py`、任何 loss 文件、`buddi.py` 模型结构、`train_module.py`。

---

## 五、训练命令

**无条件版**（已在跑，`interx_uncond`）：
```bash
python llib/methods/hhc_diffusion/main.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02.yaml \
  --exp-opts logging.base_folder=demo/diffusion/training logging.run=interx_uncond \
             datasets.train_names=['interx'] datasets.train_composition=[1.0] \
             datasets.val_names=['interx'] \
             datasets.interx.processed_data_folder=datasets/processed/InterX \
             datasets.augmentation.use=True \
             logging.logger='tensorboard'
```

**BEV 条件版**（命令已就绪，尚未启动，见下方"当前状态速览"）：
```bash
python llib/methods/hhc_diffusion/main.py \
  --exp-cfg llib/methods/hhc_diffusion/configs/config_buddi_v02_cond_bev.yaml \
  --exp-opts logging.base_folder=demo/diffusion/training logging.run=interx_cond_bev \
             datasets.train_names=['interx'] datasets.train_composition=[1.0] \
             datasets.val_names=['interx'] \
             datasets.interx.processed_data_folder=datasets/processed/InterX \
             datasets.interx.allow_missing_bev=False \
             datasets.interx.filter_penetration=True \
             datasets.augmentation.use=True \
             model.regressor.losses.pseudogt_v2v.weight=[1000.0] \
             logging.logger='tensorboard'
```
（`filter_penetration=True` 是阶段③新加的开关，见下方"阶段③执行更新"；启动前需要先跑穿模检测脚本生成排除清单，否则会报错提醒。）

---

## 六、数据量与训练时长（4090）

95,448 帧（阶段①标定后）规模上大概率超过 CHI3D 单独能提供的真值量级，单独训练足够。BEV 条件版训练 benchmark 过：**1.16 秒/step**（4090，与无条件版训练共享 GPU 时测的）。

---

## 七、验证方法

1. **训练侧**：`tensorboard --logdir demo/diffusion/training/interx_uncond`，看 loss 是否收敛，验证集 loss 是否明显差于训练集。
2. **效果检查**：跑 `sample.py`/`sample_and_rank.py` 做采样，肉眼检查生成姿态是否合理。
3. **自动分段质量检查**：小规模抽查过阈值标定是否合理（见"阶段②数据量标定"过程）。
4. 无条件版本身不能直接用于阶段③（需要 BEV 条件版），是数据流水线/真值质量的中间检查点。

---

## 追加阶段①：训练中发现的可视化/健壮性 bug 修复（已完成）

无条件版训练跑到第 9 个 epoch 时崩溃（`cv2.putText` 要求 uint8，渲染出来是 float），修复并顺带系统排查了同一条"训练时可视化/摘要/checkpoint"代码路径。**修了 4 处**：

1. `train_module.py::render_one_method` 里两处 `roll=180.0` → `roll=0.0`（预览图倒置，纯视觉 bug，不影响训练本身，已用真实关节坐标核实过三维数据一直是对的）。
2. `render_one_method` 里 `vertex_transl_center` 复用样本0的中心导致其余样本错位居中 → 改用循环内局部变量。
3. `loss_module.py` 里 NaN loss 时 `import ipdb; ipdb.set_trace()`（无人值守训练会静默挂起）→ 改成打印 per-loss 明细后 `raise RuntimeError`。
4. `train_module.py::get_guidance_params` 的 `"contact"` 分支用错字段名 `batch["contact"]` → 改成 `batch["contact_map"]`（当前是死代码，为将来 BEV+contact 训练预先修好）。

保持现状不修的项（已跟用户确认）：`Trainer.validate()` 只评估验证集固定第一个 batch；checkpoint 不自动清理；若干确认过的纯死代码分支。

---

## 追加阶段②：生成 BEV 条件版训练数据（已完成）

**可行性验证**（真实代码跑出来的，非估算）：`Pytorch3dRenderer` 渲染单帧双人网格 ~124ms；`bev` 模型常驻进程内复用（而非像 `demo.sh` 那样每次起 CLI 子进程重新加载模型，那样 95,448 帧要跑 ~80 小时）后单帧推理 ~74ms；5/5 测试帧 BEV 都成功识别出两人。

**相机随机化方案**（模拟真实照片拍摄角度分布，用户要求"真实基础上五花八门一些"）：

| 参数 | 分布 | 理由 |
|---|---|---|
| 距离 tz | `base_dist × Uniform(0.85, 1.25)` | 取景松紧不同 |
| 方位角 yaw | `Uniform(-150°, 150°)` | 排除正后方±30°（现实中少见），其余方向等概率 |
| 俯仰角 pitch | `Normal(5°, 12°)`，截断 `[-20°,35°]` | 均值轻微向下模拟持机人平视/略俯拍 |
| 滚转角 roll | `Normal(0°, 3°)`，截断 `[-8°,8°]` | 手持拍摄的自然小幅倾斜 |

**新增代码**：`llib/data/preprocess/utils/process_interx_bev.py`（`InterXBevProcessor` 类，含 `render_sample`/`match_detections`——BEV 检测顺序跟人物 0/1 无必然对应，靠屏幕空间 x 坐标最近邻配对/`bev_to_smplx`/`process_sample`）。不改动 `single.py`/`interx.py`/`defaults/`——`bev_smplx_*` 读取路径阶段①已接好。

**✅ 全量跑批结果（真实数据，非估计）**：4735 take、95,448 帧，耗时约 4 小时20分钟。**24,339 条（25.5%）检测失败**（`information_missing=True`，全部是"两人被误检成一个人"这一种失败模式，跟相机角度无关，是姿态贴太近导致 2D 投影严重遮挡，符合"大面积接触"数据本身的预期副作用）；**71,109 条成功获得真实 BEV 估计**。

---

## 追加阶段③：穿模（身体网格自穿插）检测 —— 训练前的数据清洗

### Context

阶段①产出的 95,448 帧是专门筛"大面积接触"的互动片段——用户明确要求：训练前必须过滤掉**身体网格真的穿插在一起**的帧，且强调不能只看根关节距离这种粗糙代理，而要检测**两人身体表面任意两点之间**是否发生了几何穿插。

### 技术方案（复用仓库已有代码，未新写穿模算法）

- `llib/utils/threed/intersection.py::winding_numbers`（广义绕数法，判断查询点是否被闭合三角网格"包住"）+ `llib/utils/metrics/contact.py::MaxIntersection`（目标网格降采样加速、查询顶点保持全分辨率10,475点的标准做法）就是仓库里现成的"两人体网格互穿"评测代码。
- **发现一个隐患，绕开而非修复**：`MaxIntersection.forward_batch`（`contact.py:190`）里有 `except: import ipdb; ipdb.set_trace()`——跟之前 `loss_module.py` 修过的同一类地雷，无人值守脚本一旦命中会静默挂起而非报错退出。新脚本 `check_interx_penetration.py` **不直接调用 `crit.forward()`**，而是复用 `MaxIntersection` 里安全的辅助方法（`prep_mesh`/`close_mouth`/`to_lowres`），自己重新实现一份逻辑等价、不含裸 `except` 的批量评分函数（已用真实数据 A/B 验证输出一致）。未修改 `llib/utils/metrics/contact.py` 本体。
- **实测标定分辨率/批大小**（4090，真实数据）：`lowres=500` 和 `1000` 结果几乎一致（穿透深度差<0.001m），`100`/`300` 会漏检浅穿模（实测有一帧 0.0035m 穿透在低分辨率下判定成 0）；选定 `lowres=500`。`batch_size=16` 帧时显存峰值 15.4GB（24GB卡安全），`32` 直接 OOM。

### 新增代码

`llib/data/preprocess/utils/check_interx_penetration.py`：扫描 `processed.pkl` 全部帧，对每帧算 `min_dist`/`max_v1_in_v2`/`mean_v1_in_v2`/`max_v2_in_v1`/`mean_v2_in_v1`/`max_penetration`，存 `datasets/processed/InterX/diagnostics/penetration_report.pkl`，打印分位数统计并出直方图 `penetration_depth_hist.png`；`--exclude-threshold <米>` 可选参数生成排除清单 `penetration_exclude_list.pkl`（内容是一个 `imgname` 字符串集合——用 `imgname` 而不是 `take_id+下标`，因为 `imgname` 在整个数据集里唯一，不受缓存展平/重载影响）。支持缓存复用：只要 `penetration_report.pkl` 已存在，换阈值重新生成排除清单不需要重跑 GPU 扫描（除非传 `--force-rescan`）。

### ✅ 阶段③执行更新（已完成的部分）

1. **50-take 小规模验证**（803 帧）：脚本无报错跑通，检测到穿模帧占比 **64.9%**，穿透深度分位数 p50=2.07cm / p90=5.46cm / p99=8.61cm / max=9.82cm。占比看起来高，但这符合预期——数据本来就是按"大面积接触"专门筛出来的，SMPL-X 是刚性网格（不模拟衣物/皮肤压缩），真实接触天然容易伴随几厘米级别的浅穿插。
2. **肉眼抽查验证检测有效**：渲染了穿透最深的2帧（9.6cm/9.82cm）、中位数帧（2cm）、零穿模帧，正面+侧面视角对比：
   - 9.6~9.82cm 的两帧从侧视图能清楚看到一条腿/手臂直接穿进对方躯干内部，明显异常。
   - 2cm 的中位数帧只是牵手/贴近时手部有轻微重叠，是正常接触。
   - 0cm 的帧是干净的握手姿态，两人网格完全没有重叠。
   检测器的数值判断跟肉眼观察完全一致，方法验证通过。
3. **排除清单接入训练数据加载逻辑**（已写代码，尚未启用）：
   - `check_interx_penetration.py::save_exclude_list` 改成存 `imgname` 集合（而非最初设计的 `take_id→下标`，理由见上）。
   - `llib/data/preprocess/interx.py::InterX.__init__`/`load()` 新增 `filter_penetration` 开关：为 `True` 时读取 `diagnostics/penetration_exclude_list.pkl` 并按 `imgname` 过滤；排除清单不存在时**报错**（`FileNotFoundError`，不会静默用未过滤数据训练）。这个过滤在**缓存 `{split}_diffusion.pkl` 加载之后**执行，跟 `allow_missing_information` 过滤一样，所以**切换 `filter_penetration` 不需要删缓存**。
   - `llib/defaults/datasets/datasets.py::InterX` dataclass 新增字段 `filter_penetration: bool = False`（默认关闭，不影响现有行为），通过 `single.py` 里已有的 `**self.dataset_cfg` 自动透传，无需改 `single.py`。
4. **全量 95,448 帧穿模扫描**：🟡 **进行中**（后台进程，日志 `/tmp/interx_penetration_full_run.log`）。实测吞吐比早期纯 GPU 微基准慢（真实脚本含数据搬运开销），约 1.31 it/s（16帧/批），**预计总耗时约 1 小时15分钟**（早前计划里写的"30分钟"是只算了纯 GPU 计算的微基准，偏乐观，如实更正）。

### 待办（等全量扫描跑完）

1. 看全量的穿透深度分位数分布 + 不同阈值的排除比例，用户选定 `--exclude-threshold`。
2. 用选定阈值重跑一次 `check_interx_penetration.py --exclude-threshold <米>`（会复用已缓存的 report，几秒钟内出排除清单，不用重新扫描)。
3. 训练命令里加 `datasets.interx.filter_penetration=True`，启动 BEV 条件版训练。
