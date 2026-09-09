# 运行环境

这份文档记录本 fork 实际跑通的环境，供复现参考。上游的通用安装说明见
[INSTALL.md](./INSTALL.md)；这里只记与之不同、或实际踩过坑的地方。

## 实测配置

| 项目 | 版本 |
|---|---|
| OS | Ubuntu 26.04 LTS (kernel 7.0.0-28-generic) |
| GPU | NVIDIA GeForce RTX 4090, 24 GB |
| NVIDIA 驱动 | 580.173.02 |
| 系统 CUDA toolkit | 12.4 (`nvcc` V12.4.131) |
| conda | miniconda3 |
| Python | 3.9.25 |

主要 Python 包：

| 包 | 版本 |
|---|---|
| torch | 2.0.1 (cu118) |
| torchvision | 0.15.2 |
| pytorch3d | 0.7.8 |
| smplx | 0.1.28 |
| simple-romp | 1.1.3 |
| numpy | 1.23.5 |
| scipy | 1.13.1 |
| trimesh | 4.12.2 |
| pyrender | 0.1.45 |
| chumpy | 0.70 |

完整清单见 [`environment/environment_buddi.yml`](../environment/environment_buddi.yml)
（conda，含 channel 与 pip 段）和
[`environment/requirements_buddi_freeze.txt`](../environment/requirements_buddi_freeze.txt)
（纯 pip 版本对照）。

## 重建环境

```bash
conda env create -n buddi -f environment/environment_buddi.yml
conda activate buddi
```

导出时已去掉 `prefix:` 行，所以可以直接在别的机器上用。`environment_buddi.yml`
用 `--no-builds` 导出，不锁死 build string，跨机器容错更好；如果解不出依赖，
再退回用 requirements 文件逐个对版本。

## 几个要注意的点

**torch 的 CUDA 版本和系统的不一致，这是正常的。** torch 2.0.1 是 cu118 构建，
而系统 `nvcc` 是 12.4。NVIDIA 驱动向后兼容，torch 自带运行时，所以
`torch.cuda.is_available()` 为 `True`，不需要为了对齐版本去重装 CUDA toolkit。
只有在源码编译扩展（比如自行编译 pytorch3d）时才需要关心系统 toolkit 版本。

**numpy 必须 < 1.24。** `chumpy`（SMPL 系列的依赖）在 `__init__.py` 第 11 行有
`from numpy import bool, int, float, complex, object, unicode, str, nan, inf`，
这些别名在 numpy 1.20 被弃用、**1.24 正式移除**，所以装了 1.24 或更高就会在
`import chumpy` 这一步直接 `ImportError`，而不是等到运行时才报。注意上限是 1.24
而不是 2.0。这里锁的是 1.23.5。

**无头渲染必须设 `PYOPENGL_PLATFORM=egl`。** 所有走 pyrender 的可视化脚本
（`sample.py --save-vis`、`inspect_interx_frame.py`、优化流程的渲染输出）在没有
显示器的服务器上不设这个变量会直接报错。

**`PYTHONPATH` 要指向仓库根目录。** 脚本都以 `llib.xxx` 的形式导入，不设的话
会 `ModuleNotFoundError`。

每次开工的标准三行：

```bash
conda activate buddi
cd /path/to/buddi
export PYTHONPATH=$(pwd)
export PYOPENGL_PLATFORM=egl
```

## 环境之外还需要准备的

仓库不含数据和权重，需要另行获取（见 [DATA.md](./DATA.md)）：

- `essentials/` — SMPL-X / SMPL / SMIL 身体模型、ViTPose 权重，约 5.3 GB
- `~/.romp/` — BEV 与 ROMP 的 checkpoint，约 270 MB，`simple-romp` 首次运行会自动下载
- `datasets/original/`、`datasets/processed/` — 各数据集需自行申请
