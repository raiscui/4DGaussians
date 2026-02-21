# 笔记: notebook 依赖迁移到 Pixi

时间: 2026-02-21T06:26:20+00:00

## 现状扫描

### 1) notebook 里现有安装方式

- `4DGaussians.ipynb`
  - 第 0 个 code cell: `pip install -r requirements.txt` + `pip install -e submodules/...`
  - 后续 cell: `pip3 install torch torchvision torchaudio`
  - 训练/渲染通过 `!python train.py ...` / `!python render.py ...`

- `4DGaussians_rais.ipynb`
  - 第 0 个 code cell: `%pip install -r requirements.txt`(且带了一行 `!export ...proxy...`,但在 Jupyter 的 `!` 语义下不持久)
  - 后续 cell: `apt-get build-essential`,`pip3 install torch ...`
  - 有一个 cell 会给 `submodules/simple-knn/simple_knn.cu` 头部补 `#include <cfloat>`,然后 `pip install -e ...` 重新安装两个 CUDA 扩展
  - 训练/渲染同样用 `!python ...`

### 2) requirements.txt 内容(明显不完整)

当前 `requirements.txt` 只有:

- `mmcv==1.6.0`
- `matplotlib`
- `argparse`
- `lpips`
- `plyfile`
- `pytorch_msssim`
- `open3d`
- `imageio[ffmpeg]`

但代码实际还会用到 `torch`,`torchvision`,`numpy`,`scipy`,`tqdm`,`opencv-python(cv2)`,`easydict` 等.
README 里也提示要参考 3D-GS 项目的依赖安装,说明 `requirements.txt` 并不等于完整环境.

### 3) 代码侧关键依赖(从 import 粗扫得到)

核心训练/渲染链路相关:

- `torch`,`torchvision`
- `numpy`,`scipy`
- `open3d`
- `mmcv`(用于 `Config.fromfile`,train/render 都会用)
- `cv2`(opencv)
- `lpips`,`pytorch_msssim`
- `plyfile`
- `imageio`
- `tqdm`
- 两个 CUDA 子模块 python 包:
  - `diff_gaussian_rasterization`(来自 `submodules/depth-diff-gaussian-rasterization`)
  - `simple_knn`(来自 `submodules/simple-knn`)

非核心但在仓库里出现过的:

- `sklearn`(主要在 `utils/point_utils.py` 中)
- `torch_cluster`(同上,且安装会更麻烦,后续可作为可选项处理)

## 迁移策略(想法)

- pixi 的正确姿势是: 依赖写在 `pixi.toml`,用 `pixi install` 安装,再用 `pixi run ...` 执行脚本.
- notebook 本身运行在默认 kernel 里也没关系,只要所有耗依赖的脚本执行都走 `pixi run python ...`,就不会受 kernel 的包污染.

## 2026-02-21T06:26:20+00:00 后续更新

### 1) 为了降低依赖复杂度,移除 mmcv

- 原先 `train.py`/`render.py`/`export_perframe_3DGS.py`/`merge_many_4dgs.py` 使用 `mmcv.Config.fromfile()` 解析 `arguments/**/*.py` 配置.
- `mmcv` 在 notebook/新环境里经常触发编译,稳定性很差.
- 已在 `utils/params_utils.py` 增加轻量 `load_config_file()`:
  - 支持 `_base_` 继承(递归加载)
  - 支持 dict 深度合并(子配置覆盖父配置)
- 这些脚本已改为使用 `load_config_file()` + `merge_hparams()`.

### 2) pixi.toml 的依赖来源

- Conda: python, pytorch/torchvision + pytorch-cuda, numpy/scipy, open3d, opencv, ffmpeg, 等.
- PyPI: lpips, plyfile, pytorch-msssim, easydict, 以及两个本地 editable CUDA 扩展(子模块路径依赖).

### 3) Pixi install 可能的 DNS 问题

- 现象: `pixi install` 在下载 repodata 时失败,报 `conda.anaconda.org` 的 DNS 解析错误(Name does not resolve).
- 处理: 用 `pixi install --config pixi.mirrors.toml` 通过镜像站重定向 conda channels.
- 已落地:
  - 仓库新增 `pixi.mirrors.toml`
  - notebook 中 `pixi install` 增加了自动回退逻辑

## 2026-02-21T08:13:26+00:00 追加更新

### 1) 更正: Pixi CLI 不支持 `pixi install --config ...`

- `pixi install --help` 中没有 `--config` 参数.
- 正确做法是把镜像配置写入项目本地配置文件: `.pixi/config.toml`.
- notebook 已调整为自动回退:
  - 先尝试 `pixi install`
  - 失败则写入 `.pixi/config.toml` 后再重试 `pixi install`

### 2) Python 3.12 + CUDA 12.6 + PyTorch 2.6 的实现方式

- conda 的 `pytorch` channel 目前最高是:
  - `pytorch 2.5.x`
  - `pytorch-cuda 12.4`
  因此 conda 无法满足 `pytorch 2.6` 或 `pytorch-cuda 12.6`.
- 已改用 PyPI 的 cu126 官方 wheel:
  - `torch ~=2.6.0`
  - `torchvision ~=0.21.0`
  - index: `https://download.pytorch.org/whl/cu126`

### 3) 修复: conda+PyPI 混合依赖时,GitHub mapping 拉取失败

- Pixi 在 conda+PyPI 混合依赖下,默认会拉取 conda<->pypi 名称映射.
- 某些网络环境下访问 `raw.githubusercontent.com` 会 DNS 失败,导致 `pixi lock/install` 失败.
- 解决:
  - 在 `pixi.toml` 增加 `conda-pypi-map` 指向本地 `conda_pypi_map.json`
  - 避免运行期访问 GitHub

### 4) 两个 CUDA 子模块的安装方式调整

- 由于 uv 在 lock 阶段会执行子模块的 `setup.py` 生成 metadata,而它们顶层 `import torch`,
  会在 torch 未安装时失败.
- 处理方式:
  - 不再把两个子模块作为 `pixi.toml` 的 pypi path 依赖参与求解.
  - 改为提供 Pixi task: `pixi run install-ext`
  - notebook 会在 `pixi install` 之后自动执行 `pixi run install-ext` 编译并安装扩展.

## 2026-02-21T08:31:00+00:00 追加更新: editable 安装的 build isolation 与 `simple_knn` 导入问题

### 1) `pixi run install-ext` 报 `ModuleNotFoundError: torch`

- 现象: `pip install -e ...` 在 "Getting requirements to build editable" 阶段报:
  - `ModuleNotFoundError: No module named 'torch'`
- 本质: pip 默认启用 build isolation,会创建临时构建环境(`/tmp/pip-build-env-*/overlay/...`),
  而两个子模块的 `setup.py` 顶层会 `import torch`,导致隔离环境里缺 torch 时直接失败.
- 修复: `pixi.toml` 的 `install-ext` task 增加 `--no-build-isolation`,让构建过程复用 Pixi 环境里的 torch.

### 2) `simple_knn` 已安装但无法导入

- 现象: pip 显示 `Successfully installed simple_knn-0.0.0`,但运行期:
  - `import simple_knn` 或 `from simple_knn._C import distCUDA2` 仍会失败.
- 本质: 子模块目录 `submodules/simple-knn/simple_knn/` 缺少 `__init__.py`.
  在 PEP660 editable 的 import hook 机制下,顶层包映射要求目标路径存在:
  - `simple_knn/__init__.py` 或
  - `simple_knn.*`(例如 `simple_knn.py`,`simple_knn.so`),
  否则导入系统无法生成 module spec.
- 修复: 新增 `submodules/simple-knn/simple_knn/__init__.py`,
  让 `simple_knn` 成为可导入的包,并从 `._C` 暴露 `distCUDA2`.

### 3) 快速自检命令

- `pixi run python -c "import torch; print(torch.__version__, torch.version.cuda)"`
- `pixi run python -c "import diff_gaussian_rasterization; from simple_knn._C import distCUDA2; print('ok')"`
