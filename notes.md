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

## 2026-02-21T08:50:00+00:00 笔记: MultipleView 数据格式与从视频生成的关键点

### MultipleView 的数据格式(本仓库约定)

来源: `README.md` + `scene/multipleview_dataset.py`.

- 输入帧结构:
  - `data/multipleview/<dataset>/cam01/frame_00001.jpg`
  - `data/multipleview/<dataset>/cam02/frame_00001.jpg`
  - ...
- 训练读取逻辑:
  - `scene/multipleview_dataset.py` 会用 `cam01` 的帧数作为全局长度,并假设所有相机帧数一致.
  - 每个相机按 `frame_%05d.jpg` 读取,时间归一化为 `i / image_length`.
- 派生文件:
  - `sparse_/cameras.bin` + `sparse_/images.bin`(由 COLMAP 估计相机内外参)
  - `points3D_multipleview.ply`(初始化点云)
  - `poses_bounds_multipleview.npy`(用于生成 spiral 渲染相机轨迹)

### 现有脚本链路的问题

- `multipleviewprogress.sh` 假设你已经把每个相机的视频抽成帧并放到 `camXX/frame_*.jpg`.
- `multipleviewprogress.sh` 还会 `git clone https://github.com/Fyusion/LLFF.git` 并 `pip install scikit-image`,
  这在离线/受限网络环境下不稳定.
- `scripts/extractimages.py` 只会取每个相机的第一帧(`frame_00001`)去跑 COLMAP,
  用于估计多机位的静态相机参数.

### 目标视频目录的初步观察

- 路径: `/cloud/cloud-s3fs/SelfCap/bar-release/videos`
- 文件: `02.mp4 ... 19.mp4`(共 18 路)
- 用 `ffprobe` 看 `02.mp4`:
  - codec: hevc
  - 分辨率: 2110x3760(竖屏)
  - 帧率: 60fps
  - 帧数: 3540(约 59 秒)

### 生成 `poses_bounds_multipleview.npy` 的关键算法(参考 LLFF 逻辑)

LLFF 的核心做法是:

1. 从 COLMAP `cameras.bin/images.bin` 构造 `c2w` pose,并拼上 `[H,W,focal]` 成为 3x5.
2. 做一次坐标轴重排,从 `[r, -u, t]` 转换到 `[-u, r, -t]`.
3. 用 `points3D.bin` 的可见性(track)统计每个相机可见点的深度分布,
   取 0.1% 和 99.9% 分位作为 near/far bounds.

因此我们不需要 `git clone LLFF`,只要在仓库里复刻上述小段逻辑即可.

### COLMAP 在 headless 环境的两个坑

这次在容器/无 display 环境里实际跑 COLMAP 时遇到两类崩溃:

1. `qt.qpa.xcb: could not connect to display`
   - 解决: 运行 COLMAP 命令前设置环境变量 `QT_QPA_PLATFORM=offscreen`.
2. `opengl_utils.cc: Check failed: context_.create()`
   - 这通常是 exhaustive_matcher 默认走 SiftGPU/OpenGL,而 headless 环境无法创建 OpenGL context.
   - 解决: 对 feature_extractor/matcher 加:
     - `--SiftExtraction.use_gpu 0`
     - `--SiftMatching.use_gpu 0`

因此脚本里需要显式设置上述选项,否则会出现 SIGABRT,看起来像 "colmap 自己崩了".

## 2026-02-21T10:00:00+00:00 笔记: 对标 FreeTimeGsVanilla 的“体量”参数建议

### FreeTimeGsVanilla 的默认体量是什么?

FreeTimeGsVanilla 的 mp4 pipeline 示例通常是:

- frame range: `[0, 61)`(end exclusive)
- 含义: 每路相机取 61 帧连续帧(如果原视频是 60fps,约等于 1 秒)

此外它会用 `DATA_FACTOR=4/8` 在训练侧对图片下采样.
这意味着:

- 它可能保存原图抽帧,但训练实际看到的是更小的分辨率.

### 我们的 MultipleView 生成脚本如何对齐?

本仓库的 `scripts/preprocess_multipleview_from_videos.py` 主要可控项是:

- `--max-frames`: 控制每路相机输出帧数
- `--fps`: 控制抽帧频率(单位: 帧/秒)
- `--max-size`: 控制输出图片最长边,等价于"提前做 data_factor 下采样"

因此,想对齐 "61 帧" 的体量,可直接:

- `--fps 60 --max-frames 61`

想对齐 `DATA_FACTOR` 的有效分辨率,可以用下面的近似关系:

- `max_size ≈ max(original_H, original_W) / DATA_FACTOR`

以 bar-release 竖屏视频(最长边约 3760)为例:

- `DATA_FACTOR=4` 约等价 `--max-size 960`
- `DATA_FACTOR=8` 约等价 `--max-size 480`
