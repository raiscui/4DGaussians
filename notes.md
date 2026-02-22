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

## 2026-02-21T13:37:57+00:00 追加笔记: COLMAP feature_extractor returncode=-9(SIGKILL) 排查要点

### 现象

`scripts/preprocess_multipleview_from_videos.py` 调用:

- `colmap feature_extractor ...`

失败,Python 侧看到:

- `returncode=-9`

### 初步判断(本质)

- `subprocess` 的负返回码表示"被信号终止".
- `-9` 对应 `SIGKILL`,常见原因:
  - 系统/容器内存不足触发 OOM killer.
  - 外部手动 `kill -9` 或调度器强制终止.

### 关键发现: 脚本里用了比 COLMAP 默认更激进的 SIFT 参数

用 `colmap feature_extractor --help` 可看到默认值(摘关键项):

- `--SiftExtraction.max_image_size (=3200)`
- `--SiftExtraction.max_num_features (=8192)`
- `--SiftExtraction.estimate_affine_shape (=0)`
- `--SiftExtraction.domain_size_pooling (=0)`
- `--SiftExtraction.num_threads (=-1)`(默认吃满 CPU)

而脚本当前硬编码为:

- `max_image_size=4096`
- `max_num_features=16384`
- affine/dsp 都开启

这会显著放大 CPU/内存峰值,在接近 4K 的帧上更容易触发 OOM.

### 结论

优先修复方向:

1. 把 SIFT 参数改回更接近默认值.
2. 暴露为 CLI 参数,便于用户按机器资源调参.
3. 当检测到 SIGKILL 时,在异常信息里明确提示"可能是 OOM",并给出降参建议.

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

### 更正: bar-release 的 1/4 长边是 940,不是 960

- bar-release 原视频尺寸是 `2110x3760`(可用 `ffprobe`/PIL 查看).
- FreeTimeGsVanilla 的 `results/bar_release_full/out_0_61/cfg.yml` 里 `data_factor: 4`,
  等效分辨率更接近:
  - `2110/4 ≈ 528`
  - `3760/4 = 940`
  它的输出 mp4(`traj_4d_step29999.mp4`)显示为 `528x944`,主要是对齐到 16 的倍数.
- 因此本仓库 README 里 bar-release 的示例命令,已把 `--max-size` 从 960 更正为 940.
- `--max-size 960` 更适合标准 4K(3840 长边)按 1/4 下采样的场景.

## 2026-02-21T12:24:13+00:00 笔记: 对齐 FreeTimeGsVanilla 的 data_factor 语义(训练侧下采样)

### 关键事实(来自 FreeTimeGsVanilla 代码与产物)

- FreeTime 的 `data_factor` 是训练/加载侧下采样:
  - `datasets/FreeTime_dataset.py` 会把 `K[:2, :] /= factor`,并在 `__getitem__` 里用:
    - `cv2.resize(..., dsize=(W // factor, H // factor))`
  - 也就是说它用的是 floor 下采样,并同步缩放 intrinsics,确保 FOV 语义不变.
- 数据落盘仍保留原始帧分辨率:
  - 例如 `results/bar_release_full/out_0_61/cfg.yml` 里 `data_factor: 4`,
    但 `work_0_61/data/images/*/*.jpg` 仍是 `2110x3760`.

### 对本仓库 MultipleView 的含义

- 如果我们在生成阶段用 `--max-size` 把帧缩小,会发生不可逆降质,不利于公平评估对比.
- 更合理的对齐方式是:
  1) 生成阶段 `--max-size 0` 保留原始帧质量.
  2) 训练阶段使用 `--resolution 4/8` 做等价 `data_factor` 的下采样(只影响加载进训练的尺寸与 focal).

## 2026-02-21T17:04:40+00:00 笔记: 提交/推送前的本地文件卫生检查

### 结论

- `data.zip` 属于本地大文件产物,不应提交,应加入 `.gitignore`.
- `.envrc.private` 已被 `.gitignore` 忽略,不会被提交.
- `.envrc` 本身不包含 GitHub token 的字面量,仅通过环境变量引用,可安全纳入提交,用于 direnv/代理/非交互式 git 等开发辅助配置.

## 2026-02-21T17:52:38+00:00 补充笔记: 让 `.envrc` 可以安全提交

- `.envrc` 现在明确为可公开提交的 direnv 配置,不再包含"把 PAT 粘贴到这里"这类误导文案.
- 私密信息只允许写到 `.envrc.private`:
  - 例如 `export GITHUB_TOKEN=github_pat_xxx`
  - 该文件已被 `.gitignore` 忽略.
- 当检测到 `GITHUB_USERNAME/GITHUB_TOKEN` 都存在时,`.envrc` 会自动生成:
  - `.direnv/git-askpass.sh`
  用于 `git push` 的 https 非交互认证.
  `.direnv/` 目录已被 `.gitignore` 忽略,不会进入仓库历史.
- 代理也改为显式开关:
  - 只有设置 `USE_LOCAL_PROXY=1` 时才启用本地 7897,避免默认误伤无代理环境.

## 2026-02-22T07:39:20+00:00 笔记: MultipleView 渲染"视角乱跳"的根因

### 复现与证据

- 用户运行:
  - `pixi run python render.py --model_path output --skip_train --configs arguments/multipleview/default.py`
- 输出目录里同时存在两条 mp4:
  - `output/test/ours_30000/video_rgb.mp4`
  - `output/video/ours_30000/video_rgb.mp4`
- 统计帧数(通过 renders 图片数量验证):
  - `output/test/ours_30000/renders`: 54 张
    - 54 = 18 cams * 3 frames,符合当前 MultipleView test split 的实现.
  - `output/video/ours_30000/renders`: 300 张
    - 来自 `get_spiral(..., N_views=300)` 的渲染相机轨迹.

### 结论(根因)

- "视角乱跳"主要来自 `test` 视频把多相机视角的帧串接成一条 mp4.
- MultipleView 当前 test split 是"每个相机抽 3 帧",并不是"按相机 hold-out".
  - 因此 test mp4 既会跳视角,也很难用来观察完整时间序列的动态效果.

### 额外确认: spiral 相机轨迹本身是连续的

- 对 `poses_bounds_multipleview.npy` 生成的 spiral camera path 做数值检查:
  - 相机位置步长的 max/mean 约 1.53,没有离群的 spike.
  - 相机旋转步长也平滑,没有翻转式跳变.
- 因此 `output/video/...` 的"乱跳"概率较低,更可能是用户打开了 `output/test/...` 的 mp4.

## 2026-02-22T07:56:51+00:00 笔记: imageio 写 mp4 的 macro_block_size 自动 resize 警告

### 现象

- 渲染 MultipleView 后,终端出现多条 warning:
  - `IMAGEIO FFMPEG_WRITER WARNING: input image is not divisible by macro_block_size=16, resizing from (527, 940) to (528, 944) ...`

### 根因

- MultipleView 当前在 `--resolution 4` 下采用 floor 下采样:
  - `H = 2110 // 4 = 527`
  - `W = 3760 // 4 = 940`
- (527,940) 不是 16 的倍数.
- imageio 的 ffmpeg writer 为了编码兼容性会自动把帧 resize 到 (528,944).
  - 这不是训练/渲染数值错误.
  - 但它属于"静默改变输出像素",也会让日志看起来像出错.

### 解决思路

- 更稳的做法是写 mp4 前对帧做 padding 到 16 倍数:
  - 不缩放,只补边.
  - 兼容性也更好.

---

# 笔记: 为什么本项目 iteration_30000 很小,而 FreeTime ckpt 很大

时间: 2026-02-22T08:44:30+00:00

## 本项目(4DGaussians)实际保存了什么

目录: `output/point_cloud/iteration_30000/`

- `point_cloud.ply`
  - PLY 头部: `element vertex 119497`
  - property 数: 62 个 float32,即每个高斯 62*4=248 bytes.
  - 预期数据体积: 119497*248=28.26MB,与文件实际体积(约 28.26MB)一致.
- `deformation.pth`
  - `torch.save(self._deformation.state_dict(), ...)`,主要是 deformation 网络权重.
  - 体积约 9.5MB.
- `deformation_table.pth`
  - bool tensor,shape=(119497,),约 0.11MB.
- `deformation_accum.pth`
  - float32 tensor,shape=(119497,3),约 1.37MB.

结论: 本项目的 `iteration_30000` 更像是"可渲染/可加载的模型快照",不包含 optimizer state.

代码位置:
- `scene/__init__.py`: `Scene.save()` 只调用 `save_ply()` + `save_deformation()`.
- `scene/gaussian_model.py`: `save_ply()` 写 PLY,`save_deformation()` 只保存 deformation 相关张量.
- `scene/gaussian_model.py`: `capture()` 里才会包含 `self.optimizer.state_dict()`.
- `train.py`: 只有传 `--checkpoint_iterations` 才会 `torch.save((gaussians.capture(), iteration), ...)` 生成 `chkpnt_*.pth`.

## FreeTimeGsVanilla checkpoint 实际保存了什么

文件: `/cloud/cloud-ssd1/FreeTimeGsVanilla/results/bar_release_full/out_0_61/ckpts/ckpt_29999.pt`

- 文件大小: 978MB.
- 顶层 keys: `step`, `splats`, `optimizers`.
- `splats`(模型参数)共 9 个 tensor,高斯数量 N=1,335,131.
  - splats 总张量体积约 325.96MB.
  - 其中 `shN` 约 229.19MB(最大头).
- `optimizers`(Adam 状态)也按 9 组参数存.
  - optimizers 总张量体积约 651.92MB.
  - 其中 `optimizers.shN` 约 458.38MB,来自 `exp_avg` + `exp_avg_sq` 两份与 `shN` 同 shape 的张量.

结论: FreeTime 的 `.pt` 是"可继续训练的完整 checkpoint".
它既保存了模型参数,也保存了 Adam 的一阶/二阶动量,因此体积大约是(模型参数) + 2*(模型参数).
此外,它的高斯数量 133 万,也显著多于本项目示例的 11.9 万,模型参数本身就会更大.

## 2026-02-22T08:48:56+00:00 追加: MultipleView video 镜头在中心视角停留更久

### 用户诉求

- 用户渲染命令:
  - `pixi run python render.py --model_path "output" --skip_train --configs arguments/multipleview/default.py`
- 关注的输出:
  - `output/video/ours_30000/video_rgb.mp4`
- 希望:
  - spiral novel-view 轨迹不要一开始就快速绕圈.
  - 镜头在"中心视角"附近能停留更久(相机更稳,更便于观察动作).

### 轨迹生成链路(定位结果)

- `render.py` 的 `name=="video"` 会渲染 `scene.getVideoCameras()`.
- MultipleView 的 `video_cameras` 来自:
  - `scene/dataset_readers.py:readMultipleViewinfos()` -> `test_cam_infos.video_cam_infos`
  - `scene/multipleview_dataset.py:multipleview_dataset.__init__()` 在 `split=="test"` 时构建 `video_cam_infos`
  - `scene/multipleview_dataset.py:get_video_cam_infos()` 读取 `poses_bounds_multipleview.npy`,调用 `scene/neural_3D_dataset_NDC.py:get_spiral()` 生成 spiral pose.

### 落地方案(已实现)

- 在 MultipleView 数据集侧对 spiral 轨迹做可配置改良:
  - `video_spiral_hold_start`: 在起始 pose 处重复若干帧(相机不动),但 time 仍随帧推进.
  - `video_spiral_n_rots`: 控制绕圈次数(默认 2,可改成 1 让运动更慢更稳).
  - `video_spiral_rads_scale`: 控制轨迹半径缩放(默认 1.0,可调小让镜头更靠近中心).
  - `video_n_views`: spiral 轨迹的基础采样帧数(默认 300).
- 默认配置已写入:
  - `arguments/multipleview/default.py`: `ModelParams.video_spiral_hold_start=60`,`video_spiral_n_rots=1`.

### 兼容性处理(重要)

- 旧模型的 `output/cfg_args` 里不包含新字段.
- 为避免渲染阶段缺字段导致参数无法生效:
  - `utils/params_utils.py:merge_hparams()` 改为允许在 merge 配置文件时“补齐字段”.
  - `scene/__init__.py` 在调用 MultipleView reader 时用 `getattr(args, ..., default)` 做兜底.

## 2026-02-22T09:43:07+00:00 追加: MultipleView video 的 time 应按真实帧数 loop

### 用户反馈

- 用户指出: “如果你想生成 300 帧动画,但实际有效帧只有 x<300,那么你可以 loop”.

### 根因总结

- MultipleView 的真实时间序列长度是 x(例如每路相机 61 帧).
- spiral video 的相机轨迹采样帧数是 N(默认 300).
- 若 time 用线性 `i/N`,动作会被拉慢约 N/x 倍,肉眼就会觉得“该动的没动”.

### 修复方案(已落地)

- 增加 `video_time_mode`:
  - `linear`: 旧行为 `time=i/N`.
  - `loop`: 新行为 `time=(i % x)/x`,其中 x 来自 `data/.../cam01` 的帧数统计.
- MultipleView 默认配置改为启用 loop:
  - `arguments/multipleview/default.py`: `video_time_mode=\"loop\"`.
  - 同时把 `video_spiral_hold_start` 默认回退为 0(不额外停留).

### 验证证据(本地)

- 对 `bar-release_fullres_0_61`:
  - `video_len` 为 300.
  - `time[60]≈60/61`, `time[61]=0.0`,证明 time 每 61 帧循环一次.
