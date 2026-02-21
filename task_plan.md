# 任务计划: 将 notebook 依赖安装迁移到 Pixi

时间: 2026-02-21T06:26:20+00:00

## 目标

把项目里 notebook(主要是 `4DGaussians.ipynb` 和 `4DGaussians_rais.ipynb`)的依赖安装方式,从 `pip install -r requirements.txt` 等命令,改成基于 `pixi.toml` 的 `pixi install`.

同时把 notebook 里所有直接运行的脚本命令:

- `!python train.py ...`
- `!python render.py ...`

统一改成:

- `!pixi run python train.py ...`
- `!pixi run python render.py ...`

这样 notebook 即使运行在 Colab/Jupyter 的默认 kernel 里,也能保证训练/渲染脚本是在 Pixi 环境中执行,依赖来源一致且可复现.

## 阶段

- [x] 阶段1: 现状调研与方案确定
- [x] 阶段2: 设计 Pixi 依赖清单
- [x] 阶段3: 落地 `pixi.toml` 与 notebook 改造
- [x] 阶段4: 自检与交付(补文档与工作记录)

## 方案备选(至少 2 条路径)

### 方案A(推荐): 完整 Pixi 化(更可复现,一次到位)

- 新增 `pixi.toml`,把 notebook 运行所需的关键依赖(含 PyTorch,Open3D,以及两个 CUDA 扩展等)写入 Pixi manifest.
- notebook 里只做三件事:
  1) 安装 pixi CLI(若未安装).
  2) `pixi install` 安装环境.
  3) 所有脚本统一 `pixi run python ...` 执行.
- 优点: notebook 运行路径最干净,依赖来源统一,排障成本更低.
- 代价: 需要一次性梳理依赖,并处理少数包的 conda/pypi 选择.

### 方案B: 先能用(改动最少,后续再优雅)

- 只在 notebook 里把 `pip install` 换成:
  - `pixi install`
  - `pixi run pip install -r requirements.txt`
- 依赖仍主要由 `requirements.txt` 驱动.
- 优点: 迁移快,基本不需要梳理 conda 依赖.
- 代价: 仍会碰到编译/轮子缺失问题,可复现性也较弱.

## 做出的决定

- 采用方案A.
  - 理由: 目标是"把 notebook 的依赖安装改造成 pixi 安装",方案A能让 notebook 从入口到执行都走 Pixi,语义最一致.

## 关键问题

1. Pixi 环境的 Python 版本选什么?
   - 决定: 先按 notebook/Colab 的现实情况选 `python=3.10.*`.
2. PyTorch/CUDA 版本怎么 pin?
   - 决定: 先 pin 到论文作者 README 提到的 `pytorch=1.13.1` + `cuda=11.6`(通过 `pytorch-cuda=11.6`).

## 遇到的错误

- (暂无)

## 状态

**已完成**: notebook 依赖安装已迁移到 Pixi,并完成收尾记录与自检.

---

## 2026-02-21T06:26:20+00:00 追加: Pixi DNS 报错修复

### 现象

用户在执行 `pixi install` 时失败,报错类似:

- `failed to solve requirements ...`
- `dns error: failed to lookup address information: Name does not resolve`
- 目标域名: `conda.anaconda.org`

### 诊断(本质)

这不是依赖冲突,而是网络层面的 DNS 解析失败.
Pixi 默认会从 `conda.anaconda.org` 拉取 `conda-forge`/`pytorch`/`nvidia` 三个 channel 的 repodata.
当 DNS 无法解析该域名时,solver 会在下载 repodata 阶段直接失败.

### 修复(回到现象层)

- 新增 `pixi.mirrors.toml`,提供 conda channel 镜像重定向配置.
- notebook 的 `pixi install` 改为:
  - 先尝试 `pixi install`
  - 失败则自动回退 `pixi install --config pixi.mirrors.toml`
- README 增加 DNS 报错时的替代命令说明.

### 状态

已落地上述修复.

---

## 2026-02-21T08:13:26+00:00 追加: 版本组合切换与安装流程调整

### 现象

- 用户遇到 conda 侧报错:
  - `pytorch-cuda 11.6 ... is excluded because candidate not in requested channel: 'nvidia'`
  - 或 `No candidates were found for pytorch-cuda 12.6.*`
- 用户希望使用: `python 3.12` + `cuda 12.6` + `pytorch 2.6`.

### 诊断(本质)

- `pixi install` 不支持 `--config`,之前的写法需要更正为 `.pixi/config.toml`.
- conda 的 `pytorch` channel 当前并没有 `pytorch 2.6` 或 `pytorch-cuda 12.6`,因此无法用纯 conda 满足该组合.
- 当项目同时包含 conda 与 PyPI 依赖时,Pixi 默认会拉取 conda<->pypi mapping,部分网络环境下访问 GitHub 也会失败.
- 本仓库的两个 CUDA 扩展子模块 `setup.py` 顶层会 `import torch`,如果把它们当成 pypi path 依赖参与 lock,会在 torch 未安装时失败.

### 修复(回到现象层)

- PyTorch 版本组合改为 PyPI 官方 wheel:
  - `torch ~=2.6.0`, `torchvision ~=0.21.0`
  - index: `https://download.pytorch.org/whl/cu126`(对应 CUDA 12.6)
- 增加本地 mapping:
  - `pixi.toml` 增加 `conda-pypi-map = { conda-forge = "conda_pypi_map.json" }`
- CUDA 扩展改为安装后置:
  - `pixi.toml` 提供 task: `pixi run install-ext`
  - notebook 在 `pixi install` 后会自动执行 `pixi run install-ext`
- 镜像配置正确落地:
  - 失败回退逻辑改为写入 `.pixi/config.toml`,而不是使用不存在的 `--config` 参数.

### 状态

已完成上述调整,并已通过 `pixi lock` 验证求解成功.

---

## 2026-02-21T08:31:00+00:00 追加: 修复 `pixi run install-ext` 失败(torch 在 build isolation 中缺失)

### 现象

用户在 notebook 里执行 `pixi run install-ext` 时失败,核心报错为:

- `ModuleNotFoundError: No module named 'torch'`
- 出现场景: pip "Getting requirements to build editable" 阶段

### 诊断(本质)

- `pip install -e ...` 默认启用 PEP517/PEP660 的 build isolation.
- pip 会创建一个临时的构建虚拟环境(`/tmp/pip-build-env-*/overlay/...`).
- 两个 CUDA 扩展子模块的 `setup.py` 在顶层直接 `import torch`,
  导致临时构建环境里没有 torch 时立刻报错.

### 方案备选(至少 2 条路径)

#### 方案A(推荐): 禁用 build isolation,复用 Pixi 环境里的 torch

- 修改 Pixi task `install-ext`,为 pip 增加 `--no-build-isolation`.
- 优点: 不改子模块源码,改动集中且可控.
- 代价: 需要保证在执行 `install-ext` 前,Pixi 环境已安装好 torch.

#### 方案B: 改子模块 `setup.py`,避免顶层 import torch

- 把 `from torch.utils.cpp_extension import ...` 延迟到 `setup()` 调用内部再 import.
- 优点: 即使隔离构建也能跑过 metadata 阶段,更符合打包最佳实践.
- 代价: 会修改 submodule,需要更谨慎,也可能和上游更新冲突.

### 阶段(本次追加任务)

- [x] 阶段1: 调整 Pixi task(禁用 build isolation)
- [x] 阶段2: 复验 torch 导入与扩展安装
- [x] 阶段3: 更新记录(笔记/错误修复/工作日志)

### 状态

**已完成**: `pixi run install-ext` 在 Pixi 环境下可稳定编译安装,并修复 `simple_knn` 运行期导入问题.

---

## 2026-02-21T08:50:00+00:00 追加: 从多机位视频生成 MultipleView 数据集

### 目标

给定一个包含多机位视频的目录(例如 `/cloud/cloud-s3fs/SelfCap/bar-release/videos`),
自动生成本仓库 MultipleView 训练所需的数据结构与派生文件,最终满足:

- `data/multipleview/<dataset>/camXX/frame_00001.jpg ...`
- `data/multipleview/<dataset>/sparse_/cameras.bin` + `images.bin` + `points3D.bin`
- `data/multipleview/<dataset>/points3D_multipleview.ply`
- `data/multipleview/<dataset>/poses_bounds_multipleview.npy`

### 现象(用户输入)

- 视频目录: `/cloud/cloud-s3fs/SelfCap/bar-release/videos`
- 文件命名示例: `02.mp4 ... 19.mp4`

### 诊断(本质)

- 现有 `multipleviewprogress.sh` 假设输入已经是每个相机一个 `camXX/frame_00001.jpg` 这样的帧图.
- 现有 `scripts/extractimages.py` 只会拷贝每个相机的第一帧去跑 COLMAP.
- 因此缺的关键环节是: "从多路视频 -> 统一抽帧并落盘到 camXX".
- 同时,`multipleviewprogress.sh` 里还包含 `git clone LLFF` 的网络依赖,对离线/受限网络不友好.

### 方案备选(至少 2 条路径)

#### 方案A(推荐,一次到位): Python 脚本全流程(抽帧 + COLMAP + poses_bounds + 点云)

- 新增一个脚本 `scripts/preprocess_multipleview_from_videos.py`.
- 输入: 视频目录 + dataset name.
- 输出: 直接生成 MultipleView 训练所需的完整目录与文件.
- 优点: 单入口,可重复执行,不依赖 `git clone LLFF`,更稳定.
- 代价: 需要在脚本里实现 `poses_bounds_multipleview.npy` 的生成逻辑(按 LLFF 规则).

#### 方案B(先能用,最省改动): 只做抽帧,后续仍跑 `multipleviewprogress.sh`

- 新增脚本仅负责抽帧到 `data/multipleview/<dataset>/camXX`.
- 其余仍由 `multipleviewprogress.sh` 生成 sparse/点云/poses_bounds.
- 优点: 改动更少.
- 代价: 仍依赖 `git clone LLFF` 和 `pip install scikit-image`,网络不稳定时会炸.

### 做出的决定

- 采用方案A.
  - 理由: 用户明确希望"从视频生成 multipleview 所需数据",全流程脚本更符合预期,也更可复现.

### 阶段

- [x] 阶段1: 调研现有 multipleview 数据格式与代码路径
- [x] 阶段2: 设计脚本输入/输出与抽帧策略(含截断对齐)
- [x] 阶段3: 实现脚本(抽帧 + COLMAP + poses_bounds + 点云)
- [x] 阶段4: 小样本验证(至少对 2 个视频跑通)并记录到 WORKLOG

### 状态

**已完成**: 已提供全流程脚本,并用 `/cloud/cloud-s3fs/SelfCap/bar-release/videos` 前 2 路视频完成小样本验证.

---

## 2026-02-21T10:00:00+00:00 追加: 把 MultipleView 生成命令手册化写入 README,并给出对标 FreeTimeGsVanilla 的推荐参数

### 目标

1. 在本仓库 `README.md` 中,把 MultipleView 数据生成相关命令手册化:
   - 传统 `multipleviewprogress.sh` 路径
   - 新增的 "多机位视频一键生成" 路径(`pixi run prep-multipleview`)
   - 参数解释与常见坑(尤其是 headless COLMAP)
2. 在不改变 FreeTimeGsVanilla 的前提下,给出一个"体量相近"的生成参数建议,用于公平对比.

### 阶段

- [x] 阶段1: 补齐 README 文档(命令 + 参数 + 输出结构)
- [x] 阶段2: 基于 FreeTimeGsVanilla 的默认 frame 体量,给出推荐命令组合
- [x] 阶段3: 更新 WORKLOG/notes 记录

### 状态

**已完成**: README 已补齐命令手册,并给出对标 FreeTimeGsVanilla demo 体量的推荐生成参数.

---

## 2026-02-21T12:24:13+00:00 追加: 为公平评估对比,保持输入数据质量对等(训练侧下采样,而不是生成侧缩放)

### 现象(用户反馈)

- README 里示例用了 `--max-size 940` 之类的参数,看起来像把数据在生成阶段就缩放到 1/4.
- 但用户关心的不是预览视频分辨率,而是最终 `.pt`/模型评估对比要公平.
- 因此需要保证"输入数据质量对等": 尽量保留原始帧质量,避免不可逆的信息损失.

### 诊断(本质)

- FreeTimeGsVanilla 的 `data_factor` 语义是"训练/加载侧下采样",而不是"落盘帧文件就缩小".
  - 证据: `results/bar_release_full/out_0_61/cfg.yml` 里 `data_factor: 4`,
    但 `work_0_61/data/images/*/*.jpg` 仍是 `2110x3760` 原图.
- 本仓库当前 MultipleView 训练读取(`scene/multipleview_dataset.py`)是直接按落盘分辨率读入,
  没有类似 `data_factor` 的训练侧下采样开关.
- 因此如果我们用 `--max-size` 在生成阶段缩放,会让输入质量不可逆下降,对公平对比不利.

### 方案备选(至少 2 条路径)

#### 方案A(推荐): 生成阶段保留原始分辨率,训练阶段用 `--resolution` 做等价 data_factor 下采样

- 生成数据时使用 `--max-size 0` 保留原始帧尺寸.
- 在训练/评估时通过新增的 MultipleView 读取侧下采样能力,让 `--resolution 4/8` 等价于 FreeTime 的 `data_factor`.
- 优点:
  - 输入数据质量对等(原图保留).
  - 可逆(同一份数据可用不同 factor 做消融/对比).
  - 更贴近 FreeTime 的语义,对比更公平.
- 代价:
  - 需要改一小段数据加载代码,并在 README 里把“缩放发生在哪一侧”讲清楚.

#### 方案B(现状): 生成阶段直接 `--max-size` 缩放,训练阶段不做下采样

- 优点: 实现简单.
- 代价: 不可逆降质,且语义与 FreeTime `data_factor` 不一致,不利于公平评估.

### 做出的决定

- 采用方案A.

### 阶段(本次追加任务)

- [x] 阶段1: 增加 MultipleView 训练侧下采样(`--resolution` 作为 data_factor)
- [x] 阶段2: preprocess 脚本补齐 `--frame-start/--frame-end`(对齐 `[0,61)` 语义)
- [x] 阶段3: 更新 README 手册(明确推荐 `--max-size 0` + `--resolution 4/8`)
- [x] 阶段4: 自检与记录(验证读取尺寸/FOV 与预期一致,更新 WORKLOG/notes/LATER_PLANS)

### 状态

**已完成**: 已对齐 FreeTime 的 data_factor 语义(训练侧下采样),并补齐按帧索引截取能力与 README 手册说明.

---

## 2026-02-21T13:08:56+00:00 追加: README 记录"对比命令"(生成 + 训练)并提交

### 目标

把下面这组用于对标 FreeTimeGsVanilla 的命令,以"一眼就能照抄"的形式写入 README:

1. 生成数据(保留原始分辨率,按帧索引对齐 `[0,61)`):
   - `pixi run prep-multipleview ... --frame-start 0 --frame-end 61 --max-size 0`
2. 训练评估对齐 data_factor=4:
   - `pixi run train ... --resolution 4`

### 阶段

- [x] 阶段1: README 补充完整对比命令块
- [x] 阶段2: 更新 WORKLOG 记录
- [x] 阶段3: git commit(不包含任何大文件产物)

### 状态

**已完成**: README 已补齐对比命令(生成 + 训练),WORKLOG 已记录,并已完成 git 提交(未包含任何大文件产物).

---

## 2026-02-21T13:09:14+00:00 追加: 在 README 记录 FreeTime 对比用的两条命令(生成 + 训练)

### 目标

- 把用户确认可用于公平评估对比的两条命令,以可直接复制粘贴的形式写入 README.
  - `pixi run prep-multipleview ...`
  - `pixi run train ... --resolution 4`

### 状态

已完成.

---

## 2026-02-21T13:37:57+00:00 追加: 修复 COLMAP feature_extractor 被 SIGKILL(-9) 杀死

### 现象(用户反馈)

在执行多机位视频预处理脚本 `scripts/preprocess_multipleview_from_videos.py` 时,跑到 COLMAP 特征提取阶段失败:

- `RuntimeError: 命令失败(returncode=-9): colmap feature_extractor ...`
- COLMAP 输出只停在:
  - `Feature extraction`

### 诊断(本质)

- `returncode=-9` 通常表示进程被 `SIGKILL` 强制终止.
  - 常见原因是系统/容器触发 OOM killer(内存不够时直接杀进程),或被外部手动 kill.
- 脚本原先对 `colmap feature_extractor` 使用了偏激进的 SIFT 参数:
  - `--SiftExtraction.max_image_size 4096`
  - `--SiftExtraction.max_num_features 16384`
  - `--SiftExtraction.estimate_affine_shape 1`
  - `--SiftExtraction.domain_size_pooling 1`
  这组参数比 COLMAP 默认值更耗内存/更慢,在高分辨率视频(接近 4K)上更容易被 OOM 杀死.
- 另外,脚本没有显式限制 COLMAP 的线程数,默认 `num_threads=-1` 会吃满 CPU 核数,也会放大内存峰值.

### 方案备选(至少 2 条路径)

#### 方案A(推荐): 调整为更稳的默认参数 + 暴露 CLI 可配置项

- 把默认值拉回到更接近 COLMAP 默认的水平,并允许用户通过命令行覆盖:
  - `--SiftExtraction.max_image_size`/`--SiftExtraction.max_num_features`
  - `--SiftExtraction.num_threads`
  - affine/dsp 开关
- 同时改进错误提示: 当检测到 `SIGKILL` 时,明确提示"可能是 OOM",并给出可直接复制的降参命令.

优点: 稳定性提升,并且可控可调,不引入太多复杂逻辑.
代价: 默认的重建质量可能略降(但对静态相机标定通常足够).

#### 方案B: 保持当前默认参数,遇到 SIGKILL 自动降级重试

- 先用当前参数跑一次.
- 若检测到被信号杀死(尤其 SIGKILL),自动切到更保守的参数重试一次.

优点: 默认尽量维持"高质量".
代价: 逻辑更复杂,且重复跑一次会浪费时间.

### 做出的决定

- 先落地方案A.
- 如果后续仍有环境不稳定,再补方案B 的自动降级重试.

### 阶段(本次追加任务)

- [x] 阶段1: 复核 COLMAP 参数默认值与 -9 含义,明确调参方向
- [x] 阶段2: 为 preprocess 脚本增加 COLMAP SIFT 参数/线程的 CLI 选项,并调整默认值
- [x] 阶段3: 改进 `--keep-colmap-tmp` 行为(失败时也能保留临时目录)
- [x] 阶段4: 更新 README 参数说明 + 本地快速自检

### 状态

**已完成**: 脚本已暴露 COLMAP 调参开关并降低默认开销,同时增强 `--keep-colmap-tmp` 与错误提示,README 已同步.

---

## 2026-02-21T17:04:40+00:00 追加: git 提交并推送到 raiscui/4DGaussians

### 目标

- 把当前工作区改动整理成一次干净的提交(commit).
- 避免把本地大文件或私密配置推到远端:
  - `data.zip`(本地数据压缩包,体积很大)
  - `.envrc.private`(私密环境变量,已在 `.gitignore` 里忽略)
- 将远端推送目标切换到: `https://github.com/raiscui/4DGaussians.git`

### 现状

- 当前分支: `master`
- 当前 `origin` 仍指向上游 fork,需要改为 `raiscui/4DGaussians`.
- 子模块工作区无改动,无需更新 submodule 指针.

### 方案备选(至少 2 条路径)

#### 方案A(推荐): 单次 commit + 推送 master(最少噪音)

- 在 `.gitignore` 增加 `data.zip`,防止误提交.
- 提交范围: 本次脚本改良 + 文档/记录 + 新增 `.envrc`(不含 token 实值,仅引用环境变量).
- `git remote set-url origin https://github.com/raiscui/4DGaussians.git`
- `git push -u origin master`

优点: 历史最干净,一次性完成,后续同步也最直觉.
代价: 需要确保本机具备 GitHub 的 push 权限(HTTP token/SSH).

#### 方案B: 不改 origin,新增 remote 再推送(更保守)

- 保留现有 `origin`(方便对比/回溯),新增:
  - `git remote add raiscui https://github.com/raiscui/4DGaussians.git`
  - `git push -u raiscui master`

优点: 不破坏现有 remote.
代价: remote 变多,后续默认 push 目标不够直观.

### 做出的决定

- 采用方案A.
  - 理由: 你明确要求 push 到 `raiscui/4DGaussians`,直接把 `origin` 对齐为目标仓库更符合常规协作模型.

### 阶段(本次追加任务)

- [x] 阶段1: 复核改动列表,确认不提交大文件/私密文件
- [ ] 阶段2: 补齐忽略规则(`data.zip`),并完成暂存(staging)
- [ ] 阶段3: 创建 commit(信息清晰,可回滚)
- [ ] 阶段4: 配置远端并 push 到 `raiscui/4DGaussians`

### 状态

**进行中**: 正在补齐忽略规则并准备 staging.
