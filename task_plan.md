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
- 已将 `origin` 指向 `https://github.com/raiscui/4DGaussians.git`.
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
- [x] 阶段2: 补齐忽略规则(`data.zip`),并完成暂存(staging)
- [x] 阶段3: 创建 commit(信息清晰,可回滚)
- [x] 阶段4: 配置远端并 push 到 `raiscui/4DGaussians`

### 状态

**已完成**: 已创建提交并 push 到 `raiscui/4DGaussians` 的 `master` 分支.

---

## 2026-02-21T18:07:39+00:00 追加: MultipleView 训练 `--configs` 占位符澄清与文档修复

### 目标

- 让 README 的示例命令不再引用不存在的 `arguments/multipleview/xxx.py`.
- 解释清楚: `--configs` 需要传入一个实际存在的 `arguments/**/*.py` 配置文件,用于覆盖 `train.py` 的参数默认值.

### 方案备选(至少 2 条路径)

#### 方案A(推荐): 直接用默认配置,示例命令可立即运行

- README 示例命令统一改为 `--configs arguments/multipleview/default.py`.
- 同时补一句说明: 若需要按数据集微调,再复制/继承 `default.py` 生成自己的 `<dataset>.py`.

优点: 开箱即用,最少踩坑.
代价: 如果用户想按数据集精细调参,需要额外一步复制/修改.

#### 方案B: 完全按原文意图,要求用户手动创建 `<dataset>.py`

- 保持 README 的 "每个数据集一个 config" 思路.
- 但要给出明确模板,避免用户不知道该写哪些 key.

优点: 结构更清晰,每个数据集参数独立.
代价: 新手会被 "先写配置文件" 卡住,不如方案A顺滑.

### 做出的决定

- 采用方案A,并在 README 补充方案B的写法(复制/继承 default).
  - 理由: 用户反馈的痛点是 "文件不存在",优先让示例命令可直接运行,同时保留可扩展路径.

### 阶段(本次追加任务)

- [x] 阶段1: 调研仓库内 MultipleView 可用 config
- [x] 阶段2: 修复 README 的占位符与拼写
- [x] 阶段3: 自检(确认文档中的路径都存在)
- [x] 阶段4: 记录到 WORKLOG

### 状态

**已完成**: README 的 MultipleView 示例命令改为 `arguments/multipleview/default.py`,并补充了可选的 per-dataset 配置写法.

---

## 2026-02-22T07:39:20+00:00 新任务: MultipleView 渲染视频视角乱跳排查与修复

### 目标

- 让 MultipleView 的渲染产物不再出现"视角乱跳"的观感,更接近 dnerf bouncingballs 的观看体验.
- 同时把 MultipleView 的 train/test 划分从"抽几帧"改为"按相机划分",避免 test 集合既短又不利于观察时序.

### 现象(用户反馈)

- 命令:
  - `pixi run python render.py --model_path "output" --skip_train --configs arguments/multipleview/default.py`
- 结果:
  - MultipleView 渲染出来的 mp4 视角乱跳,不像 dnerf bouncingballs 那样稳定.

### 初步调查(证据)

- 当前 `output/test/ours_30000/renders` 为 54 张:
  - 54 = 18 cams * 3 frames.
  - 说明: MultipleView 的 test split 现状是"每个相机只取 3 帧",并把多相机视角直接串成一个视频.
- 当前 `output/video/ours_30000/renders` 为 300 张:
  - spiral 相机轨迹在数值上是连续的,没有明显跳变点.

### 诊断(本质)

- "乱跳"不是渲染器坏了,而是 test 视频的帧序列本身来自不同相机视角.
- MultipleView 当前的 test 集合不是 camera hold-out,而是"每个相机抽少量帧",因此:
  - 视频观看会跳视角.
  - evaluation 语义也更像 quick sanity,而不是标准的 novel-view 测试.

### 方案备选(至少 2 条路径)

#### 方案A(推荐): 按相机切分 train/test + test 输出按相机分别生成 mp4

- 在 `readMultipleViewinfos` 中使用 `llffhold` 在"相机维度"切分 train/test.
- test 集合对 hold-out 相机保留完整时间序列(不再只取 3 帧).
- `render.py` 对 MultipleView 的 test 集合按 `camXX` 分组写 mp4,避免把不同视角强行拼成一条视频.

优点:
- 观看体验符合直觉(单相机视频不跳).
- train/test 语义更正确,更接近 DyNeRF 的 hold-out view 评估方式.
代价:
- test 帧数变多,会多写一些渲染产物,README 需要同步更新说明.

#### 方案B: 先能用: 不改 train/test 语义,只改 render 输出的 mp4 命名/分组

- 保持 test 仍为"每相机 3 帧"的 quick sanity 设计.
- render 侧不再生成一个混合视角的 `video_rgb.mp4`,改为按相机分别输出 mp4,避免观感"乱跳".

优点:
- 对训练/评估流程影响更小.
代价:
- 每个 test mp4 只有 3 帧,仍不适合作为"时序视频"去观察动态效果.

### 做出的决定

- 优先落地方案A.
  - 理由: 用户的诉求是"视频看起来稳定",根因是 test split + 拼接策略,方案A能从语义层面把数据和输出一起改正确.

### 阶段(本次任务)

- [x] 阶段1: 复现与定位(确认是哪一个 mp4 在跳)
- [x] 阶段2: 设计并实现 MultipleView camera-level split
- [x] 阶段3: 改良 render 输出(按相机生成 test mp4)
- [x] 阶段4: 自检与文档同步(README/WORKLOG/ERRORFIX)

### 状态

**已完成**: MultipleView 的 test 集合改为 camera-level hold-out,render 输出提供 per-camera mp4,默认 `video_rgb.mp4` 不再因多相机拼接而显得"视角乱跳". 文档与工作记录已同步.

---

## 2026-02-22T07:56:51+00:00 追加: 消除 render mp4 的 macro_block_size 自动 resize 警告

### 目标

- 让 `render.py` 输出 mp4 时不再出现 `IMAGEIO FFMPEG_WRITER WARNING: ... macro_block_size=16 ... resizing ...` 的警告.
- 避免 imageio/ffmpeg 在写 mp4 时对帧做隐式缩放(插值),保持渲染内容像素不被改动.

### 现象(用户终端输出)

- MultipleView 渲染结束后出现多条警告,示例:
  - `resizing from (527, 940) to (528, 944) ...`

### 诊断(本质)

- MultipleView 在 `--resolution 4` 下是 floor 下采样:
  - 原始高度 2110 -> `2110 // 4 = 527`
  - 原始宽度 3760 -> `3760 // 4 = 940`
- 由于 (H,W) 不是 16 的倍数,imageio 的 ffmpeg writer 为了 codec/播放器兼容性会自动 resize 到 (528,944).
- 这虽然不是 error,但属于"静默改变输出",并且会把相同 warning 打很多次(因为我们会写多条 mp4).

### 方案备选(至少 2 条路径)

#### 方案A(推荐): 写 mp4 前做 padding 到 16 倍数

- 对每帧做 edge padding,把 (H,W) 补齐到最接近的 16 倍数.
- 优点:
  - 不做缩放插值,内容像素保持不变.
  - 仍满足大多数编码器/播放器的尺寸要求.
- 代价:
  - mp4 尺寸会比 png 大一圈(通常只多 0-15 像素边框,很小).

#### 方案B: `macro_block_size=1` 禁止自动 resize

- 直接传 `imageio.mimwrite(..., macro_block_size=1)`.
- 优点: mp4 尺寸与渲染帧完全一致.
- 代价: 可能牺牲部分播放器兼容性(尤其是奇数尺寸).

### 做出的决定

- 采用方案A.
  - 理由: 我们更关心"不被静默缩放"与"兼容性稳定",padding 比禁用约束更稳.

### 阶段(本次追加任务)

- [x] 阶段1: 在 render 写 mp4 前实现 padding
- [x] 阶段2: 自检(渲染不再打印 warning)
- [x] 阶段3: 记录到 WORKLOG/LATER_PLANS/task_plan

### 状态

**已完成**: `render.py` 写 mp4 前会自动做 edge padding 到 16 倍数,不再触发 `macro_block_size=16` warning,也避免了隐式 resize.

---

# 任务计划: 对比输出文件体积差异(4DGaussians vs FreeTimeGsVanilla)

时间: 2026-02-22T08:38:30+00:00

## 目标

解释为什么本项目 `output/point_cloud/iteration_30000/` 体积较小,而 FreeTimeGsVanilla 的 `ckpt_29999.pt` 很大.

同时给出如何做公平对比,以及如何保存同类产物的建议.

## 阶段

- [ ] 阶段1: 现状确认(两边各保存了什么)
- [ ] 阶段2: 结构拆解(哪些字段占体积)
- [ ] 阶段3: 结论与对齐方案(怎么让两边可比)
- [ ] 阶段4: 记录与交付

## 方案备选(至少 2 条路径)

### 方案A(推荐): 对齐"保存内容"再比体积

- 本项目用 `--checkpoint_iterations 30000` 生成 `chkpnt_*_30000.pth`,再与 FreeTime 的 `ckpt_29999.pt` 对比.
- 或在 FreeTime 侧导出点云(只含高斯参数),再与本项目 `point_cloud.ply` 对比.

### 方案B: 先能解释清楚,后续再补工具

- 先基于代码与实际文件,解释体积差异主要来自: 文件格式,是否保存 optimizer state,以及高斯数量/参数维度.
- 后续如需要,再补一个 `scripts/inspect_ckpt.py` 自动打印 ckpt 各部分占用.

## 关键问题

1. `output/point_cloud/iteration_30000/` 具体包含哪些文件,分别是什么?
2. FreeTime 的 `ckpt_29999.pt` 里包含哪些 key,是否含 optimizer/ema/数据缓存?
3. 两边高斯数量,每个高斯的参数维度,是否同量级?

## 状态

**目前在阶段1**: 收集两边产物的真实内容与体积.

---

# 任务计划: MultipleView video 镜头在中心停留更久

时间: 2026-02-22T08:40:57+00:00

## 目标

用户执行:

- `pixi run python render.py --model_path "output" --skip_train --configs arguments/multipleview/default.py`

观看:

- `output/video/ours_30000/video_rgb.mp4`

希望: 镜头(spiral novel-view 轨迹)在"中心视角"附近停留更久,不要一开始就快速绕圈.

## 阶段

- [ ] 阶段1: 现状定位(轨迹生成链路与可改点)
- [ ] 阶段2: 方案确定(至少两条路径,选一条落地)
- [ ] 阶段3: 实现与参数化(兼容旧 cfg_args)
- [ ] 阶段4: 自检与交付(最小验证 + 记录)

## 方案备选(至少 2 条路径)

### 方案A(推荐): 在数据集侧改良 spiral 轨迹(可配置)

在 `scene/multipleview_dataset.py:get_video_cam_infos()` 生成 spiral pose 时:

- 支持配置 `video_spiral_hold_start`:
  - 在 spiral 开始前,重复若干帧"起始 pose"(相机不动),但 time 仍按总帧数线性推进.
  - 视觉效果: 镜头会在中心视角停留更久,同时场景动态仍会继续播放.
- 支持配置 `video_spiral_n_rots`/`video_spiral_rads_scale`:
  - 允许把默认 2 圈改成 1 圈,或缩小半径,让镜头整体更"稳".

优点:
- 改动在相机轨迹生成处,语义正确,输出的 renders 图片序列也与 mp4 一致.
- 参数可控,不同数据集/需求可复用.
- 兼容旧模型: 即使 `output/cfg_args` 没有这些字段,也能用默认值跑通.

代价:
- 需要改动 MultipleView 读数链路(把参数从 args 传到 dataset),并做一次最小自检.

### 方案B: 先能用: 只在 render 写 mp4 前做“首帧重复”

在 `render.py` 写 `video_rgb.mp4` 前:

- 若 `cam_type=="MultipleView" && name=="video"`,把 `render_images[0]` 重复 N 次再写入 mp4.

优点:
- 改动最小,不碰数据集与相机轨迹.

代价:
- renders 目录的 png 序列不含这些停留帧,mp4 与 png 不一致.
- 不容易调更复杂的“慢速绕圈/缩半径”等需求.

## 做出的决定

- 采用方案A.
  - 理由: 用户诉求本质是"相机轨迹的观感",应在轨迹生成处解决,并保持 png 序列与 mp4 一致.

## 状态

**目前在阶段1**: 继续定位 MultipleView video spiral 轨迹生成与参数传递点.

## 2026-02-22T08:43:13+00:00 进展更新

- [x] 阶段1: 现状确认(两边各保存了什么)
  - 4DGaussians: `output/point_cloud/iteration_30000/` = `point_cloud.ply` + `deformation*.pth`,总计约 40MB.
  - FreeTime: `ckpt_29999.pt` 顶层含 `splats` + `optimizers`,文件约 978MB.
- [x] 阶段2: 结构拆解(哪些字段占体积)
  - FreeTime `splats` 约 326MB(主要是 `shN`),`optimizers` 约 652MB(Adam 的 `exp_avg/exp_avg_sq`).
  - FreeTime 高斯数量约 133 万,本项目示例高斯数量约 12 万.
- [x] 阶段3: 结论与对齐方案(怎么让两边可比)
  - 对比"推理模型体积"时,应只比模型参数(FreeTime 的 `splats`,本项目的 `point_cloud.ply`+`deformation.pth`).
  - 对比"可继续训练的 checkpoint"时,本项目需要启用 `--checkpoint_iterations` 生成 `chkpnt_*.pth`(含 optimizer state).
- [x] 阶段4: 记录与交付

## 状态

**已完成**: 已解释体积差异的根因(保存内容不同 + 高斯数量不同 + optimizer state),并给出对齐比较的建议.

## 2026-02-22T08:48:56+00:00 进展更新: MultipleView video 镜头中心停留更久

- [x] 阶段1: 现状定位(轨迹生成链路与可改点)
  - `output/video/.../video_rgb.mp4` 来自 MultipleView 的 `video_cam_infos`(spiral).
  - 生成点在 `scene/multipleview_dataset.py:get_video_cam_infos()` -> `scene/neural_3D_dataset_NDC.py:get_spiral()`.
- [x] 阶段2: 方案确定
  - 选择在轨迹生成处落地(而不是 render 写 mp4 时硬重复首帧),保证 png 序列与 mp4 一致.
- [x] 阶段3: 实现与参数化(兼容旧 cfg_args)
  - 增加可配置参数:
    - `video_spiral_hold_start`: 起始 pose 停留帧数(相机不动,但 time 仍推进).
    - `video_spiral_n_rots`/`video_spiral_rads_scale`/`video_n_views`: 控制绕圈次数/半径/采样帧数.
  - 兼容旧模型:
    - `utils/params_utils.py:merge_hparams()` 允许在 merge 配置时补齐旧 cfg_args 缺字段.
    - `scene/__init__.py` 使用 `getattr(..., default)` 兜底.
- [x] 阶段4: 自检与交付
  - `python3 -m py_compile` 对改动文件做了语法自检.
  - 默认参数写入 `arguments/multipleview/default.py`,用户命令不变即可生效.

## 状态

**已完成**: MultipleView 的 video(spiral) 轨迹已支持“中心视角停留更久”的可配置参数,默认会在起始视角停留 60 帧并把绕圈次数降为 1 圈.

## 2026-02-22T09:41:04+00:00 决策修正: video 时间维度应 loop,而不是靠 hold

### 现象

- 用户反馈: `output/video/ours_30000/video_rgb.mp4` 里“内容该运动的没有动”.

### 诊断(本质)

- MultipleView 的真实时间序列长度为 x(例如 cam01 有 61 帧).
- spiral video 的相机轨迹帧数通常是 N(默认 300).
- 若 time 按 `idx/N` 线性推进,则动作会被“拉慢”约 N/x 倍.
- 用 `video_spiral_hold_start` 只是让相机不动,但并不能解决“动作循环播放”的诉求.

### 修正方向(做出的决定)

- 增加 video time 的 loop 模式:
  - 相机轨迹仍按 N 帧生成(镜头足够平滑).
  - time 改为按 x 帧循环: `time = (idx % x) / x`.
- MultipleView 的默认配置改为启用该 loop 模式,并把 `video_spiral_hold_start` 默认回退为 0.

### 状态

**目前在执行**: 实现 MultipleView video time loop,并做最小自检与记录同步.

---

# 任务计划: README 补充说明(点数增长机制与 checkpoint 体积)

时间: 2026-02-22T09:15:26+00:00

## 目标

把以下知识点写清楚到 `README.md`:

1. 为什么训练出来的高斯点数可能只有 12 万左右.
2. 哪些参数/代码逻辑决定 densification 的停止与上限.
3. `output/point_cloud/iteration_x/` 与 `chkpnt_*_x.pth` 的区别.
4. 为什么 FreeTimeGsVanilla 的 `ckpt_*.pt` 体积巨大,以及如何做公平对比.

## 阶段

- [ ] 阶段1: 定位 README 插入位置与现有内容
- [ ] 阶段2: 补充 FAQ/说明段落(含参数名与对齐建议)
- [ ] 阶段3: 自检(README 可读性,命令示例正确)
- [ ] 阶段4: 记录与交付

## 方案备选(至少 2 条路径)

### 方案A(推荐): 在 MultipleView/FreeTime 对齐小节旁边加 FAQ

- 优点: 读者在做对齐评估时,紧接着就能看到"点数/体积"这两个最常见疑问.
- 代价: README 的 MultipleView 小节会更长.

### 方案B: 单独新增 "FAQ" 章节

- 优点: 结构更清晰,不会把 FreeTime 对齐内容拉得太长.
- 代价: 读者可能看完对齐段落后不去翻 FAQ.

## 做出的决定

- 采用方案A.
  - 理由: 该问题强相关 MultipleView + FreeTime 对齐评估,放一起更贴合用户心智路径.

## 状态

**目前在阶段1**: 读取 README 现有结构,确认插入点.

## 2026-02-22T09:17:41+00:00 进展更新

- [x] 阶段1: 定位 README 插入位置与现有内容
- [x] 阶段2: 补充 FAQ/说明段落(含参数名与对齐建议)
- [x] 阶段3: 自检(README 可读性,命令示例正确)
- [x] 阶段4: 记录与交付

说明:
- 已在 MultipleView 的 "Fair comparison with FreeTimeGsVanilla" 小节后,追加了两个 FAQ:
  - "点数为什么只有 12 万左右"(densify_until_iter 时间窗口 + 360000 硬上限 + prune).
  - "为什么 iteration_x 快照比 FreeTime ckpt 小"(快照 vs 完整 checkpoint/optimizer state).
