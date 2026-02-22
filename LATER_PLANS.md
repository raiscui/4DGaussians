# LATER_PLANS

> 只追加,不在中间插入. 记录本次没做但值得后续做的事.

## 2026-02-21T06:26:20+00:00

- (候选) 把 `torch_cluster` 依赖做成 Pixi feature 可选项,并用 `[pypi-options].find-links` 指向 PyG 的 wheel 索引,避免用户手动装.
- (候选) notebook 里那行 `!export ...proxy...` 在 Jupyter 语义下不持久,后续可以统一改为 `os.environ[...] = ...` 的方式,让代理对所有 `!` 命令生效.

## 2026-02-21T06:26:20+00:00 补充

- 已在 `4DGaussians_rais.ipynb` 的首个 cell 提供了基于 `os.environ[...]` 的代理写法(默认注释掉,避免误伤未使用代理的人).
- 仍未把代理逻辑做成"自动检测/开关式",后续如有需要再继续做成更通用的方案.

## 2026-02-21T10:00:00+00:00

- (候选) 增加 `--strict-frame-count` 开关: 当多路相机帧数不一致时直接报错,而不是自动截断到最短长度(更利于发现数据源问题).

## 2026-02-21T12:24:13+00:00

- 已完成: `scripts/preprocess_multipleview_from_videos.py` 已增加 `--frame-start/--frame-end/--frame-step`.
  - 现可精确对齐 FreeTime 的 `[start_frame,end_frame)` 语义,不再依赖 `--fps` 的时间戳重采样近似.

## 2026-02-21T13:43:41+00:00

- (候选) `scripts/preprocess_multipleview_from_videos.py` 在检测到 `SIGKILL`/`OOM` 时,自动用更保守的 COLMAP SIFT 参数重试一次(减少用户手动调参成本).

## 2026-02-22T07:47:00+00:00

- (候选) `render.py` 的 `imageio.mimwrite` 在输出 mp4 时可能触发 `macro_block_size=16` 的自动 resize 警告.
  - 可选改良方向:
    - 方案A: 输出前对帧做 padding,让 (H,W) 都是 16 的倍数(不改有效内容,避免 codec 限制).
    - 方案B: `mimwrite(..., macro_block_size=1)` 禁止自动 resize(但可能牺牲部分播放器兼容性).

## 2026-02-22T07:59:50+00:00

- 已完成: 已在 `render.py` 写 mp4 前对帧做 edge padding 到 16 的倍数.
  - 不再触发 `macro_block_size=16` warning.
  - 也避免了 imageio 的隐式 resize(插值).

## 2026-02-22T08:44:40+00:00

- (候选) 增加一个小工具脚本: 输入 `.ply` 或 `.pt/.pth` 路径,自动打印"高斯数量,每部分 tensor 占用(MB),是否包含 optimizer state",方便对比不同方法的体积来源.

## 2026-02-22T08:48:56+00:00

- (候选) MultipleView 的 spiral video 轨迹再增强:
  - 增加 `video_spiral_hold_end`: 结尾也停留 N 帧,方便视频收尾更稳.
  - 增加 `video_spiral_hold_pose`: 支持选择停留在 `avg_pose`(平均位姿) 或 `start_pose`(spiral 第 1 帧),便于对齐“中心视角”的直觉.
  - 增加 `video_spiral_theta_schedule`: 例如 ease-in-out,在整段轨迹上做平滑加减速,减少“突然开始转”的观感.
