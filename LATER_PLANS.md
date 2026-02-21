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
