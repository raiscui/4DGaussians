# LATER_PLANS

> 只追加,不在中间插入. 记录本次没做但值得后续做的事.

## 2026-02-21T06:26:20+00:00

- (候选) 把 `torch_cluster` 依赖做成 Pixi feature 可选项,并用 `[pypi-options].find-links` 指向 PyG 的 wheel 索引,避免用户手动装.
- (候选) notebook 里那行 `!export ...proxy...` 在 Jupyter 语义下不持久,后续可以统一改为 `os.environ[...] = ...` 的方式,让代理对所有 `!` 命令生效.

## 2026-02-21T06:26:20+00:00 补充

- 已在 `4DGaussians_rais.ipynb` 的首个 cell 提供了基于 `os.environ[...]` 的代理写法(默认注释掉,避免误伤未使用代理的人).
- 仍未把代理逻辑做成"自动检测/开关式",后续如有需要再继续做成更通用的方案.
