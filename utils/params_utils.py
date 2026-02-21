from __future__ import annotations

from pathlib import Path
from typing import Any


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    深度合并两个 dict.

    规则:
    - 两边都是 dict 的 key,递归合并.
    - 其他情况,以 override 覆盖 base.

    说明:
    - 这里实现的是项目实际用到的一小部分 mmcv.Config 合并语义.
    - 当前 `arguments/**/*.py` 配置里主要用到 `_base_` 继承 + dict 覆盖,无需引入 mmcv 这种重依赖.
    """

    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge_dict(merged[key], value)  # type: ignore[arg-type]
            continue
        merged[key] = value
    return merged


def load_config_file(config_path: str) -> dict[str, Any]:
    """
    加载 `arguments/**/*.py` 形式的配置文件,并支持 `_base_` 继承.

    为什么要自己实现:
    - 原代码使用 `mmcv.Config.fromfile`.
    - 但 notebook/环境里安装 `mmcv` 经常触发编译,非常不稳定.
    - 我们只需要 `_base_` + dict 合并即可,完全没必要引入 mmcv 的重依赖.
    """

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    visited: set[Path] = set()
    return _load_config_recursive(path, visited)


def _load_config_recursive(path: Path, visited: set[Path]) -> dict[str, Any]:
    if path in visited:
        chain = " -> ".join(str(p) for p in list(visited) + [path])
        raise ValueError(f"检测到配置循环引用: {chain}")
    visited.add(path)

    code = path.read_text(encoding="utf-8")

    # 用 exec 执行配置文件,得到配置变量.
    # 注意: 配置文件属于项目仓库内文件,我们只把它当作"配置 DSL"使用.
    global_ns: dict[str, Any] = {"__file__": str(path)}
    local_ns: dict[str, Any] = {}
    exec(compile(code, str(path), "exec"), global_ns, local_ns)

    base_spec = local_ns.get("_base_")
    current_cfg: dict[str, Any] = {
        k: v
        for k, v in local_ns.items()
        # 过滤掉 python 注入的特殊变量.
        if k not in {"__builtins__"}
    }

    # 先加载 base,再让当前配置覆盖 base.
    merged_base: dict[str, Any] = {}
    if base_spec:
        if isinstance(base_spec, (str, Path)):
            base_list = [base_spec]
        elif isinstance(base_spec, (list, tuple)):
            base_list = list(base_spec)
        else:
            raise TypeError(f"_base_ 只支持 str/list/tuple,但得到: {type(base_spec)}")

        for base_item in base_list:
            base_path = (path.parent / str(base_item)).expanduser().resolve()
            base_cfg = _load_config_recursive(base_path, visited)
            merged_base = _deep_merge_dict(merged_base, base_cfg)

    # 删除 _base_ 本身,避免干扰后续 merge_hparams 的 keys().
    current_cfg.pop("_base_", None)
    merged = _deep_merge_dict(merged_base, current_cfg)
    return merged


def merge_hparams(args, config):
    params = ["OptimizationParams", "ModelHiddenParams", "ModelParams", "PipelineParams"]
    for param in params:
        if param in config.keys():
            for key, value in config[param].items():
                if hasattr(args, key):
                    setattr(args, key, value)

    return args
