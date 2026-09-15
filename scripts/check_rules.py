"""Check source locations and static Python imports against AGENTS.md."""
import ast
from pathlib import Path
import subprocess
import sys


PYTHON = {".py", ".pyi"}
WEB = {".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".html", ".css"}
SOURCE_ROOTS = {
    "citadel": PYTHON,
    "citadel_client": PYTHON,
    "gui": WEB,
    "scripts": PYTHON | WEB | {".sh"},
    "tests": PYTHON | WEB | {".sh"},
}
INTERNAL = {
    "citadel.domain": ("citadel.domain",),
    "citadel.application": ("citadel.application", "citadel.domain", "citadel.configuration"),
    "citadel.infrastructure": ("citadel.infrastructure", "citadel.domain", "citadel.configuration",
                               "citadel.application.ports"),
    "citadel.server": ("citadel.server", "citadel.application", "citadel.domain", "citadel.configuration"),
    "citadel.configuration": ("citadel.configuration", "citadel.domain"),
}
ENTRY_FILES = {"__init__.py", "__main__.py", "bootstrap.py", "configuration.py"}
IO_MODULES = {"http", "urllib", "socket", "sqlite3", "shelve", "dbm", "subprocess"}


def location_error(path):
    if path.suffix not in SOURCE_ROOTS.get(path.parts[0], set()):
        return "源码位置不符合 AGENTS.md；放入对应的产品、脚本或测试目录"
    if path.parts[0] == "citadel":
        if len(path.parts) == 2:
            if path.name not in ENTRY_FILES:
                return "citadel 根目录只放配置、组装、入口和包初始化；业务代码放入对应分层"
        elif path.parts[1] not in {"domain", "application", "infrastructure", "server"}:
            return "citadel 下的业务代码必须放入 domain/application/infrastructure/server"
    return None


def imports(path, tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node.lineno, alias.name
        elif isinstance(node, ast.ImportFrom):
            package = list(path.parent.parts)
            prefix = package[:len(package) + 1 - node.level] if node.level else []
            base = prefix + (node.module.split(".") if node.module else [])
            for alias in node.names:
                yield node.lineno, ".".join(base + ([] if alias.name == "*" else [alias.name]))


def import_error(path, target):
    parts = target.split(".")
    if parts[0] in {"artifacts", "source_snapshot", "cache", "_cache"} or (
        parts[0] in SOURCE_ROOTS and {"artifacts", "source_snapshot"}.intersection(parts[1:])
    ):
        return "不得导入产物、缓存或历史源码快照"
    source_root = path.parts[0]
    if source_root in {"citadel", "citadel_client"}:
        forbidden = {"scripts", "tests", "citadel_client" if source_root == "citadel" else "citadel"}
        if parts[0] in forbidden:
            return "产品代码不得依赖工具、测试或另一端的实现"
    module = ".".join(path.with_suffix("").parts)
    layer = ".".join(path.parts[:2]) if len(path.parts) > 2 else module
    allowed = INTERNAL.get(layer)
    if parts[0] == "citadel" and allowed is not None:
        if not any(target == item or target.startswith(item + ".") for item in allowed):
            return f"依赖方向违规：{layer} 不得导入 {target}"
    if layer in {"citadel.domain", "citadel.application"}:
        if parts[0] in IO_MODULES or parts[0] not in sys.stdlib_module_names | {"citadel", "pydantic"}:
            return "领域和应用层不得引入网络、存储、媒体或供应商依赖；通过协议注入基础设施"
    return None


def check(root):
    root = Path(root).resolve()
    output = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root,
        text=True, encoding="utf-8")
    errors = []
    for name in sorted(set(output.split("\0")) - {""}):
        path = Path(name)
        source = root / path
        if path.suffix not in PYTHON | WEB | {".sh"}:
            continue
        if not source.exists() and not source.is_symlink():
            continue
        reason = location_error(path)
        if reason:
            errors.append(f"{path}:1: {reason}")
            continue
        if source.is_symlink():
            errors.append(f"{path}:1: 源码必须在约定目录中保存，不得链接到其他位置")
            continue
        if path.suffix not in PYTHON:
            continue
        try:
            tree = ast.parse(source.read_text(encoding="utf-8"), filename=name)
        except SyntaxError as exc:
            errors.append(f"{path}:{exc.lineno}: Python 语法错误：{exc.msg}")
            continue
        for line, target in imports(path, tree):
            reason = import_error(path, target)
            if reason:
                errors.append(f"{path}:{line}: {reason}（{target}）")
    return errors


def main(root=Path(__file__).resolve().parents[1]):
    errors = check(root)
    print("\n".join(errors) if errors else "Repository rules passed.")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
