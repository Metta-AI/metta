#!/usr/bin/env python
"""
Rewrite `from foo import bar` style imports to `import foo` and update usage to `foo.bar`.

The script walks all tracked Python files in the repository, replaces qualifying
`ImportFrom` statements, and rewrites references to the imported symbols so that
they access the attribute through the module namespace instead.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import subprocess
import sys
import typing

import libcst as cst
import libcst.metadata
import libcst.metadata.scope_provider

SKIP_DIR_NAMES: typing.Set[str] = {"src", "python", "tests"}


@dataclasses.dataclass(frozen=True)
class AliasInfo:
    import_node: cst.ImportFrom
    attr_parts: typing.Tuple[str, ...]
    module_parts: typing.Tuple[str, ...]
    original_name: str


def iter_tracked_python_files(root: pathlib.Path) -> typing.Iterable[pathlib.Path]:
    result = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        yield root / line


def has_init_marker(directory: pathlib.Path) -> bool:
    return (directory / "__init__.py").exists()


def compute_module_parts(
    file_path: pathlib.Path, root: pathlib.Path
) -> typing.Tuple[typing.Tuple[str, ...], typing.Tuple[str, ...]]:
    file_path = file_path.resolve()
    root = root.resolve()
    if not file_path.is_file():
        raise ValueError(f"{file_path} is not a file.")
    if file_path.suffix != ".py":
        raise ValueError(f"{file_path} is not a Python source file.")

    rel_parts = list(file_path.relative_to(root).parts)
    if not rel_parts:
        raise ValueError(f"Could not infer module path for {file_path}")

    is_init = rel_parts[-1] == "__init__.py"
    if is_init:
        rel_parts = rel_parts[:-1]
    else:
        rel_parts[-1] = pathlib.Path(rel_parts[-1]).stem

    if "src" in rel_parts:
        last_src_index = len(rel_parts) - 1 - rel_parts[::-1].index("src")
        rel_parts = rel_parts[last_src_index + 1 :]

    while rel_parts and not rel_parts[0].isidentifier():
        rel_parts = rel_parts[1:]

    module_parts = tuple(rel_parts)
    if not module_parts:
        raise ValueError(f"Could not infer module path for {file_path}")

    package_parts = module_parts if is_init else module_parts[:-1]
    return module_parts, package_parts


def attr_node_to_parts(node: typing.Optional[cst.BaseExpression]) -> typing.List[str]:
    if node is None:
        return []
    if isinstance(node, cst.Name):
        return [node.value]
    if isinstance(node, cst.Attribute):
        return attr_node_to_parts(node.value) + [node.attr.value]
    raise ValueError(f"Unsupported module node type: {type(node)}")


def build_attribute(parts: typing.Sequence[str]) -> cst.BaseExpression:
    if not parts:
        raise ValueError("Cannot build attribute for empty parts.")
    expr: cst.BaseExpression = cst.Name(parts[0])
    for part in parts[1:]:
        expr = cst.Attribute(expr, cst.Name(part))
    return expr


class FromImportTransformer(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (
        libcst.metadata.ScopeProvider,
        libcst.metadata.ExpressionContextProvider,
        libcst.metadata.ParentNodeProvider,
    )

    def __init__(self, module_parts: typing.Tuple[str, ...], package_parts: typing.Tuple[str, ...]):
        self.module_parts = module_parts
        self.package_parts = package_parts
        self.alias_map: typing.Dict[str, typing.List[AliasInfo]] = {}
        self.existing_imports: typing.Set[str] = set()
        self.added_imports: typing.Set[str] = set()
        self.changed: bool = False

    def visit_Import(self, node: cst.Import) -> typing.Optional[bool]:
        for alias in node.names:
            module_str = ".".join(attr_node_to_parts(alias.name))
            if module_str:
                self.existing_imports.add(module_str)
        return True

    def leave_ImportFrom(
        self, original_node: cst.ImportFrom, updated_node: cst.ImportFrom
    ) -> typing.Optional[cst.BaseStatement]:
        if isinstance(original_node.names, cst.ImportStar):
            return updated_node

        module_parts = self._resolve_module_parts(original_node)
        module_str = ".".join(module_parts)
        if not module_str:
            return updated_node
        if module_parts == ("__future__",):
            return updated_node

        for alias in original_node.names:
            original_name = self._extract_alias_name(alias)
            alias_name = self._extract_binding_name(alias)
            attr_parts = tuple(list(module_parts) + [original_name])
            info = AliasInfo(
                import_node=original_node,
                attr_parts=attr_parts,
                module_parts=module_parts,
                original_name=original_name,
            )
            self.alias_map.setdefault(alias_name, []).append(info)

        self.changed = True
        replacement_stmts: typing.List[cst.BaseSmallStatement] = []
        needs_import = module_str not in self.existing_imports and module_str not in self.added_imports
        if needs_import:
            self.added_imports.add(module_str)
            replacement_stmts.append(
                cst.Import(
                    names=[
                        cst.ImportAlias(name=build_attribute(module_parts)),
                    ]
                )
            )

        if self._inside_type_checking(original_node):
            for alias in original_node.names:
                binding_name = self._extract_binding_name(alias)
                original_name = self._extract_alias_name(alias)
                value = build_attribute(tuple(list(module_parts) + [original_name]))
                assign = cst.Assign(
                    targets=[cst.AssignTarget(target=cst.Name(binding_name))],
                    value=value,
                )
                replacement_stmts.append(assign)

        if replacement_stmts:
            return cst.FlattenSentinel(replacement_stmts)
        return cst.RemoveFromParent()

    def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
        alias_infos = self.alias_map.get(original_node.value)
        if not alias_infos:
            return updated_node

        try:
            ctx = self.get_metadata(libcst.metadata.ExpressionContextProvider, original_node)
            scope = self.get_metadata(libcst.metadata.ScopeProvider, original_node)
        except KeyError:
            return updated_node

        if ctx is not libcst.metadata.ExpressionContext.LOAD:
            return updated_node

        assignments = self._lookup_assignments(scope, original_node.value)
        if not assignments:
            return updated_node

        for info in alias_infos:
            if any(
                isinstance(assign, libcst.metadata.scope_provider.ImportAssignment) and assign.node is info.import_node
                for assign in assignments
            ):
                self.changed = True
                return build_attribute(info.attr_parts)
        return updated_node

    def _extract_alias_name(self, alias: cst.ImportAlias) -> str:
        if isinstance(alias.name, cst.Name):
            return alias.name.value
        raise ValueError("Expected ImportAlias name to be a simple Name.")

    def _extract_binding_name(self, alias: cst.ImportAlias) -> str:
        if alias.asname:
            return alias.asname.name.value
        if isinstance(alias.name, cst.Name):
            return alias.name.value
        raise ValueError("Expected ImportAlias name to be a simple Name.")

    def _resolve_module_parts(self, node: cst.ImportFrom) -> typing.Tuple[str, ...]:
        module_parts = attr_node_to_parts(node.module)
        level = len(node.relative)
        if level == 0:
            return tuple(module_parts)

        if not self.package_parts and level > 0:
            raise ValueError("Relative import encountered outside a package.")

        if level - 1 > len(self.package_parts):
            raise ValueError("Relative import level exceeds available package depth.")

        cutoff = len(self.package_parts) - (level - 1) if level > 0 else len(self.package_parts)
        base = list(self.package_parts[:cutoff])
        return tuple(base + module_parts)

    def _lookup_assignments(self, scope: libcst.metadata.scope_provider.Scope, name: str) -> typing.List[object]:
        assignments: typing.List[object] = []
        current: typing.Optional[libcst.metadata.scope_provider.Scope] = scope
        visited: typing.Set[int] = set()
        while current is not None and id(current) not in visited:
            visited.add(id(current))
            try:
                scope_assignments = current.assignments[name]
            except KeyError:
                pass
            else:
                assignments.extend(scope_assignments)
            current = getattr(current, "parent", None)  # type: ignore[assignment]
        return assignments

    def _inside_type_checking(self, node: cst.CSTNode) -> bool:
        parent = self.get_metadata(libcst.metadata.ParentNodeProvider, node, None)
        while parent is not None:
            if isinstance(parent, cst.If):
                if self._is_type_checking_condition(parent.test):
                    return True
            parent = self.get_metadata(libcst.metadata.ParentNodeProvider, parent, None)
        return False

    def _is_type_checking_condition(self, test: cst.BaseExpression) -> bool:
        if isinstance(test, cst.Name):
            return test.value == "TYPE_CHECKING"
        if isinstance(test, cst.Attribute):
            return (
                isinstance(test.value, cst.Name) and test.value.value == "typing" and test.attr.value == "TYPE_CHECKING"
            )
        return False


def rewrite_file(path: pathlib.Path, root: pathlib.Path) -> bool:
    module_parts, package_parts = compute_module_parts(path, root)
    source = path.read_text()
    module = cst.parse_module(source)
    wrapper = libcst.metadata.MetadataWrapper(module)
    transformer = FromImportTransformer(module_parts, package_parts)
    new_module = wrapper.visit(transformer)
    if transformer.changed:
        path.write_text(new_module.code)
        return True
    return False


def main(argv: typing.Optional[typing.Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Rewrite from-imports to module imports.")
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("."),
        help="Repository root. Defaults to current directory.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=pathlib.Path,
        help="Optional specific files or directories to rewrite. Defaults to all tracked Python files.",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.paths:
        targets: typing.List[pathlib.Path] = []
        for path in args.paths:
            resolved = (root / path).resolve() if not path.is_absolute() else path.resolve()
            if resolved.is_dir():
                targets.extend(resolved.rglob("*.py"))
            else:
                targets.append(resolved)
        files = sorted({path.resolve() for path in targets if path.suffix == ".py"})
    else:
        files = list(iter_tracked_python_files(root))

    changed_files = []
    for file_path in files:
        try:
            if rewrite_file(file_path, root):
                changed_files.append(file_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] Failed to rewrite {file_path}: {exc}", file=sys.stderr)
    print(f"Rewritten {len(changed_files)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
