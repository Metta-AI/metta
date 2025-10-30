# Repository Guidelines

## Project Structure & Module Organization

- Core Nim: `tribal_village.nim` (entry), `src/` for modules like `environment.nim`, `ai.nim`, `renderer.nim`,
  `tribal_village_interface.nim`.
- Python wrapper: `tribal_village_env/` with `environment.py` loading the shared library.
- Assets/data: `data/`.
- Packaging/build: `tribal_village.nimble`, `pyproject.toml`, `setup.py`, `MANIFEST.in`.

## Build, Test, and Development Commands

- Install Nim deps: `nimble install`
- Run standalone game: `nim r -d:release tribal_village.nim`
- Build shared lib for Python: `nimble buildLib` (creates `libtribal_village.{so|dylib|dll}` in repo root). Ensure it is
  available at `tribal_village_env/libtribal_village.so` (rename/symlink on macOS):
  `ln -sf libtribal_village.dylib tribal_village_env/libtribal_village.so`.
- Quick Python smoke test: `python -c "from tribal_village_env import TribalVillageEnv; TribalVillageEnv()"`
- Editable install (after building the lib): `pip install -e .`

## Coding Style & Naming Conventions

- Nim: 2-space indent; modules `snake_case.nim`; procs/vars `lowerCamelCase`; types/consts `PascalCase`; export with
  trailing `*`. Prefer small, focused procs; avoid global state.
- Python: PEP 8 + type hints; modules `snake_case.py`; classes `PascalCase`; functions `snake_case`.
- Formatting: run `nimpretty src` for Nim. Use `black` for Python if available.

## Testing Guidelines

- No formal test suite yet. Add smoke tests before PRs:
  - Nim: run `nim r tribal_village.nim` and verify basic interaction.
  - Python: instantiate `TribalVillageEnv()` and run a few `step`s.
- If adding Python tests, place under `tests/` as `test_*.py` (pytest style).

## Commit & Pull Request Guidelines

- Commits: short, imperative subject (≤72 chars), optional scope, meaningful body when changing behavior or performance.
- PRs: include purpose, key changes, perf/behavior notes, and reproduction steps. Link issues. Add screenshots for UI
  changes or brief metrics for performance work.
- Keep diffs surgical; avoid unrelated refactors.

## Security & Configuration Tips

- The Python wrapper loads `tribal_village_env/libtribal_village.so`; ensure the library exists and matches your
  platform.
- Requires Nim 2.2.4+ and OpenGL for rendering. Python: 3.9+ with `numpy`, `gymnasium`, `pufferlib`.
