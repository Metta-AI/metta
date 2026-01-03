# Style Guide

## Philosophy

Write lean code that will be kept. Favor simplicity over completeness.

## Working Principles

**Write less code.** A smaller change that doesn't fully achieve the goal is better than a larger change that does. The
goal is code that gets kept, not a messy MVP.

**Search before writing.** The codebase likely has something similar. Look for existing implementations first.

**Trust your environment.** Don't add defensive checks for conditions guaranteed by the project structure.

**Self-documenting code.** Clear names over comments. Only comment the "why", not the "what".

## Python Style

- Type hints on all function parameters
- No docstrings unless behavior is non-obvious
- Absolute imports only (`from metta.x import Y`, not `from .x import Y`)
- Private members start with `_`
- Empty `__init__.py` files (except public packages that need exports)
- `Optional[X]` over `X | None`
- Top-level imports only (no inline `from x import Y` inside functions)
- Use `collections.defaultdict` instead of `dict.setdefault()`

### Imports

```python
from __future__ import annotations  # When needed for forward refs
from metta.common.types import X    # Shared types from types.py
```

Circular import? Extract types to `types.py` or use module import (`import x.y as y_mod`).

### SQLModel/SQLAlchemy Async Patterns

**Database session management:**

- Entry point methods use `@with_db` decorator to establish session
- Internal methods called within a session use `get_db()` directly (no decorator)
- Never nest decorators - if caller has `@with_db`, callees just use `get_db()`

**Query syntax:**

- Use `filter_by()` for simple equality: `select(X).filter_by(user_id=uid, active=True)`
- Use `.where()` for complex conditions: `.where(col(X.id).in_(ids))`
- Combine result and processing in one expression:

```python
# Good: single expression
matches = list(
    (await session.execute(select(Match).filter_by(pool_id=pool_id))).scalars().all()
)

# Avoid: separate result variable
result = await session.execute(select(Match).filter_by(pool_id=pool_id))
matches = list(result.scalars().all())
```

## What Not To Do

- Don't add error handling for impossible cases
- Don't create abstractions for one-time operations
- Don't add comments that restate the code
- Don't add backwards-compatibility shims for unused code
- Don't run lint/tests automatically (too slow)
- Don't push to git remote (humans do that)
