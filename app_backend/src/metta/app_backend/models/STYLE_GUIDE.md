### SQLModel/SQLAlchemy Async Patterns

**Database session management:**

- Entry point methods use `@with_db` decorator to establish session
- Internal methods called within a session use `get_db()` directly (no decorator)
- Never nest decorators - if caller has `@with_db`, callees just use `get_db()`

**Query syntax:**

- Use `filter_by()` for simple equality: `select(X).filter_by(user_id=uid, active=True)`
- Use `.where()` for complex conditions: `.where(col(X.id).in_(ids))`
- Use `col()` for column operations that the type checker doesn't understand:
  - `.in_()`: `col(X.id).in_(ids)`
  - `.is_not(None)`: `col(X.score).is_not(None)`
  - `.desc()/.asc()`: `col(X.created_at).desc()`
  - `func.filter()`: `func.count().filter(col(X.status) == Status.done)`
- Simple comparisons in `.where()` don't need `col()`: `.where(X.status == Status.active)`
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

**Use relationship-based joins:**

Prefer `.join(Model.relationship)` over explicit `.join(OtherModel, Model.fk == OtherModel.id)`:

```python
# Verbose: explicit join condition
select(Match).join(Pool, Match.pool_id == Pool.id).join(Season, Pool.season_id == Season.id)

# Clean: relationship-based joins
select(Match).join(Match.pool).join(Pool.season)
```

**Prefer one query with eager loading over multiple queries:**

Don't fetch IDs, then fetch related objects, then stitch them together in Python. Use joins and `selectinload` to get
everything in one query and access data through relationships.

```python
# Bad: multiple queries, manual resolution
pools = await get_pools(session, season_name)
players = await get_players(session, list(pools.keys()))
pvs = await get_policy_versions(session, [p.pv_id for p in players])
return [Summary(name=pvs[p.pv_id].policy.name, pool=pools[p.pool_id]) for p in players]

# Good: one query, access through relationships
players = (await session.execute(
    select(PoolPlayer)
    .join(Pool).join(Season).where(Season.name == season_name)
    .options(
        selectinload(PoolPlayer.policy_version).selectinload(PolicyVersion.policy),
        selectinload(PoolPlayer.pool),
    )
)).scalars().all()
return [Summary(name=p.policy_version.policy.name, pool=p.pool.name) for p in players]
```
