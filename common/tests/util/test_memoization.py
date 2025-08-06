import asyncio

import pytest

from metta.common.util.memoization import memoize


@pytest.mark.asyncio
async def test_memoize_caches_result():
    call_count = 0

    @memoize(max_age=1)
    async def expensive_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0.001)
        return x * 2

    result1 = await expensive_function(5)
    result2 = await expensive_function(5)

    assert result1 == 10
    assert result2 == 10
    assert call_count == 1


@pytest.mark.asyncio
async def test_memoize_expires_after_max_age():
    call_count = 0

    @memoize(max_age=0.01)
    async def timed_function(x: int) -> int:
        nonlocal call_count
        call_count += 1
        return x + 10

    result1 = await timed_function(3)
    assert result1 == 13
    assert call_count == 1

    await asyncio.sleep(0.02)

    result2 = await timed_function(3)
    assert result2 == 13
    assert call_count == 2
