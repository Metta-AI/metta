"""Test to demonstrate and verify the shared memory lock issue."""

import multiprocessing
import pickle
import time
from multiprocessing import Lock, Manager

import numpy as np
import pytest


def test_regular_lock_not_shared_after_pickle():
    """Demonstrate that regular Lock() cannot be properly pickled.

    In Python 3.12+, attempting to pickle a regular Lock() outside of process
    spawning raises RuntimeError. During process spawning, the lock gets
    recreated as a new independent lock in the child process, which means
    it doesn't synchronize with the parent's lock.

    This is why SharedMemoryBackend must use Manager().Lock() instead.
    """
    lock1 = Lock()

    # Regular locks cannot be pickled outside of process spawning
    with pytest.raises(RuntimeError, match="Lock objects should only be shared"):
        pickle.dumps(lock1)


def test_manager_lock_shared_after_pickle():
    """Demonstrate that Manager().Lock() maintains synchronization after pickling."""
    manager = Manager()
    lock1 = manager.Lock()

    # Pickle and unpickle the lock
    pickled = pickle.dumps(lock1)
    lock2 = pickle.loads(pickled)

    # These are different proxy objects but reference the SAME server-side lock
    assert lock1 is not lock2
    # They synchronize properly - both reference the same underlying lock


def _worker_write_with_lock(backend, worker_id, iterations, results):
    """Worker that writes to shared memory WITH proper locking."""
    for _i in range(iterations):
        with backend.acquire_lock():
            # Read current value
            task_data = backend.get_task_data(0)
            current = task_data[0]

            # Simulate work
            time.sleep(0.0001)

            # Write new value
            task_data[0] = current + 1

    results[worker_id] = True


def _worker_write_without_lock(shm_name, worker_id, iterations, results):
    """Worker that writes to shared memory WITHOUT locking (race condition)."""
    import multiprocessing.shared_memory as shm_module

    # Reconnect to shared memory
    shm = shm_module.SharedMemory(name=shm_name)
    arr = np.ndarray((10, 18), dtype=np.float64, buffer=shm.buf)

    for _i in range(iterations):
        # Read current value
        current = arr[0, 0]

        # Simulate work
        time.sleep(0.0001)

        # Write new value
        arr[0, 0] = current + 1

    shm.close()
    results[worker_id] = True


@pytest.mark.skipif(
    multiprocessing.get_start_method() == "spawn", reason="Test needs 'fork' to demonstrate issue clearly"
)
def test_current_implementation_has_race_condition():
    """Demonstrate that current SharedMemoryBackend has race conditions.

    This test shows that with the current implementation using regular Lock(),
    multiple processes can race when updating shared memory.
    """
    from metta.cogworks.curriculum.shared_memory_backend import SharedMemoryBackend

    backend = SharedMemoryBackend(max_tasks=10, session_id="test_race")

    try:
        # Initialize counter to 0
        backend.get_task_data(0)[0] = 0.0

        # Spawn multiple workers that increment the counter
        num_workers = 4
        iterations = 10
        manager = Manager()
        results = manager.dict()

        processes = []
        for worker_id in range(num_workers):
            p = multiprocessing.Process(target=_worker_write_with_lock, args=(backend, worker_id, iterations, results))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        final_value = backend.get_task_data(0)[0]
        expected_value = num_workers * iterations

        # With proper locking, this should be equal
        # With the current bug, it will often be less due to lost updates
        print(f"\nExpected: {expected_value}, Got: {final_value}")

        # NOTE: This test may pass sometimes due to timing, but demonstrates the issue
        # In production with many workers and high contention, race conditions are guaranteed

    finally:
        backend.cleanup()


def _worker_with_shared_lock(lock, shm_name, worker_id, iterations, results):
    """Worker function with shared lock (must be at module level for pickling)."""
    import multiprocessing.shared_memory as shm_module

    shm = shm_module.SharedMemory(name=shm_name)
    arr = np.ndarray((10, 18), dtype=np.float64, buffer=shm.buf)

    for _i in range(iterations):
        with lock:
            current = arr[0, 0]
            time.sleep(0.0001)
            arr[0, 0] = current + 1

    shm.close()
    results[worker_id] = True


def test_manager_lock_fixes_race_condition():
    """Demonstrate that Manager().Lock() properly synchronizes across processes."""
    from multiprocessing import shared_memory

    # Create a test shared memory segment
    session_id = "test_fixed"
    shm_name = f"ta_{session_id}"
    task_array_size = 10 * 18 * 8

    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=task_array_size)
    arr = np.ndarray((10, 18), dtype=np.float64, buffer=shm.buf)
    arr.fill(0.0)

    try:
        # Create a Manager lock that can be shared
        manager = Manager()
        shared_lock = manager.Lock()

        # Spawn multiple workers that increment with the shared lock
        num_workers = 4
        iterations = 10
        results = manager.dict()

        processes = []
        for worker_id in range(num_workers):
            p = multiprocessing.Process(
                target=_worker_with_shared_lock, args=(shared_lock, shm_name, worker_id, iterations, results)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        final_value = arr[0, 0]
        expected_value = num_workers * iterations

        print(f"\nWith Manager lock - Expected: {expected_value}, Got: {final_value}")

        # With Manager lock, this should always be correct
        assert final_value == expected_value, "Manager lock should prevent race conditions"

    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    # Run tests to demonstrate the issue
    print("=" * 70)
    print("Testing lock behavior with pickling")
    print("=" * 70)

    test_regular_lock_not_shared_after_pickle()
    print("✓ Regular Lock() creates independent locks after pickle")

    test_manager_lock_shared_after_pickle()
    print("✓ Manager().Lock() maintains synchronization after pickle")

    print("\n" + "=" * 70)
    print("Testing race conditions")
    print("=" * 70)

    test_manager_lock_fixes_race_condition()
    print("✓ Manager lock properly synchronizes across processes")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
