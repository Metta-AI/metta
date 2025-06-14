#!/usr/bin/env python3
"""
Demo script showing how to use Redis-based distributed locks.
"""

import logging
import multiprocessing
import os
import sys
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from redis_lock import get_redis_client, redis_lock

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(processName)s - %(message)s")
logger = logging.getLogger(__name__)


def worker_task(worker_id: int):
    """Simulates a worker trying to access a shared resource."""
    logger.info(f"Worker {worker_id} starting")

    # Try to acquire lock for shared resource
    try:
        # redis_lock will automatically load from redis.env
        with redis_lock("shared_resource", timeout=30, retry_interval=0.5, max_retries=20):
            logger.info(f"Worker {worker_id} acquired lock!")

            # Simulate some work
            for i in range(5):
                logger.info(f"Worker {worker_id} working... {i + 1}/5")
                time.sleep(1)

            logger.info(f"Worker {worker_id} finished work")

    except TimeoutError:
        logger.error(f"Worker {worker_id} couldn't acquire lock")
    except Exception as e:
        logger.error(f"Worker {worker_id} error: {e}")


def test_redis_connection():
    """Test Redis connection and basic operations."""
    logger.info("Testing Redis connection...")

    try:
        # This will automatically load from redis.env
        client = get_redis_client()

        # Test connection
        client.ping()
        logger.info("✓ Redis connection successful")

        # Test set/get
        test_key = "test:connection"
        test_value = f"Connected at {time.time()}"
        client.set(test_key, test_value, ex=60)
        retrieved = client.get(test_key)
        logger.info(f"✓ Set/Get test successful: {retrieved}")

        # Clean up
        client.delete(test_key)

        return True

    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        logger.error("Check that redis.env exists and contains valid credentials")
        return False


def main():
    """Run the demo."""
    # Test connection first
    if not test_redis_connection():
        sys.exit(1)

    logger.info("\nStarting distributed lock demo...")
    logger.info("Launching 4 workers that will compete for the same lock")

    # Create multiple processes that will compete for the lock
    processes = []
    for i in range(4):
        p = multiprocessing.Process(target=worker_task, args=(i,), name=f"Worker-{i}")
        p.start()
        processes.append(p)
        time.sleep(0.1)  # Slight delay between starts

    # Wait for all processes to complete
    for p in processes:
        p.join()

    logger.info("\nDemo complete!")

    # Show final state
    client = get_redis_client()
    lock_key = "lock:shared_resource"
    current_holder = client.get(lock_key)
    if current_holder:
        logger.info(f"Lock still held by: {current_holder}")
        ttl = client.ttl(lock_key)
        logger.info(f"Lock expires in: {ttl} seconds")
    else:
        logger.info("Lock is free")


if __name__ == "__main__":
    main()
