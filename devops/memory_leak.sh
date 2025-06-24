#!/bin/bash

# Memory leak simulation script
# This script runs a Python program that allocates 1GB per second

python3 << 'EOF'
import time
import psutil
import os
import signal
import sys
from datetime import datetime

class MemoryLeakSimulator:
    def __init__(self):
        self.allocated_chunks = []
        self.total_allocated_gb = 0
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

    def get_memory_usage_mb(self):
        """Get current process memory usage in MB"""
        try:
            return round(self.process.memory_info().rss / 1024 / 1024, 1)
        except:
            return "N/A"

    def allocate_1gb(self):
        """Allocate approximately 1GB of memory"""
        # Allocate 1GB as a bytearray filled with zeros
        # Using bytearray is more memory-efficient than string multiplication
        chunk = bytearray(1024 * 1024 * 1024)  # 1GB
        self.allocated_chunks.append(chunk)
        self.total_allocated_gb += 1

    def cleanup(self, signum=None, frame=None):
        """Cleanup function for graceful exit"""
        print(f"\n\nCleaning up and exiting...")
        print(f"Total memory allocated: {self.total_allocated_gb}GB")
        print(f"Number of chunks: {len(self.allocated_chunks)}")
        print(f"Final memory usage: {self.get_memory_usage_mb()}MB")
        sys.exit(0)

    def run(self):
        """Main simulation loop"""
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.cleanup)
        signal.signal(signal.SIGTERM, self.cleanup)

        print("Starting memory leak simulation...")
        print("Allocating 1GB per second. Press Ctrl+C to stop.")
        print("Time | Total Allocated | Process Memory Usage")
        print("-----|-----------------|---------------------")

        try:
            while True:
                current_time = time.time()
                elapsed = int(current_time - self.start_time)

                # Allocate 1GB
                print(f"Allocating 1GB... ", end="", flush=True)
                start_alloc = time.time()
                self.allocate_1gb()
                alloc_time = time.time() - start_alloc

                # Get current memory usage
                current_memory = self.get_memory_usage_mb()

                # Log the status
                print(f"{elapsed:4d}s | {self.total_allocated_gb:13d}GB | {current_memory:>15}MB (alloc: {alloc_time:.2f}s)")

                # Wait for the remainder of the second
                sleep_time = max(0, 1.0 - alloc_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            self.cleanup()
        except MemoryError:
            print(f"\nMemoryError: Unable to allocate more memory!")
            print(f"Successfully allocated {self.total_allocated_gb}GB before hitting limit")
            self.cleanup()
        except Exception as e:
            print(f"\nError: {e}")
            self.cleanup()

if __name__ == "__main__":
    simulator = MemoryLeakSimulator()
    simulator.run()
EOF
