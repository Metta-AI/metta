#!/usr/bin/env python3
"""Fuzzer for the commander function."""

import random
import string

from tools.commander import CommanderError, commander


# Complex test classes
class DatabaseConfig:
    def __init__(self):
        self.host = "localhost"
        self.port = 5432
        self.ssl_enabled = True
        self.timeout = 30.5
        self.pool_size = 10


class ServerConfig:
    def __init__(self):
        self.name = "web-server"
        self.port = 8080
        self.workers = 4
        self.debug = False
        self.database = DatabaseConfig()


class FeatureFlags:
    def __init__(self):
        self.analytics = True
        self.new_ui = False
        self.beta_features = False


class ComplexApp:
    def __init__(self):
        self.server = ServerConfig()
        self.features = FeatureFlags()
        self.allowed_hosts = ["localhost", "127.0.0.1", "example.com"]
        self.metadata = {
            "version": "1.0.0",
            "build": 12345,
            "deployment": {
                "environment": "production",
                "region": "us-west-2",
                "replicas": [
                    {"id": 1, "status": "running"},
                    {"id": 2, "status": "stopped"},
                    {"id": 3, "status": "running"},
                ],
            },
        }
        self.limits = {"cpu": 2.0, "memory": "4GB", "disk": "100GB"}


if __name__ == "__main__":
    print("Running commander fuzzer...\n")

    # Good commands array
    good_commands = [
        "--server.port 9090",
        "--server.database.host db.example.com",
        "--server.database.port 3306",
        "--server.database.ssl_enabled",
        "--features.analytics",
        "--features.new_ui --features.beta_features",
        "--allowed_hosts.0 production.example.com",
        "--metadata.version '2.0.0'",
        '--metadata.deployment.environment staging --metadata.deployment.replicas.1.status "running"',
        "--limits.cpu 4.0 --limits.memory '8GB' --server.workers 8",
    ]

    random.seed(42)

    print("Running 10,000 fuzzing iterations...")

    for i in range(10000):
        # Pick random command and mutation type
        base_command = random.choice(good_commands)
        mutation_type = random.randint(1, 5)
        command = base_command

        # Mutate the command
        if command and mutation_type == 1:  # Add random character
            pos = random.randint(0, len(command))
            char = random.choice(string.ascii_letters + string.digits + string.punctuation + " ")
            command = command[:pos] + char + command[pos:]
        elif command and mutation_type == 2:  # Add random space
            pos = random.randint(0, len(command))
            command = command[:pos] + " " + command[pos:]
        elif command and mutation_type == 3:  # Remove chunk
            if len(command) >= 2:
                start = random.randint(0, len(command) - 1)
                end = random.randint(start, min(start + 10, len(command)))
                command = command[:start] + command[end:]
        elif command and mutation_type == 4:  # Remove from front
            chars_to_remove = min(random.randint(1, 5), len(command))
            command = command[chars_to_remove:]
        elif command and mutation_type == 5:  # Remove from back
            chars_to_remove = min(random.randint(1, 5), len(command))
            command = command[:-chars_to_remove] if chars_to_remove < len(command) else ""

        # Test the mutated command
        test_tree = ComplexApp()
        try:
            commander(command, test_tree)
        except CommanderError:
            pass  # Expected error, continue
        except Exception as e:
            print(f"❌ UNEXPECTED ERROR on iteration {i + 1}: {command}")
            print(f"Error: {type(e).__name__}: {e}")
            exit(1)

        # Progress every 1000 iterations
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i + 1}/10000 iterations completed...")

    print("✅ All 10,000 iterations completed successfully!")
