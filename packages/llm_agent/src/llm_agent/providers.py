"""LLM provider utilities for model selection and availability checks."""

import subprocess
import sys

from openai import OpenAI


def check_ollama_available() -> bool:
    try:
        client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        # Try to list models as a health check
        client.models.list()
        return True
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        # Parse output: skip header line, extract model names
        lines = result.stdout.strip().split("\n")[1:]  # Skip header
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return []


def get_openai_models() -> list[tuple[str, str]]:
    return [
        ("gpt-4o-mini", "Cheapest - Fast and cost-effective"),
        ("gpt-4o", "Capable - Best GPT-4 model"),
        ("gpt-5.1", "Advanced - GPT-5 with strong reasoning"),
        ("gpt-5.2", "Best - Latest GPT-5 for complex reasoning"),
    ]


def select_openai_model() -> str:
    models = get_openai_models()

    print("\n" + "=" * 60)
    print("Select OpenAI Model:")
    print("=" * 60)
    for idx, (model_name, description) in enumerate(models, 1):
        print(f"  [{idx}] {model_name}")
        print(f"      {description}")
    print("=" * 60)

    while True:
        try:
            selection = input(f"\nSelect a model (1-{len(models)}): ").strip()
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                model = models[idx][0]
                print(f"\n✓ Selected: {model}\n")
                return model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\n⚠️  No model selected. Exiting.\n")
            sys.exit(0)


def get_anthropic_models() -> list[tuple[str, str]]:
    return [
        ("claude-haiku-4-5", "Cheapest - Fastest with near-frontier intelligence"),
        ("claude-sonnet-4-5", "Best - Smartest for complex agents & coding"),
        ("claude-opus-4-5", "Premium - Maximum intelligence & performance"),
    ]


def select_anthropic_model() -> str:
    models = get_anthropic_models()

    print("\n" + "=" * 60)
    print("Select Claude Model:")
    print("=" * 60)
    for idx, (model_name, description) in enumerate(models, 1):
        print(f"  [{idx}] {model_name}")
        print(f"      {description}")
    print("=" * 60)

    while True:
        try:
            selection = input(f"\nSelect a model (1-{len(models)}): ").strip()
            idx = int(selection) - 1
            if 0 <= idx < len(models):
                model = models[idx][0]
                print(f"\n✓ Selected: {model}\n")
                return model
            else:
                print(f"Please enter a number between 1 and {len(models)}")
        except ValueError:
            print("Please enter a valid number")
        except (KeyboardInterrupt, EOFError):
            print("\n\n⚠️  No model selected. Exiting.\n")
            sys.exit(0)


def ensure_ollama_model(model: str | None = None) -> str:
    if not check_ollama_available():
        raise RuntimeError(
            "Ollama server is not running. Please start it with 'ollama serve' or install from https://ollama.ai"
        )

    available_models = list_ollama_models()

    # If no model specified, prompt user to select
    if model is None:
        if not available_models:
            # No models available, prompt user
            print("\n" + "=" * 60)
            print("⚠️  No Ollama models found!")
            print("=" * 60)
            print("\nOptions:")
            print("  1. Install default model (llama3.2) - ~2GB download")
            print("  2. Install a model manually with 'ollama pull <model>'")
            print("  3. Use llm-anthropic or llm-openai instead")
            print("=" * 60)

            try:
                response = input("\nInstall default model (llama3.2)? [y/N]: ").strip().lower()
                if response in ("y", "yes"):
                    model = "llama3.2"
                    print(f"\n📥 Pulling {model}...")
                    print("(This may take a few minutes...)\n")
                    subprocess.run(["ollama", "pull", model], check=True)
                    print(f"\n✓ Successfully installed {model}\n")
                    return model
                else:
                    print("\n" + "=" * 60)
                    print("To use Ollama:")
                    print("  1. Pull a model: ollama pull llama3.2")
                    print("  2. Run again: cogames play -m <mission> -p llm-ollama")
                    print("\nAlternatively, use cloud LLMs:")
                    print("  • cogames play -m <mission> -p llm-openai")
                    print("  • cogames play -m <mission> -p llm-anthropic")
                    print("=" * 60 + "\n")
                    sys.exit(0)
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠️  Cancelled by user.\n")
                sys.exit(0)

        # Show available models and prompt user to select
        print("\n" + "=" * 60)
        print("Available Ollama Models:")
        print("=" * 60)
        for idx, model_name in enumerate(available_models, 1):
            print(f"  [{idx}] {model_name}")
        print("=" * 60)

        while True:
            try:
                selection = input(f"\nSelect a model (1-{len(available_models)}): ").strip()
                idx = int(selection) - 1
                if 0 <= idx < len(available_models):
                    model = available_models[idx]
                    print(f"\n✓ Selected: {model}\n")
                    return model
                else:
                    print(f"Please enter a number between 1 and {len(available_models)}")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\n\n⚠️  No model selected. Exiting.\n")
                sys.exit(0)

    # Model was explicitly specified, check if it's available
    if any(model in m for m in available_models):
        return model

    # Try to pull the specified model
    print(f"\nModel '{model}' not found. Pulling from Ollama...")
    try:
        subprocess.run(
            ["ollama", "pull", model],
            check=True,
            capture_output=False,  # Show progress
        )
        print(f"\n✓ Successfully pulled model: {model}\n")
        return model
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to pull Ollama model '{model}': {e}") from e
