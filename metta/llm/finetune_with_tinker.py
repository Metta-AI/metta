"""Fine-tune LLM on MettaGrid using Tinker."""

import asyncio
import json
from pathlib import Path

import tinker


class MettagridTinkerTrainer:
    """Trainer for MettaGrid using Tinker API."""

    def __init__(
        self,
        dataset_path: str,
        model_name: str = "meta-llama/Llama-3.2-1B",
        lora_rank: int = 32,
        learning_rate: float = 2e-4,
        batch_size: int = 128,
        max_length: int = 2048,  # MettaGrid obs shouldn't need full 32k
        num_epochs: int = 1,
    ):
        self.dataset_path = Path(dataset_path)
        self.model_name = model_name
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_epochs = num_epochs

        # Initialize Tinker clients
        self.service_client = tinker.ServiceClient()
        self.training_client = None

    def load_dataset(self):
        """Load JSONL dataset."""
        with open(self.dataset_path) as f:
            return [json.loads(line) for line in f]

    async def train(self):
        """Run async training loop."""
        # 1. Create LoRA training client
        self.training_client = self.service_client.create_lora_training_client(
            base_model=self.model_name,
            rank=self.lora_rank,
        )

        # 2. Load dataset
        dataset = self.load_dataset()
        print(f"Loaded {len(dataset)} training examples")

        # 3. Training loop
        num_steps = (len(dataset) // self.batch_size) * self.num_epochs
        print(f"Training for {num_steps} steps...")

        for epoch in range(self.num_epochs):
            for step, batch_start in enumerate(range(0, len(dataset), self.batch_size)):
                batch_data = dataset[batch_start : batch_start + self.batch_size]

                # Create Adam params (beta1=0.9, beta2=0.95, eps=1e-8 are common)
                adam_params = tinker.AdamParams(
                    learning_rate=self.learning_rate,
                    beta1=0.9,
                    beta2=0.95,
                    eps=1e-8,
                )

                # Submit forward-backward pass
                fwd_bwd_future = await self.training_client.forward_backward_async(
                    data=batch_data,
                    loss_fn="cross_entropy",
                )

                # Submit optimizer step
                optim_step_future = await self.training_client.optim_step_async(adam_params)

                # Await results
                fwd_bwd_result = await fwd_bwd_future.result_async()
                await optim_step_future.result_async()

                # Log progress
                if step % 10 == 0:
                    # Extract loss from results
                    logprobs = [x["logprobs"] for x in fwd_bwd_result.loss_fn_outputs]
                    avg_loss = sum(logprobs) / len(logprobs) if logprobs else 0
                    print(f"Epoch {epoch}, Step {step}/{num_steps}: loss={avg_loss:.4f}")

                # Save checkpoint periodically
                if step % 100 == 0 and step > 0:
                    await self.training_client.save_state(f"checkpoint_step_{step}")

        # 4. Save final model and get sampling client
        print("Training complete! Saving model...")
        sampling_client = self.training_client.save_weights_and_get_sampling_client(name="mettagrid_llm_v1")

        print(f"Model saved! Path: {sampling_client.model_path}")
        return sampling_client


def main():
    """Entry point for training."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    trainer = MettagridTinkerTrainer(
        dataset_path=args.dataset,
        model_name=args.model,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
    )

    # Run async training
    asyncio.run(trainer.train())


if __name__ == "__main__":
    main()
