import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.map.train_vae import VAE, add_pretty_border, tensor_to_ascii_lines


def main():
    parser = argparse.ArgumentParser(description="Sample maps from trained VAE.")
    parser.add_argument("--model_path", type=str, default="vae.pth", help="Path to the trained model.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    height = 40
    width = 40

    model = VAE(input_height=height, input_width=width).to(device)

    checkpoint = torch.load(args.model_path, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        # For backward compatibility
        model.load_state_dict(checkpoint)

    model.eval()

    samples = model.sample(args.num_samples, device)  # type: ignore

    for i, sample in enumerate(samples):
        sample_lines = tensor_to_ascii_lines(sample.cpu())
        bordered = add_pretty_border(sample_lines)
        print(f"Sample {i + 1}:")
        for line in bordered:
            print(line)
        print("-" * 40)


if __name__ == "__main__":
    main()
