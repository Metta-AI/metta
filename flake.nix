{
  description = "Metta development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/nixos-unstable";
    nixpkgs-python.url = "github:cachix/nixpkgs-python";
  };

  nixConfig = {
    extra-substituters = [
      "https://nixpkgs-python.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nixpkgs-python.cachix.org-1:hxjI7pFxTyuTHn2NkvWCrAUcNZLNS3ZAvfYNuYifcEU="
    ];
  };

  outputs = { self, nixpkgs, nixpkgs-unstable, nixpkgs-python, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      unstable = import nixpkgs-unstable {
        inherit system;
        config.allowUnfree = true;
      };
      mettaPython = nixpkgs-python.packages.${system}."3.12.11";
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "metta-dev";

        buildInputs = with pkgs; [
          mettaPython
          unstable.uv
          bazel_7
          stdenv.cc.cc.lib
          pnpm
          nodejs_22
          typescript

          nim
          nimble
          emscripten
          xorg.libX11
          xorg.libXext
          xorg.libXcursor
          libGL
          curl
          udev
          libevdev
          zlib
          zstd

          python3Packages.jupyterlab
        ];

        shellHook = ''
          # Prevent uv from downloading its own Python
          export UV_PYTHON="${mettaPython}/bin/python3.12"
          # Clear PYTHONPATH to avoid conflicts
          export PYTHONPATH=""

          # Set LD_LIBRARY_PATH for bazel to run properly during uv sync
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

          # Provide X/GL shared libraries for mettascope.
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.xorg.libX11 pkgs.xorg.libXext pkgs.xorg.libXcursor pkgs.libGL pkgs.curl pkgs.udev pkgs.libevdev pkgs.zlib ]}:$LD_LIBRARY_PATH"

          # Add zstd library path for PyTorch dependencies
          export LD_LIBRARY_PATH="${pkgs.zstd.out}/lib:$LD_LIBRARY_PATH"

          # Use a writable cache for Emscripten.
          # Emscripten default cache points to the read-only nix store, which does not work.
          export EM_CACHE="$HOME/.cache/emscripten"
          mkdir -p "$EM_CACHE"

          # Check for AMD GPU (ROCm support) by looking for /dev/kfd
          if [ -e "/dev/kfd" ]; then
            echo "# AMD GPU detected, configuring PyTorch for ROCm 6.4"
            export FORCE_CUDA=0

            # Create and activate virtual environment with uv
            uv sync
            source .venv/bin/activate

            # Check if ROCm PyTorch is already installed to avoid slow reinstall
            if python -c "import torch; print(torch.__version__)" 2>/dev/null | grep -q "rocm"; then
              echo "# ROCm PyTorch already installed, skipping reinstall"
            else
              echo "# Installing ROCm PyTorch..."
              pip install --force-reinstall --extra-index-url https://download.pytorch.org/whl/rocm6.4 torch==2.9.0+rocm6.4 pytorch-triton-rocm==3.5.0
            fi
          else
            echo "# No AMD GPU detected, using default PyTorch installation"
            # Create and activate a virtual environment with uv
            uv sync
            source .venv/bin/activate
          fi

          echo "# Python version: $(python --version)"
          echo "# uv version: $(uv --version)"
          echo "# -------------------------------------------"
          echo "# ./tools/run.py train arena run=my_experiment evaluator.evaluate_remote=false"
          echo "# ./tools/run.py play arena policy_uri=file://./train-dir/my_experiment/checkpoints/YOUR-CHECKPOINT-HERE"
          echo "# ./tools/run.py replay arena policy_uri=file://./train-dir/my_experiment/checkpoints/YOUR-CHECKPOINT-HERE"
          echo "# -------------------------------------------"
          echo "# Jupyter commands:"
          echo "#   jupyter lab                                    # Start JupyterLab"
          echo "#   jupyter lab notebooks/colab/Cogames_Training.ipynb  # Open specific notebook"
          echo "#   jupyter notebook                               # Start classic Jupyter"
          echo "# -------------------------------------------"
        '';
      };
    };
}
