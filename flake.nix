{
  description = "Metta development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
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

  outputs = { self, nixpkgs, nixpkgs-python, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
      mettaPython = nixpkgs-python.packages.${system}."3.11.7";
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "metta-dev";

        buildInputs = with pkgs; [
          mettaPython
          uv
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
          libGL
          curl
        ];

        shellHook = ''
          # Prevent uv from downloading its own Python
          export UV_PYTHON="${mettaPython}/bin/python3.11"
          # Clear PYTHONPATH to avoid conflicts
          export PYTHONPATH=""

          # Set LD_LIBRARY_PATH for bazel to run properly during uv sync
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

          # Provide X/GL shared libraries for mettascope2.
          export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.xorg.libX11 pkgs.xorg.libXext pkgs.libGL pkgs.curl ]}:$LD_LIBRARY_PATH"

          # Use a writable cache for Emscripten.
          # Emscripten default cache points to the read-only nix store, which does not work.
          export EM_CACHE="$HOME/.cache/emscripten"
          mkdir -p "$EM_CACHE"

          # Create and activate a virtual environment with uv
          uv sync
          source .venv/bin/activate

          # Build frontend
          pushd mettascope
          corepack enable
          pnpm install
          tsc
          python tools/gen_atlas.py
          echo "Frontend built"
          popd

          echo "# Python version: $(python --version)"
          echo "# uv version: $(uv --version)"
          echo "# -------------------------------------------"
          echo "# ./tools/train.py run=my_experiment wandb=off"
          echo "# ./tools/sim.py run=my_experiment wandb=off"
          echo "# ./tools/play.py run=my_experiment wandb=off"
          echo "# -------------------------------------------"
        '';
      };
    };
}
