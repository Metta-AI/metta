{
  description = "Metta development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
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
          cmake
          stdenv.cc.cc.lib
          nodejs_22
          typescript
        ];

        shellHook = ''
          # Prevent uv from downloading its own Python
          export UV_PYTHON="${mettaPython}/bin/python3.11"
          # Clear PYTHONPATH to avoid conflicts
          export PYTHONPATH=""

          # Set LD_LIBRARY_PATH for cmake to run properly during uv sync
          export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

          # Create and activate a virtual environment with uv
          uv sync
          source .venv/bin/activate

          # Build frontend
          pushd mettascope
          npm install
          tsc
          python tools/gen_atlas.py
          echo "Frontend built"
          popd

          echo "# Python version: $(python --version)"
          echo "# uv version: $(uv --version)"
          echo "# -------------------------------------------"
          echo "# ./tools/train.py run=my_experiment +hardware=macbook wandb=off"
          echo "# ./tools/sim.py run=my_experiment +hardware=macbook wandb=off"
          echo "# ./tools/play.py run=my_experiment +hardware=macbook wandb=off"
          echo "# -------------------------------------------"
        '';
      };
    };
}
