let
  pkgs = import (fetchTarball {
    url = "https://nixos.org/channels/nixos-25.05/nixexprs.tar.xz";
  }) {
    config.allowUnfree = true;
  };

  flake-compat = import (fetchTarball {
    url = "https://github.com/edolstra/flake-compat/archive/master.tar.gz";
  });
  nixpkgs-python = (flake-compat {
    src = fetchTarball "https://github.com/cachix/nixpkgs-python/archive/master.tar.gz";
  }).defaultNix;
  mettaPython = nixpkgs-python.packages.x86_64-linux."3.11.7";
in
pkgs.mkShell {
  name = "metta-dev";

  buildInputs = with pkgs; [
    mettaPython
    uv
    cmake
    stdenv.cc.cc.lib

    # for mettascope
    nodejs_24
    typescript
  ];

  shellHook = ''
    # Prevent uv from downloading its own Python
    export UV_PYTHON="${mettaPython}/bin/python3.11"
    # Clear PYTHONPATH to avoid conflicts
    export PYTHONPATH=""

    # set LD_LIBRARY_PATH for cmake to run properly during uv sync
    export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH"

    # Create and activate a virtual environment with uv
    uv sync
    source .venv/bin/activate

    # build frontend
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
}