let
  pkgs = import (fetchTarball {
    url = "https://nixos.org/channels/nixos-24.11/nixexprs.tar.xz";
    sha256 = "1a6qqh62vcl61x07s26jxrqjq725v49qrg4w5bzwk1mh73sx6iy9";
  }) {
    config.allowUnfree = true;
  };

  flake-compat = import (fetchTarball {
    url = "https://github.com/edolstra/flake-compat/archive/master.tar.gz";
    sha256 = "09m84vsz1py50giyfpx0fpc7a4i0r1xsb54dh0dpdg308lp4p188";
  });
  nixpkgs-python = (flake-compat {
    src = fetchTarball {
      url = "https://github.com/cachix/nixpkgs-python/archive/master.tar.gz";
      sha256 = "12pdyv8pf99jdp7aw7x1qgd6ralg1j6dd6k79cr1xbfmb2fz10lj";
    };
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

    # nix 24.11 comes with gcc13, which works with nvcc 12

    # for mettascope
    nodejs_22
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
