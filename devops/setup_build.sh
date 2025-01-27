pip install -r requirements.txt

mkdir deps
cd deps

echo "Cloning fast_gae in to $(pwd)"
git clone https://github.com/Metta-AI/fast_gae.git
cd fast_gae
git pull
echo "Building fast_gae in to $(pwd)"
python setup.py build_ext --inplace
echo "Installing fast_gae in to $(pwd)"
pip install -e .
cd ..

echo "Cloning pufferlib in to $(pwd)"
git clone https://github.com/Metta-AI/pufferlib.git
cd pufferlib
echo "Fetching pufferlib in to $(pwd)"
git fetch
echo "Checking out metta in to $(pwd)"
git checkout metta
# Install numpy first to ensure headers are available
pip install numpy
# Set CFLAGS to include numpy headers
export CFLAGS="$CFLAGS $(python -c 'import numpy as np; print(f"-I{np.get_include()}")')"
echo "Building pufferlib in to $(pwd)"
python setup.py build_ext --inplace
echo "Installing pufferlib in to $(pwd)"
pip install -e .
echo "Stashing pufferlib in to $(pwd)"
git stash
cd ..

echo "Cloning mettagrid in to $(pwd)"
git clone https://github.com/Metta-AI/mettagrid.git
cd mettagrid
echo "Fetching mettagrid in to $(pwd)"
git fetch
echo "Checking out main in to $(pwd)"
git checkout main
echo "Installing mettagrid in to $(pwd)"
pip install -r requirements.txt
echo "Building mettagrid in to $(pwd)"
python setup.py build_ext --inplace
echo "Installing mettagrid in to $(pwd)"
pip install -e .
cd ..

#git clone https://github.com/imbue-ai/carbs.git
git clone https://github.com/kywch/carbs.git
cd carbs
echo "Fetching carbs in to $(pwd)"
git pull
echo "Installing carbs in to $(pwd)"
pip install -e .
cd ..

git clone https://github.com/Metta-AI/wandb_carbs.git
cd wandb_carbs
echo "Fetching wandb_carbs in to $(pwd)"
git pull
echo "Installing wandb_carbs in to $(pwd)"
pip install -e .
cd ..
