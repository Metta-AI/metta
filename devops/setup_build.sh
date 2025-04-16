set -e

pip install -r requirements.txt

mkdir -p deps
cd deps

if [ ! -d "fast_gae" ]; then
  echo "Cloning fast_gae into $(pwd)"
  git clone https://github.com/Metta-AI/fast_gae.git
fi
cd fast_gae
git pull
echo "Building fast_gae into $(pwd)"
python setup.py build_ext --inplace
echo "Installing fast_gae into $(pwd)"
pip install -e .
cd ..

if [ ! -d "pufferlib" ]; then
  echo "Cloning pufferlib into $(pwd)"
  git clone https://github.com/Metta-AI/pufferlib.git
fi
cd pufferlib
echo "Fetching pufferlib into $(pwd)"
git fetch
echo "Checking out metta into $(pwd)"
git checkout metta
git pull
echo "Installing pufferlib into $(pwd)"
pip install -e .
echo "Stashing pufferlib into $(pwd)"
git stash
cd ..

if [ ! -d "mettagrid" ]; then
  echo "Cloning mettagrid into $(pwd)"
  git clone https://github.com/Metta-AI/mettagrid.git
fi
cd mettagrid
echo "Fetching mettagrid into $(pwd)"
git fetch

# Check out the specified reference
if [ -n "$METTAGRID_REF" ]; then
  echo "Checking out mettagrid reference: $METTAGRID_REF"
  git checkout "$METTAGRID_REF"
fi

echo "Installing mettagrid into $(pwd)"
pip install -r requirements.txt
echo "Building mettagrid into $(pwd)"
python setup.py build_ext --inplace
echo "Installing mettagrid into $(pwd)"
pip install -e .
cd ..

if [ ! -d "carbs" ]; then
  #git clone https://github.com/imbue-ai/carbs.git
  git clone https://github.com/kywch/carbs.git
fi
cd carbs
echo "Fetching carbs into $(pwd)"
git pull
echo "Installing carbs into $(pwd)"
pip install -e .
cd ..

if [ ! -d "wandb_carbs" ]; then
  git clone https://github.com/Metta-AI/wandb_carbs.git
fi
cd wandb_carbs
echo "Fetching wandb_carbs into $(pwd)"
git pull
echo "Installing wandb_carbs into $(pwd)"
pip install -e .
cd ..
