pip install -r requirements.txt

mkdir deps
cd deps

git clone https://github.com/Metta-AI/fast_gae.git
cd fast_gae
python setup.py build_ext --inplace
pip install -e .
cd ..

git clone https://github.com/Metta-AI/puffergrid.git
cd puffergrid
git pull
python setup.py build_ext --inplace
pip install -e .
cd ..

git clone https://github.com/Metta-AI/pufferlib.git
cd pufferlib
git fetch
git stash
git checkout metta
git pull
python setup.py build_ext --inplace
pip install -e .
cd ..

git clone https://github.com/Metta-AI/mettagrid.git
cd mettagrid
git pull
pip install -r requirements.txt
python setup.py build_ext --inplace
pip install -e .
cd ..

#git clone https://github.com/imbue-ai/carbs.git
git clone https://github.com/kywch/carbs.git
cd carbs
git pull
pip install -e .
cd ..

git clone https://github.com/Metta-AI/wandb_carbs.git
cd wandb_carbs
git pull
pip install -e .
cd ..

git clone https://github.com/Metta-AI/sample-factory.git
cd sample-factory
git pull
pip install -e .
cd ..
