mkdir deps
cd deps

git clone https://github.com/Metta-AI/fast_gae.git
cd fast_gae
python setup.py build_ext --inplace
pip install -e .
cd ..

git clone https://github.com/Metta-AI/puffergrid.git
cd puffergrid
python setup.py build_ext --inplace
pip install -e .
cd ..

git clone https://github.com/Metta-AI/pufferlib.git
cd puffergrid
git checkout dev
python setup.py build_ext --inplace
pip install -e .
cd ..

git clone https://github.com/Metta-AI/mettagrid.git
cd mettagrid
python setup.py build_ext --inplace
pip install -e .
cd ..

git clone https://github.com/Metta-AI/sample-factory.git
cd sample_factory
pip install -e .
cd ..
