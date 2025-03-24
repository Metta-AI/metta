if [ -z "$SKIP_BUILD" ] || [ "$SKIP_BUILD" = "0" ]; then
    git pull
    # cd deps/pufferlib && git pull && python setup.py build_ext --inplace && cd ../..
    cd deps/fast_gae && git pull && python setup.py build_ext --inplace && cd ../..
    cd deps/mettagrid && git pull && python setup.py build_ext --inplace && cd ../..
    cd deps/wandb_carbs && git pull && cd ../..
fi
