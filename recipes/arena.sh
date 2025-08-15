for i in 1 2 3 4 5; do
  seed=$(( $(od -An -N4 -t u4 /dev/urandom) % 1073741824 ))
  ./devops/skypilot/launch.py train \
    --gpus=4 \
    --nodes=1 \
    --no-spot \
    run=$USER.arena_recipe_muon_on_1x4_no_contrastive.$(date +%m-%d-%H-%M).s${seed} \
    ++trainer.curriculum=/env/mettagrid/curriculum/arena/learning_progress \
    ++trainer.optimizer.learning_rate=0.0045 \
    ++trainer.optimizer.type=muon \
    ++trainer.simulation.evaluate_interval=50 \
    ++trainer.contrastive.enabled=false\
    "$@"
done

