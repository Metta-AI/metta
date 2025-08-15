for i in 1 2 3 4 5; do
  seed=$(( $(od -An -N4 -t u4 /dev/urandom) % 1073741824 ))
  ./devops/skypilot/launch.py train \
    run=$USER.nav_base_s${seed} \
    seed=${seed} \
    trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
    +trainer.contrastive.enabled=false \
    --skip-git-check \
    --no-spot \
    "$@"
done

for i in 1; do
  seed=$(( $(od -An -N4 -t u4 /dev/urandom) % 1073741824 ))
  metta tool train \
    run=$USER.nav_base_s${seed} \
    seed=${seed} \
    ++trainer.curriculum=/env/mettagrid/curriculum/navigation/bucketed \
    ++trainer.contrastive.enabled=false \
    "$@"
done
