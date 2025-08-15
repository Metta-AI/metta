coef_values=(0.10) # 0.20 0.40) # 0.03
temperature_values=(0.03 0.05 0.07 0.12)
num_negatives_values=(8 16 32)

base_args="sim=arena \
  ++trainer.curriculum=/env/mettagrid/curriculum/arena/learning_progress \
  ++trainer.optimizer.type=muon \
  ++trainer.optimizer.learning_rate=0.0045 \
  ++trainer.simulation.evaluate_interval=50 \
  ++trainer.contrastive.enabled=true \
  ++trainer.contrastive.gamma=0.99 \
  ++trainer.contrastive.logsumexp_coef=0.01 \
  ++trainer.contrastive.var_reg_coef=1.0 \
  ++trainer.contrastive.var_reg_target=1.0"

for coef in "${coef_values[@]}"; do
  for temp in "${temperature_values[@]}"; do
    for neg in "${num_negatives_values[@]}"; do
      run_id="contrastive_arena_sweep.$(date +%m-%d-%H-%M)-c${coef}-t${temp}-n${neg}"
      ./devops/skypilot/launch.py train --gpus=4 --nodes=1 --no-spot run="$run_id" \
        $base_args \
        ++trainer.contrastive.coef="$coef" \
        ++trainer.contrastive.temperature="$temp" \
        ++trainer.contrastive.num_negatives="$neg"
    done
  done
done
