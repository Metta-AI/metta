./devops/sweep.sh run=$USER.arena_contrastive \
  sweep=contrastive \
  sim=arena \
  ++trainer.curriculum=/env/mettagrid/curriculum/arena/learning_progress \
  ++trainer.optimizer.type=muon \
  ++trainer.optimizer.learning_rate=0.0045 \
  ++trainer.simulation.evaluate_interval=50
