
# ./devops/skypilot/launch.py train gd_backchain_kitchen5 trainer.curriculum=env/mettagrid/curriculum/backchain_kitchen \

# ./devops/skypilot/launch.py train gd_easy_sequence5 trainer.curriculum=env/mettagrid/curriculum/easy_sequence \

# # ./devops/skypilot/launch.py train gd_sequence_stripped3 trainer.curriculum=env/mettagrid/curriculum/sequence_stripped \

# ./devops/skypilot/launch.py train gd_kitchen_sink5 trainer.curriculum=env/mettagrid/curriculum/kitchen_sink \

# # ./devops/skypilot/launch.py train gd_backchain_none3 trainer.curriculum=env/mettagrid/curriculum/backchain_none \

# # ./devops/skypilot/launch.py train gd_backchain_seq3 trainer.curriculum=env/mettagrid/curriculum/backchain_seq \

# ./devops/skypilot/launch.py train gd_backchain3 trainer.curriculum=env/mettagrid/curriculum/backchain \

 ./devops/skypilot/launch.py train gd_backchain_full_extended trainer.curriculum=env/mettagrid/curriculum/backchain_full_extended \

 ./devops/skypilot/launch.py train gd_backchain_in_context trainer.curriculum=env/mettagrid/curriculum/backchain_in_context \

 ./devops/skypilot/launch.py train gd_backchain_scratch_hard trainer.curriculum=env/mettagrid/curriculum/backchain_scratch_hard \

 ./devops/skypilot/launch.py train gd_hard_mem_seq_pretrained trainer.curriculum=env/mettagrid/curriculum/hard_mem_seq_pretrained trainer.initial_policy.uri=wandb://run/gd_backchain_mem_pretrained \

 ./devops/skypilot/launch.py train gd_pure_mem_backchain trainer.curriculum=env/mettagrid/curriculum/pure_mem_backchain \

 ./devops/skypilot/launch.py train gd_pure_seq_backchain trainer.curriculum=env/mettagrid/curriculum/pure_seq_backchain \

 ./devops/skypilot/launch.py train gd_kitchen_hard_pretrained trainer.curriculum=env/mettagrid/curriculum/all trainer.initial_policy.uri=wandb://run/gd_backchain_mem_pretrained \






