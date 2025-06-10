
# ./devops/skypilot/launch.py train gd_backchain_kitchen5 trainer.curriculum=env/mettagrid/curriculum/backchain_kitchen \

# ./devops/skypilot/launch.py train gd_easy_sequence5 trainer.curriculum=env/mettagrid/curriculum/easy_sequence \

# # ./devops/skypilot/launch.py train gd_sequence_stripped3 trainer.curriculum=env/mettagrid/curriculum/sequence_stripped \

# ./devops/skypilot/launch.py train gd_kitchen_sink5 trainer.curriculum=env/mettagrid/curriculum/kitchen_sink \

# # ./devops/skypilot/launch.py train gd_backchain_none3 trainer.curriculum=env/mettagrid/curriculum/backchain_none \

# # ./devops/skypilot/launch.py train gd_backchain_seq3 trainer.curriculum=env/mettagrid/curriculum/backchain_seq \

# ./devops/skypilot/launch.py train gd_backchain3 trainer.curriculum=env/mettagrid/curriculum/backchain \

python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_backchain_mem  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain_mem --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_backchain  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_mem_pretrained  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/gd_backchain_mem_pretrained:v18 --skip-validation \

python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_navsequence_backchain  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_navsequence_backchain --skip-validation \
