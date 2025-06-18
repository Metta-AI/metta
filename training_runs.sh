
# #june 11th

# #RUNNING ON METTA1: georgedeane.devbox.nav_navsequence_backchain
# ./devops/skypilot/launch.py train georgedeane.sky.nav_navsequence_backchain trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_navsequence_backchain \

# # python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_navsequence_backchain  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_navsequence_backchain --skip-validation \


# #RUNNING ON METTA2: georgedeane.devbox.nav_backchain_mem
# # python -m devops.aws.batch.launch_task --cmd=train --run=b.georgedeane.nav_backchain_mem  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain_mem --skip-validation \

# ./devops/skypilot/launch.py train georgedeane.sky.nav_backchain_mem trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain_mem \


# # python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_mem_pretrained  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_mem trainer.initial_policy.uri=gd_backchain_mem_pretrained:v18 --skip-validation \

# ./devops/skypilot/launch.py train georgedeane.sky.nav_mem_pretrained trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_mem trainer.initial_policy.uri=wandb://run/gd_backchain_mem_pretrained:v18

# # python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_backchain  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain --skip-validation \

# ./devops/skypilot/launch.py train georgdeane.sky.nav_backchain trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain \

# # python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_backchain_mem_pretrained  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain_mem trainer.initial_policy.uri=wandb://run/dd_navigation_curriculum:54 --skip-validation \

# ./devops/skypilot/launch.py train georgedeane.sky.nav_backchain_mem_pretrained trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain_mem trainer.initial_policy.uri=wandb://run/dd_navigation_curriculum:v54








#OLD


# ./devops/skypilot/launch.py train gd_backchain_kitchen5 trainer.curriculum=env/mettagrid/curriculum/backchain_kitchen \

# ./devops/skypilot/launch.py train gd_easy_sequence5 trainer.curriculum=env/mettagrid/curriculum/easy_sequence \

# # ./devops/skypilot/launch.py train gd_sequence_stripped3 trainer.curriculum=env/mettagrid/curriculum/sequence_stripped \

# ./devops/skypilot/launch.py train gd_kitchen_sink5 trainer.curriculum=env/mettagrid/curriculum/kitchen_sink \

# # # ./devops/skypilot/launch.py train gd_backchain_none3 trainer.curriculum=env/mettagrid/curriculum/backchain_none \

# # # ./devops/skypilot/launch.py train gd_backchain_seq3 trainer.curriculum=env/mettagrid/curriculum/backchain_seq \

# # ./devops/skypilot/launch.py train gd_backchain3 trainer.curriculum=env/mettagrid/curriculum/backchain \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_backchain_mem  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain_mem --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_backchain2  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_mem_pretrained  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_mem trainer.initial_policy.uri=gd_backchain_mem_pretrained:v18 --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_navsequence_backchain2  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_navsequence_backchain --skip-validation \

# python -m devops.aws.batch.launch_task --cmd=train --run=b.georgdeane.nav_backchain_mem_pretrained  --git-branch=george-navsequence-experiments trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain_mem trainer.initial_policy.uri=gd_backchain_mem_pretrained:v18 --skip-validation \

# ./devops/skypilot/launch.py train georgedeane.nav_backchain_mem trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain_mem \

# ./devops/skypilot/launch.py train georgdeane.nav_mem_pretrained trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_mem trainer.initial_policy.uri=gd_backchain_mem_pretrained:v18

# ./devops/skypilot/launch.py train georgdeane.nav_backchain trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain \

# ./devops/skypilot/launch.py train georgedeane.nav_navsequence_backchain trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_navsequence_backchain \

# ./devops/skypilot/launch.py train georgedeane.nav_backchain_mem_pretrained trainer.curriculum=env/mettagrid/curriculum/navsequence/nav_backchain_mem trainer.initial_policy.uri=dd_navigation_curriculum:54 \

# ./devops/skypilot/launch.py train georgedeane.memory_pretrained trainer.curriculum=env/mettagrid/curriculum/navsequence/memory_mettascope \

# ./devops/skypilot/launch.py train georgedeane.memory_scratch trainer.curriculum=env/mettagrid/curriculum/navsequence/memory_mettascope \

# ./devops/skypilot/launch.py train georgedeane.mem_general trainer.curriculum=env/mettagrid/curriculum/navsequence/memory_mettascope trainer.initial_policy.uri=dd_navigation_curriculum \

#

# New training runs - June 17th

# # 1. Navigation curriculum from scratch
# ./devops/skypilot/launch.py train run=georgedeane.nav_scratch trainer.curriculum=env/mettagrid/curriculum/navigation --skip-git-check \

# 2. Memory mettascope curriculum with daphne.navigation.fromnumpy.skypilot initial policy
./devops/skypilot/launch.py train run=georgedeane.2mem_daphne_init trainer.curriculum=env/mettagrid/curriculum/navsequencepython -m tools.train run=george.mem_mettascope_pretrained.metta2 +hardware=pufferbox trainer.curriculum=env/mettagrid/curriculum/memory_mettascope trainer.initial_policy.uri=wandb://run/daphne.navigation.bucketed.skypilot --skip-git-check --no-spot\

# 3. Memory mettascope curriculum with relh.nav.new.42 initial policy
# ./devops/skypilot/launch.py train run=georgedeane.mem_relh_init trainer.curriculum=env/mettagrid/curriculum/navsequence/memory_mettascope trainer.initial_policy.uri=wandb://run/relh.nav.new.42 --skip-git-check \

# 4. Extended sequence curriculum from scratch
./devops/skypilot/launch.py train run=georgedeane.2extended_sequence trainer.curriculum=env/mettagrid/curriculum/extended_sequence --skip-git-check --no-spot\

# # 5. Memory training varied curriculum - comprehensive memory training and evaluation
 ./devops/skypilot/launch.py train run=georgedeane.2memory_training_varied2 trainer.curriculum=env/mettagrid/curriculum/memory_training_varied --skip-git-check --no-spot\

