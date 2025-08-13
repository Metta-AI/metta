# %%
import os
import yaml

def generate_curriculum_files():
    base_env_dir = "configs/env/mettagrid/operant_conditioning/in_context_learning"
    base_curriculum_dir = "configs/curriculum/mettagrid/operant_conditioning/in_context_learning"

    # Walk through all environment directories
    for root, dirs, files in os.walk(base_env_dir):
        # Extract the relative path from the base env directory
        rel_path = os.path.relpath(root, base_env_dir)

        # Create corresponding curriculum directory
        curriculum_dir = os.path.join(base_curriculum_dir, rel_path)
        os.makedirs(curriculum_dir, exist_ok=True)

        # Get all yaml files in this env directory
        yaml_files = [f for f in files if f.endswith('.yaml')]

        if yaml_files:
            # Create curriculum file for this directory
            curriculum_file = os.path.join(curriculum_dir, "random.yaml")

            # Create curriculum content matching the existing format
            curriculum_content = {
                "_target_": "metta.mettagrid.curriculum.random.RandomCurriculum",
                "tasks": {}
            }

            # Add all environments as tasks with weight 1
            for yaml_file in yaml_files:
                # Remove .yaml extension to get environment name
                env_name = yaml_file[:-5]
                # Use the exact path format from the existing file
                task_path = f"/env/mettagrid/curriculum/in_context_learning/{rel_path}/{env_name}"
                curriculum_content["tasks"][task_path] = 1

            # Write curriculum file
            with open(curriculum_file, 'w') as f:
                yaml.dump(curriculum_content, f, default_flow_style=False, indent=2)

            print(f"Created curriculum: {curriculum_file} with {len(yaml_files)} tasks")

if __name__ == "__main__":
    generate_curriculum_files()
# %%
