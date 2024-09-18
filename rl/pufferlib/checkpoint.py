
import os

import torch



def save_checkpoint(data):
    config = data.config
    path = os.path.join(config.data_dir, config.exp_id)
    if not os.path.exists(path):
        os.makedirs(path)

    model_name = f'model_{data.epoch:06d}.pt'
    model_path = os.path.join(path, model_name)
    if os.path.exists(model_path):
        return model_path

    torch.save(data.uncompiled_policy, model_path)

    state = {
        'optimizer_state_dict': data.optimizer.state_dict(),
        'global_step': data.global_step,
        'agent_step': data.global_step,
        'epoch': data.epoch,
        'model_name': model_name,
        'exp_id': config.exp_id,
    }
    state_path = os.path.join(path, 'trainer_state.pt')
    torch.save(state, state_path + '.tmp')
    os.rename(state_path + '.tmp', state_path)

    if data.wandb is not None:
        artifact_name = f"{config.exp_id}_model"
        artifact = data.wandb.Artifact(
            artifact_name,
            type="model",
            metadata={
                "model_name": model_name,
                "agent_step": data.global_step,
                "epoch": data.epoch,
                "exp_id": config.exp_id,
            }
        )
        artifact.add_file(model_path)
        data.wandb.run.log_artifact(artifact)

    return model_path


def try_load_checkpoint(data):
    config = data.config
    path = os.path.join(config.data_dir, config.exp_id)
    if not os.path.exists(path):
        print('No checkpoints found. Assuming new experiment')
        return

    trainer_path = os.path.join(path, 'trainer_state.pt')
    resume_state = torch.load(trainer_path)
    model_path = os.path.join(path, resume_state['model_name'])
    data.global_step = resume_state['global_step']
    data.epoch = resume_state['epoch']
    data.uncompiled_policy = torch.load(model_path, map_location=config.device)
    data.policy = data.uncompiled_policy
    if data.config.compile:
        data.policy = torch.compile(data.policy, mode=config.compile_mode)

    data.optimizer.load_state_dict(resume_state['optimizer_state_dict'])
    print(f'Loaded checkpoint {resume_state["model_name"]}')
