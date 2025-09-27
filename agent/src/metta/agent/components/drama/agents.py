import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.distributions as distributions
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from torch.cuda.amp import autocast
import numpy as np
from sub_models.laprop import LaProp
from pytorch_warmup import LinearWarmup
# from nfnets import AGC

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import EMAScalar
from line_profiler import profile
from tools import layer_init

def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per

@profile
def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device="cuda")
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device=rewards.device)
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = \
            rewards[:, t] + \
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
    return gamma_return[:, :-1]

class RunningMeanStd:
    def __init__(self, shape, device):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = torch.tensor(1e-4, dtype=torch.float32, device=device)

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)

        self.mean, self.var, self.count = self.update_mean_var_count(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def update_mean_var_count(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

class VecNormalize(nn.Module):
    def __init__(self, shape, device, epsilon=1e-8, clipob=10.0):
        super(VecNormalize, self).__init__()
        self.ob_rms = RunningMeanStd(shape, device)
        self.epsilon = epsilon
        self.clipob = clipob

    def forward(self, x):
        if x.dim() == 2:  # (B, D)
            x_flat = x
        elif x.dim() == 3:  # (B, L, D)
            B, L, D = x.shape
            x_flat = x.view(-1, D)  # Flatten to (B*L, D)
        else:
            raise ValueError("Unsupported input shape")

        self.ob_rms.update(x_flat)
        mean = self.ob_rms.mean
        var = self.ob_rms.var
        x_normalized = torch.clamp((x_flat - mean) / torch.sqrt(var + self.epsilon), -self.clipob, self.clipob)

        if x.dim() == 2:  # (B, D)
            return x_normalized
        elif x.dim() == 3:  # (B, L, D)
            return x_normalized.view(B, L, D)
    
class ActorCriticAgent(nn.Module):
    def __init__(self, conf, action_dim, device) -> None:
        super().__init__()
        feat_dim=conf.Models.WorldModel.CategoricalDim*conf.Models.WorldModel.ClassDim+conf.Models.WorldModel.HiddenStateDim
        num_layers=conf.Models.Agent.AC.NumLayers
        actor_hidden_dim=conf.Models.Agent.AC.Actor.HiddenUnits
        critic_hidden_dim=conf.Models.Agent.AC.Critic.HiddenUnits
        self.gamma = conf.Models.Agent.AC.Gamma
        self.lambd = conf.Models.Agent.AC.Lambda
        self.entropy_coef = conf.Models.Agent.AC.EntropyCoef
        self.use_amp = conf.BasicSettings.Use_amp
        self.max_grad_norm=conf.Models.Agent.AC.Max_grad_norm
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.action_dim = action_dim
        self.unimix_ratio = conf.Models.Agent.Unimix_ratio
        self.device = device

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)
        act = getattr(nn, conf.Models.Agent.AC.Act)

        actor = [
            VecNormalize(feat_dim, device=device),
            layer_init(nn.Linear(feat_dim, actor_hidden_dim, bias=True)),
            RMSNorm(actor_hidden_dim),
            act()
        ]
        for i in range(num_layers - 1):
            actor.extend([
                layer_init(nn.Linear(actor_hidden_dim, actor_hidden_dim, bias=True)),
                RMSNorm(actor_hidden_dim),
                act()
            ])
        self.actor = nn.Sequential(
            *actor,
            layer_init(nn.Linear(actor_hidden_dim, action_dim), std=0.001)
        ).to(device)
        

        critic = [
            layer_init(nn.Linear(feat_dim, critic_hidden_dim, bias=True)),
            RMSNorm(critic_hidden_dim),
            act()
        ]
        for i in range(num_layers - 1):
            critic.extend([
                layer_init(nn.Linear(critic_hidden_dim, critic_hidden_dim, bias=True)),
                RMSNorm(critic_hidden_dim),
                act()
            ])

        self.critic = nn.Sequential(
            *critic,
            layer_init(nn.Linear(critic_hidden_dim, 255), std=0.001)
        ).to(device)
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        if conf.Models.Agent.AC.Optimiser == 'Laprop':
            self.optimizer = LaProp(self.parameters(), lr=conf.Models.Agent.AC.Laprop.LearningRate, eps=conf.Models.Agent.AC.Laprop.Epsilon)
        elif conf.Models.Agent.AC.Optimiser == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=conf.Models.Agent.AC.Adam.LearningRate, 
                eps=conf.Models.Agent.AC.Adam.Epsilon
            )
        else:
            raise ValueError(f"Unknown optimiser: {conf.Models.Agent.AC.Optimiser}")
        # self.optimizer = AGC(self.parameters(), self.optimizer)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0) # No lr schedule but neccessary for the warm up
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=conf.Models.Agent.AC.Warmup_steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def policy(self, x):
        logits = self.actor(x)
        logits = self.unimix(logits)
        return logits

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    def unimix(self, logits):
        # uniform noise mixing
        if self.unimix_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self.action_dim
            mixed_probs = self.unimix_ratio * uniform + (1-self.unimix_ratio) * probs
            logits = torch.log(mixed_probs)
        return logits

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits = self.policy(latent)
            dist = distributions.Categorical(logits=logits)
            if greedy:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
        return action, logits

    def sample_as_env_action(self, latent, greedy=False):
        action, _ = self.sample(latent, greedy)
        return action.detach().cpu().squeeze(-1).numpy()
    @profile
    def update(self, latent, action, old_logits, context_latent, context_reward, context_termination, reward, termination, logger, global_step):
        '''
        Update policy and value model
        '''
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits, raw_value = self.get_logits_raw_value(latent)
            dist = distributions.Categorical(logits=logits[:, :-1])
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())
                
            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1, device=reward.device), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()
        self.warmup_scheduler.dampen()
        self.update_slow_critic()

        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item(), global_step=global_step)
            logger.log('ActorCritic/value_loss', value_loss.item(), global_step=global_step)
            logger.log('ActorCritic/entropy_loss', -entropy_loss.item(), global_step=global_step)
            logger.log('ActorCritic/S', S.item(), global_step=global_step)
            logger.log('ActorCritic/norm_ratio', norm_ratio.item(), global_step=global_step)
            logger.log('ActorCritic/total_loss', loss.item(), global_step=global_step)




class PPOAgent(nn.Module):
    def __init__(self, conf, action_dim, device):
        super().__init__()
        feat_dim=conf.Models.WorldModel.CategoricalDim*conf.Models.WorldModel.ClassDim+conf.Models.WorldModel.HiddenStateDim
        num_layers=conf.Models.Agent.PPO.NumLayers
        actor_hidden_dim=conf.Models.Agent.PPO.Actor.HiddenUnits
        critic_hidden_dim=conf.Models.Agent.PPO.Critic.HiddenUnits      
        self.gamma = conf.Models.Agent.PPO.Gamma
        self.lambd = conf.Models.Agent.PPO.Lambda
        self.entropy_coef = conf.Models.Agent.PPO.EntropyCoef
        self.eps_clip=conf.Models.Agent.PPO.EpsilonClip
        self.K_epochs=conf.Models.Agent.PPO.K_epochs
        
        self.minibatch_size=conf.Models.Agent.PPO.Minibatch
        self.c1=conf.Models.Agent.PPO.CriticCoef
        self.c2=conf.Models.Agent.PPO.EntropyCoef
        self.kl_threshold=conf.Models.Agent.PPO.KL_threshold
        self.max_grad_norm=conf.Models.Agent.PPO.Max_grad_norm
        self.use_amp = conf.BasicSettings.Use_amp
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self.action_dim = action_dim
        self.unimix_ratio = conf.Models.Agent.Unimix_ratio
        self.device = device

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)
        act = getattr(nn, conf.Models.Agent.PPO.Act)

        actor = [
            VecNormalize(feat_dim, device=device),
            layer_init(nn.Linear(feat_dim, actor_hidden_dim, bias=True)),
            RMSNorm(actor_hidden_dim),
            act()
        ]
        for i in range(num_layers - 1):
            actor.extend([
                layer_init(nn.Linear(actor_hidden_dim, actor_hidden_dim, bias=True)),
                RMSNorm(actor_hidden_dim),
                act()
            ])
        self.actor = nn.Sequential(
            *actor,
            layer_init(nn.Linear(actor_hidden_dim, action_dim), std=0.001)
        ).to(device)

        critic = [
            layer_init(nn.Linear(feat_dim, critic_hidden_dim, bias=True)),
            RMSNorm(critic_hidden_dim),
            act()
        ]
        for i in range(num_layers - 1):
            critic.extend([
                layer_init(nn.Linear(critic_hidden_dim, critic_hidden_dim, bias=True)),
                RMSNorm(critic_hidden_dim),
                act()
            ])

        self.critic = nn.Sequential(
            *critic,
            layer_init(nn.Linear(critic_hidden_dim, 255), std=0.001)
        ).to(device)

        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        if conf.Models.Agent.PPO.Optimiser == 'Laprop':
            self.optimizer = LaProp(self.parameters(), lr=conf.Models.Agent.PPO.Laprop.LearningRate, eps=conf.Models.Agent.PPO.Laprop.Epsilon)
        elif conf.Models.Agent.PPO.Optimiser == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=conf.Models.Agent.PPO.Adam.LearningRate, 
                eps=conf.Models.Agent.PPO.Adam.Epsilon
            )
        else:
            raise ValueError(f"Unknown optimiser: {conf.Models.Agent.PPO.Optimiser}")
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: 1.0) # No lr schedule but neccessary for the warm up
        self.warmup_scheduler = LinearWarmup(self.optimizer, warmup_period=conf.Models.Agent.PPO.Warmup_steps)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
    @profile
    def get_logp_val_entr(self, latent, action, longer_value=True):
        if longer_value:
            logits = self.actor(latent[:, :-1])
        else:
            logits = self.actor(latent)
        value = self.critic(latent)
        dist = distributions.Categorical(logits=logits)
        logp_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return logp_prob, value, entropy
    
    
    def unimix(self, logits):
        # uniform noise mixing
        if self.unimix_ratio > 0:
            probs = F.softmax(logits, dim=-1)
            uniform = torch.ones_like(probs) / self.action_dim
            mixed_probs = self.unimix_ratio * uniform + (1-self.unimix_ratio) * probs
            logits = torch.log(mixed_probs)
        return logits

    def sample_as_env_action(self, latent, greedy=False):
            action, _ = self.sample(latent, greedy)
            return action.detach().cpu().squeeze(-1).numpy()    
    @profile
    def comput_loss(self, latent, action, logp_old, advs, rtgs, slow_return):

        logp, raw_values, entropy = self.get_logp_val_entr(latent, action, longer_value=False)

        ratio = torch.exp(logp - logp_old)
        # Kl approx according to http://joschu.net/blog/kl-approx.html
        kl_apx = ((ratio - 1) - (logp - logp_old)).mean()
    
        clip_advs = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advs
        # Torch Adam implement tation mius the gradient, to plus the gradient, we need make the loss negative
        actor_loss = -(torch.min(ratio*advs.detach(), clip_advs.detach())).mean()

        # values = values.flatten() # I used squeeze before, maybe a mistake
        slow_critic_loss = self.symlog_twohot_loss(raw_values, slow_return.detach())
        critic_loss = self.symlog_twohot_loss(raw_values, rtgs.detach())
        # critic_loss = F.mse_loss(values, rtgs)
        # critic_loss = ((values - rtgs) ** 2).mean()

        entropy_loss = entropy.mean()

        return actor_loss, critic_loss, slow_critic_loss, entropy_loss, kl_apx 


    @profile
    def calc_gae_and_reward_to_go(self, rewards, values, termination):
        # Invert termination to have 0 if the episode ended and 1 otherwise
        inv_termination = (termination * -1) + 1

        batch_size, batch_length = rewards.shape[:2]
        
        deltas = torch.zeros((batch_size, batch_length), dtype=rewards.dtype, device=rewards.device)
        advantages = torch.zeros((batch_size, batch_length+1), dtype=rewards.dtype, device=rewards.device)
        # reward_to_go = torch.zeros((batch_size, batch_length), dtype=rewards.dtype, device=rewards.device)
    
        # Calculate deltas
        for t in range(batch_length):
            next_value = values[:, t+1]
            deltas[:, t] = rewards[:, t] + self.gamma * inv_termination[:, t] * next_value - values[:, t]
        
        # Calculate advantages (GAE)
        for t in reversed(range(batch_length)):
            next_advantage = advantages[:, t+1] if t < batch_length - 1 else 0
            advantages[:, t] = deltas[:, t] + self.gamma * self.lamb * inv_termination[:, t] * next_advantage
        
        # Calculate reward-to-go
        # for t in reversed(range(batch_length)):
        #     next_return = reward_to_go[:, t+1] if t < batch_length - 1 else 0
        #     reward_to_go[:, t] = rewards[:, t] + self.gamma * inv_termination[:, t] * next_return
        
        # Compute the final returns by adding the value function estimates to the advantages
        returns = advantages[:, :-1] + values[:, :-1]
        
        return advantages[:, :-1], returns #, reward_to_go, deltas
    
    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value
    
    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))


    @profile
    def update(self, latent, action, old_logits, context_latent, context_reward, context_termination, reward, termination, logger, global_step):
        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            feat_dim = latent.shape[-1]
            dist = distributions.Categorical(logits=old_logits)
            old_logp = dist.log_prob(action)

            flatten_latent = latent[:, :-1].reshape(-1, feat_dim)
            flatten_action = action.view(-1)
            flatten_old_logp = old_logp.view(-1).detach()

            batch_size = flatten_latent.shape[0]

            entropy_loss_list = []
            actor_loss_list = []
            critic_loss_list = []
            total_loss_list = []
            kl_approx_list = []
            
            for _ in range(self.K_epochs):
                # Recompute the value after each update Andrychowicz et al. 2020 3.5
                value = self.value(latent)
                slow_value = self.slow_value(latent)

                lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd)
                slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd)
                # context_lambda_return = calc_lambda_return(context_reward[:, 1:], context_termination[:, 1:], self.gamma, self.lamb)
                
                # advantage, lambda_return = self.calc_gae_and_reward_to_go(reward, value, termination)
                # Normalize the tensor
                # norm_advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
                upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
                S = upper_bound-lower_bound
                norm_ratio = torch.max(torch.ones(1, device=reward.device), S)  # max(1, S) in the paper
                norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
                

                flatten_advantages = norm_advantage.view(-1)
                flatten_returns = lambda_return.reshape(-1)
                flatten_slow_return = slow_lambda_return.reshape(-1)                
                # Shuffle indices
                inds = np.arange(batch_size)
                np.random.shuffle(inds)
                
                for start in range(0, batch_size, self.minibatch_size):
                    end = start + self.minibatch_size

                    minibatch_inds = inds[start:end]
                
                    actor_loss, critic_loss, slow_critic_loss, entropy_loss, kl_apx = self.comput_loss(
                        flatten_latent[minibatch_inds], 
                        flatten_action[minibatch_inds], 
                        flatten_old_logp[minibatch_inds], 
                        flatten_advantages[minibatch_inds], 
                        flatten_returns[minibatch_inds],
                        flatten_slow_return[minibatch_inds]
                    )
                    
                    total_loss = actor_loss + self.c1 * critic_loss + slow_critic_loss - self.c2 * entropy_loss

                    entropy_loss_list.append(-entropy_loss.item())
                    actor_loss_list.append(actor_loss.item())
                    critic_loss_list.append(critic_loss.item())
                    total_loss_list.append(total_loss.item())
                    kl_approx_list.append(kl_apx.item())

                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)  # for clip grad
                    torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)

                    self.update_slow_critic()
                    
            self.lr_scheduler.step()
            self.warmup_scheduler.dampen()
        if logger is not None:
            logger.log('ActorCritic/policy_loss', np.mean(actor_loss_list), global_step=global_step)
            logger.log('ActorCritic/value_loss', np.mean(critic_loss_list), global_step=global_step)
            logger.log('ActorCritic/entropy_loss', np.mean(entropy_loss_list), global_step=global_step)
            logger.log('ActorCritic/KL_approx', np.mean(kl_approx_list), global_step=global_step)
            logger.log('ActorCritic/S', S.item(), global_step=global_step)
            logger.log('ActorCritic/norm_ratio', norm_ratio.item(), global_step=global_step)
            logger.log('ActorCritic/total_loss', np.mean(total_loss_list), global_step=global_step)

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits = self.actor(latent)
            logits = self.unimix(logits)
            dist = distributions.Categorical(logits=logits)
            if greedy:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
        return action, logits

