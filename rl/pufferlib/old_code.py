

    # def rollout(self, cfg: OmegaConf, env_creator, env_kwargs, agent_creator, agent_kwargs,
    #         backend, render_mode='auto', model_path=None, device='cuda'):

    #     if render_mode != 'auto':
    #         env_kwargs['render_mode'] = render_mode

    #     # We are just using Serial vecenv to give a consistent
    #     # single-agent/multi-agent API for evaluation
    #     env = pufferlib.vector.make(env_creator, env_kwargs=env_kwargs, backend=backend)

    #     if model_path is None:
    #         agent = agent_creator(env.driver_env, agent_kwargs).to(device)
    #     else:
    #         agent = torch.load(model_path, map_location=device)

    #     ob, info = env.reset()
    #     driver = env.driver_env
    #     os.system('clear')
    #     state = None

    #     frames = []
    #     tick = 0
    #     while tick <= 1000:
    #         if tick % 1 == 0:
    #             render = driver.render()
    #             if driver.render_mode == 'ansi':
    #                 print('\033[0;0H' + render + '\n')
    #                 time.sleep(0.05)
    #             elif driver.render_mode == 'rgb_array':
    #                 frames.append(render)
    #                 import cv2
    #                 render = cv2.cvtColor(render, cv2.COLOR_RGB2BGR)
    #                 cv2.imshow('frame', render)
    #                 cv2.waitKey(1)
    #                 time.sleep(1/24)
    #             elif driver.render_mode in ('human', 'raylib') and render is not None:
    #                 frames.append(render)

    #         with torch.no_grad():
    #             ob = torch.as_tensor(ob).to(device)
    #             if hasattr(agent, 'lstm'):
    #                 action, _, _, _, state = agent(ob, state)
    #             else:
    #                 action, _, _, _ = agent(ob)

    #             action = action.cpu().numpy().reshape(env.action_space.shape)

    #         ob, reward = env.step(action)[:2]
    #         reward = reward.mean()
    #         if tick % 128 == 0:
    #             print(f'Reward: {reward:.4f}, Tick: {tick}')
    #         tick += 1

    #     # Save frames as gif
    #     #import imageio
    #     #imageio.mimsave('../docker/eval.gif', frames, fps=15, loop=0)
