from omegaconf import OmegaConf
from PIL import Image

from mettagrid.map.mapgen import MapGen

# import numpy as np
from mettagrid.resolvers import register_resolvers

# import glob
# import os

'''
It's a small script to quickly draw terrains
'''

w = 180
h = 150

for i in range(1):
    register_resolvers()
    config = OmegaConf.load("mettagrid/configs/game/map_builder/mapgen_simsam_symmetry.yaml")

    if OmegaConf.select(config, "root") is not None:
        root = config.root
    else:
        print("No root config found, using default maze")

    world_map = MapGen(w, h, root=root).build()

    Image.fromarray((world_map == 'empty')).save(f'{i}.png')
    # Image.fromarray((world_map == "empty")).resize(
    #     (((h + 2) * (500 // (h + 2))), ((w + 2) * (500 // (w + 2))))
    # ).save(f"{i}.png")


"""
This code generates a gif from the set of generated maps
additional requirements: pip install pillow
"""

# def make_gif():
#     frames = [Image.open(image) for image in sorted(glob.glob('/home/catnee/mettagrid/giffactory/*.png'), 
#               key=os.path.getmtime)]
#     frame_one = frames[0]
#     frame_one.save("my_awesome.gif", format="GIF", append_images=frames,
#                save_all=True, duration=100, loop=0)

# N = 100
# for i in range(N):
#     print(f'{i} step out of {N}')
#     t = 0.01*(1.08**i)
#     m = 0.01*i
#     room = TerrainGen(w,h, layers, sampling_params={'M':t, 'O':m}, scenario='maze', seed=11)._build()
#     room = (room != 'wall')
#     Image.fromarray(room).resize((400, 400)).save(f'giffactory/{i}.png')

# make_gif()
# print(f'{N} step out of {N}')
