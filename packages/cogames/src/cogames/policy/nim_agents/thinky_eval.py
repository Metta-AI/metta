# much simpler evaluator for thinky agents.

# Run the thinky agent through all of the missions and prints out the stats.

agent_name = "thinky"
agent_path = "cogames.policy.nim_agents.agents.ThinkyAgentsMultiPolicy"

# tag .. map name ................................... harts/A .. time .. first heart
# buggy hello_world.oxygen_bottleneck  .............. 20 .... 3s ...... 29 step
# good  hello_world.energy_starved .................. 20 .... 3s ...... 29 step
# n/a   hello_world.vibe_check  ..................... 20 .... 3s ...... 29 step
# hard  hello_world.vibe_check  ..................... 20 .... 3s ...... 29 step
# ...
# total 234 evals .................................. 120 . 3:32s .... 1232 step

evals = [
  ("hello_world.oxygen_bottleneck", "buggy"),
  ("hello_world.energy_starved", "good"),
  ("hello_world.vibe_check", "n/a"),
  ("hello_world.vibe_check", "hard"),
]

def run_eval(map_name: str, tag: str) =
  # find the eval
  # load the needed stuff
  # run the eval
  # print the stats in one line


print("tag .. map name ................................... harts/A .. time .. first heart")

for eval in evals:
  run_eval(eval[0], eval[1])

print("total ")
