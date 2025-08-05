## Replay format specification version 2:

MettaScope uses a custom replay format to store the replay data. The replay is a zlib compressed json file with
`.json.z` extension.

Here is an example of how to decompress the file, from python:

```python
file_name = "replay.json.z"
with open(file_name, "rb") as file:
    compressed_data = file.read()
decompressed_data = zlib.decompress(compressed_data)
json_data = json.loads(decompressed_data)
```

In JavaScript it's a bit more complicated, but you can use the `decompressStream` with a streaming API.

The first key in the format is `version`, which is a number that contains the version of the replay format. Valid values
are `1`, `2`, etc. This document describes version 2.

```json
{
  "version": 2,
  ...
}
```

These are the constants that are stored at the top of the replay.

- `num_agents` - The number of agents in the replay.
- `max_steps` - The maximum number of steps in the replay.
- `map_size` - The size of the map. No object may move outside of the map bounds.
- `file_name` - The name of the replay file. This helps identify the replay when processing multiple files.

```json
{
  ...
  "num_agents": 24,
  "max_steps": 1000,
  "map_size": [62, 62],
  "file_name": "example_replay.json.z",
  ...
}
```

There are several key-to-string mapping arrays that are stored in the replay. We don't want to store full strings
everywhere so we store `type_id`, `action_id`, `items`, `group_id` as numbers. They correspond to `type_names`,
`action_names`, `item_names`, `group_names`.

```json
{
  ...
  "type_names": ["agent", "wall", "altar", ... ],
  "action_names": ["noop", "move", "rotate", ... ],
  "item_names": ["hearts", "coconuts", ... ],
  "group_names": ["group1", "group2", ... ],
  ...
}
```

## Objects and time series

The most important key in the format is `objects` which is a list of objects that are in the replay. Everything is an
object - walls, buildings, and agents.

```json
{
  ...
  "objects": [
    {...},
    {...},
    {...},
    ...
  ],
  ...
}
```

Objects are stored in a condensed format. Every field of the object is either a constant or a time series of values.

**Time series fields** can be represented in two ways:
1. **Single value** - When the field never changes during the replay, it's stored as just the value.
2. **Time series array** - When the field changes, it's stored as a list of tuples where the first element is the step and the second element is the value.

The time series array format uses tuples where the first element is the step and the second element is the value, which can be a number, boolean, or a list of numbers.

```json
{
  "id": 99,
  "type_id": 2,
  "agent_id": 0,
  "rotation": [[0, 1], [10, 2], [20, 3]],
  "location": [[0, [10, 10]], [1, [11, 10]], [2, [12, 11]]],
  "inventory": [[0, []], [100, [1]], [200, [1, 1]]],
  ...
}
```

In this example, the agent `type_id` - 2 in this case - never changes, so it's a constant. When looking up
`type_names[type_id]`, we get the name of the type, which is `"agent"`. The mapping between IDs and names can change
between replays. The `id` is a constant as well. All objects have IDs. The `agent_id` is a constant as well. Note there
are two IDs, one for the object and one for the agent. Agents have two IDs. The `rotation` is a time series of values.
The rotation is 1 at step 0, 2 at step 10, and 3 at step 20.

Here is the expanded version of the `rotation` key:

```json
{
  "rotation": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
}
```

You can either expand the whole time series on load or use binary search to find the value at a specific step. At first
I was using binary search, but expanding the time series is much faster. This is up to the implementation.

The `location` key is a time series of tuples, where the first element is the step and the second element is the
location, which is a list of two numbers for x and y.

The `inventory` key is a time series of tuples, where the first element is the step and the second element is the list
of item_IDs. It starts empty and then adds items at steps 100, 200, etc.

As another example, if the `rotation` key was always 1, it could also be stored simply as `"rotation": 1`.

## Key reference

Here are the keys supported for both agents and objects:

- `id` - Usually a constant. The id of the object.
- `type_id` - Usually a constant. The type of the object that references the `type_names` array.
- `location` - The [x, y, z] location of the object (sometimes called the column and row)
- `orientation` - The rotation of the object.

- `inventory` - The current list of item_IDs that map to the `item_names` array. Example: `[0, 0, 1]`. If
  `item_names = ["hearts", "bread"]`, then inventory is 2 hearts and 1 bread. The count is how many times the item
  repeats in the list. Note: In the replay data, this is represented in the `inventory` field as a time series showing
  how inventory changes over time (e.g., `[[0, []], [100, [1]], [200, [1, 1]]]`), where each entry contains a timestamp
  and the inventory state at that time and into the future.

- `inventory_max` - Usually a constant. Maximum number of items that can be in the inventory.
- `color` - The color of the object. Must be an integer between 0 and 255.

Agent specific keys:

- `agent_id` - Usually a constant. The id of the agent.
- `action_id` - The action of the agent that references the `action_names` array.
- `action_parameter` - Single value for the action. If `action_names[action_id] == "rotate"` and
  `action_parameter == 3`, this means move to the right. The implementation does not need to know this as it can be
  inferred from the rotation and x, y positions.
- `action_success` - Boolean value that indicates if the action was successful.
- `total_reward` - The total reward of the agent.
- `current_reward` - The reward of the agent for the current step.
- `frozen` - Boolean value that indicates if the agent is frozen.
- `frozen_progress` - A countdown from `frozen_time` to 0 that indicates how many steps are left to unfreeze the agent.
- `frozen_time` - Usually a constant. How many steps does it take to unfreeze the agent.
- `group_id` - The id of the group the object belongs to.

Object specific keys:

- `recipe_input` - Usually a constant. A list of item ids that map to the `item_names` array. Example: `[0, 0, 1]`. If
  `item_names = ["hearts", "bread"]`, then recipe input is 2 hearts and 1 bread. The count is how many times the item
  repeats in the list.
- `recipe_output` - Usually a constant. A list of item ids that map to the `item_names` array. Example: `[0, 0, 0, 0]`.
  If `item_names = ["hearts", ...]`, then recipe output is 4 hearts. The count is how many times the item repeats in the
  list.
- `recipe_max` - Usually a constant. Maximum number of `recipe_output` items that can be produced by the recipe before
  stopping.
- `production_progress` - Current progress of the recipe. Starts at 0 and goes until `production_time` is reached.
- `production_time` - Usually a constant. How many steps does it take to produce the recipe.
- `cooldown_progress` - How many steps are left to cooldown after producing the recipe. Starts at 0 and goes until
  `cooldown_time` is reached.
- `cooldown_time` - Usually a constant. How many steps does it take to cooldown after producing the recipe.

Keys are allowed to be missing. If a key is missing, missing keys are always 0, false, or []. Extra keys are ignored but
can be used by later implementations. If a time series starts from some other step like 100, then the first 99 steps are
just the default value.

## Reward sharing matrix

The reward sharing matrix is a constant that stores the reward sharing between agents. It is a two-dimensional array of
numbers. The first dimension is the agent_id taking the action and the second dimension is the agent_id receiving the
reward. If you look closer, in this 4x4 matrix you can see that there are two groups sharing 10% of reward with each
other; agents don't share reward with themselves.

```json
{
  ...
  "reward_sharing_matrix": [
    [0.0, 0.1, 0.0, 0.0],
    [0.1, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.1],
    [0.0, 0.0, 0.1, 0.0],
  ]
  ...
}
```

## Realtime WebSocket

This format extends into real time with some differences. Instead of getting a compressed JSON file, you connect to a
WebSocket and get replay format as a stream of messages. Each message can omit keys and only send them if they changed.
You can then take the current replay you have and extend it with the new message. Each message has a new step field:

````json
{
  "step": 100,
  "version": 2,
  ...
  "objects": [
    {...},
    {...},
    {...},
    ...
  ],
}

In this format there are no time series for the object properties. Instead everything is a constant that happens at the specific step.

On step 0:

```json
{
  "type_id": 2,
  "id": 99,
  "agent_id": 0,
  "rotation": 3,
  "location": [12, 11],
  "inventory": [1, 1],
  ...
}
````

On later steps, only the `id` is required and any changed keys are sent. Many keys like `type_id`, `agent_id`,
`group_id`, etc. don't change so they are only sent on step 0. While other keys like `location`, `inventory`, etc. are
sent every time they change.

```json
{
  "id": 99,
  "location": [12, 11],
  "inventory": [1, 1],
  ...
}
```

If no properties change, there is no need to send the object at all. Many static objects like walls are only spent on
step 0.
