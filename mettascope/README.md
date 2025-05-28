# MettaScope - Metta Replay Viewer & Player

This advanced WebGPU viewer allows you to watch and replay any metta replay. It allows play, pause, scrub through and step through the replay. You can select individual agents and see their individual action, history, rewards, and resources.

<p align="middle">
<img src="../../docs/readme_showoff.gif" alt="Metta learning example video">
<br>
<a href="https://metta-ai.github.io/metta/?replayUrl=https%3A%2F%2Fsoftmax-public.s3.us-east-1.amazonaws.com%2Freplays%2Fandre_pufferbox_33%2Freplay.77200.json.z&play=true">Interactive demo</a>
</p>

## Usage

You can either drag and drop a replay file or pass a url parameter to the player.

`?replayUrl=...the replay file...`

Most tools dealing with replays will provide a full link.

## Here are some replays to try out:

* [Simple Environment](http://localhost:2000/?replayUrl=https://softmax-public.s3.us-east-1.amazonaws.com/replays/andre_pufferbox_33/replay.77200.json.z)

* [The 4 Rooms](http://localhost:2000/?replayUrl=https%3A%2F%2Fsoftmax-public.s3.us-east-1.amazonaws.com%2Freplays%2Fb.daphne.terrain_multiagent_24_norewardsharing_maxinv%2Freplay.1500.json.z)

* [Heart collector](http://localhost:2000/?replayUrl=https://softmax-public.s3.us-east-1.amazonaws.com/replays/b.daphne.navigation_terrain_training/replay.31200.json.z)

* [The 4 Maze](http://localhost:2000/?replayUrl=https%3A%2F%2Fsoftmax-public.s3.us-east-1.amazonaws.com%2Freplays%2Fdaphne.navigation%2Freplay.21600.json.z)

* [The 280 Agents](http://localhost:2000/?replayUrl=https%3A%2F%2Fsoftmax-public.s3.us-east-1.amazonaws.com%2Freplays%2Fdaveey.na.240.1x4%2Freplay.8100.json.z)

## Installation

You need to install Node.js (v23.11.0) and typescript (Version 5.8.3), this might be different for different operating systems.

```bash
cd mettascope
npm install
tsc
python tools/gen_atlas.py
python -m http.server 2000
```

Then open the browser and go to `http://localhost:2000` to see the player.

## Development

To regenerate the html and css files, run the following command:

```bash
python tools/gen_html.py
```

## Using

```
1) Launch VSCode/Cursor
2) File > Open Folder... `metta/`
3) (Optional) Create a `configs/user/USERNAME.yaml`
In it, add this key (or other env):

replay_job:
  sim:
    env: /env/mettagrid/laser_tag
4) In VSCode/Cursor, at the top right, click "Run and Debug" (may be hidden in the drop down menu)
5) In the second dropdown (default "Train Metta"), click "Play Metta"
6) Hit the green arrow!
```
