### Goal

`tools/train.py` trains a policy.  As part of the training loop, we periodically
evaluate the policy on a variety of environments. The results of the simulation
get written to stats collection server, which can be accessed through a frontend called
'observatory'

A part of displaying the stats is rendering the environment maps in observatory.
Currently that's being done in kind of a stupid way - the FE indexes into a pre-rendered
set of maps on S3. However, that means that we need to maintain the pre-rendered maps,
which we don't want to do.

The goal is to call the rendering code from within observatory.

### Running the stack

1. download docker https://docs.docker.com/desktop/setup/install/mac-install/. open it, etc
2. run this to get a local version of postgres running: `docker run --name postgres -p 5432:5432 -e POSTGRES_PASSWORD=password -d pgvector/pgvector:pg17`. it'll yell at you if docker isn't running
3. `cd app_backend/src/metta/app_backend && uv run python server.py` to run local observatory backend
4. `cd observatory && npm run dev` to run the local frontend (new terminal)
5. `metta install observatory-key-local`. This should get a key for you to use for your local observatory. do cat ~/.metta/observatory_tokens.yaml to confirm that there's a line for http://localhost:8000 
6. `uv run python tools/train.py +user=ptsier +hardware=macbook wandb=off` to run training followed by evaluation.

Go to `http://localhost:5173` to the `navigation` tab, and hover over the cells. Notice the maps rendering at the bottom
of the page. This is what we're trying to achieve but with better rendering.

The map rendering code is in the `gridworks/` directory - follow the directions there.  If you need to pass anything
from trainer to observatory, see you can use the attributes in `stats_client.record_episode()`

