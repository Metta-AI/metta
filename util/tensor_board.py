import os
import shutil
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# === INPUTS ===
username = "alex"
root_dir = f"/Users/{username}/software/stem/metta/util"
trace_filename = "trace_m.alex.trace.j.02_epoch_16.json"
trace_source_path = os.path.join(root_dir, trace_filename)

# === LOGDIR ===
logdir = f"/Users/{username}/tensorboard_profiles"
timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
profile_dir = os.path.join(logdir, "plugins", "profile", timestamp)
os.makedirs(profile_dir, exist_ok=True)

# === Copy the trace ===
trace_dest_path = os.path.join(profile_dir, "local.trace.json")
shutil.copy(trace_source_path, trace_dest_path)

# === Write dummy scalar to trigger TensorBoard
writer = SummaryWriter(logdir=logdir)
writer.add_scalar("init/flag", 1, 0)
writer.close()

print(f"✅ Trace installed at: {trace_dest_path}")
print(f"🚀 Run this command:\n\ntensorboard --logdir={logdir}")
