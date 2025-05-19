import os
import shutil

# Source directory containing icons in the repo
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'mettascope', 'data', 'objects')
# Destination directory for map editor
DEST_DIR = os.path.join(os.path.dirname(__file__), 'assets')

ICONS = [
    'wall.png',
    'agent.png',
    'mine.png',
    'generator.png',
    'altar.png',
    'armory.png',
    'lasery.png',
    'lab.png',
    'factory.png',
    'temple.png'
]

os.makedirs(DEST_DIR, exist_ok=True)

for icon in ICONS:
    src = os.path.join(SRC_DIR, icon)
    dest = os.path.join(DEST_DIR, icon)
    if not os.path.isfile(src):
        print(f"Warning: source icon {src} not found")
        continue
    if not os.path.isfile(dest):
        shutil.copy2(src, dest)
        print(f"Copied {icon}")
    else:
        print(f"Skipping {icon}, already exists")
