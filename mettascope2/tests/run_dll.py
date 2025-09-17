# Add the bindings/generated directory to the path
import sys

sys.path.append("bindings/generated")


import mettascope2

current_step = 0
while True:
    should_close = mettascope2.render(current_step)
    if should_close:
        break
    current_step += 1
