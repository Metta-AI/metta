## Vibe panel allows you to set vibe frequency for the agent.
## Vibe are emoji like symbols that the agent can use to communicate with
## the world and other agents.

import
  fidget2,
  common, panels

proc updateVibePanel*() =
  ## Updates the vibe panel to display the current vibe frequency for the agent.

  let x = panels.vibeTemplate.copy()
  x.position = vec2(0, 0)
  echo "vibePanel: "
  echo x.dumpTree()
  vibePanel.node.removeChildren()
  vibePanel.node.addChild(x)
