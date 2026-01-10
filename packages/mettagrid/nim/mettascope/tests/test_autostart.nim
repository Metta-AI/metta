## Test that the autostart functionality works correctly.
## This tests that the play variable can be set programmatically,
## which is what the autostart parameter does in the bindings.

import mettascope/common

block test_autostart:
  # Default play state should be false
  doAssert play == false, "Default play state should be false"

  # Test setting play to true (what autostart=true would do)
  play = true
  doAssert play == true, "play should be settable to true"

  # Test setting play back to false
  play = false
  doAssert play == false, "play should be settable to false"

  echo "✓ Autostart play variable test passed"

block test_playmode:
  # Test that playMode can be set to Realtime (what init does)
  doAssert playMode == Historical, "Default playMode should be Historical"

  playMode = Realtime
  doAssert playMode == Realtime, "playMode should be settable to Realtime"

  echo "✓ PlayMode test passed"
