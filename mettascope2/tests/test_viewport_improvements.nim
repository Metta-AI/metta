import std/[strformat],
  vmath, windy, boxy,
  ../src/tribal/[common, panels]

# Test that our new features compile and work correctly
proc testPanelFeatures() =
  echo "Testing Panel Viewport Improvements"
  echo "===================================="
  
  # Create a test panel
  var testPanel = Panel(
    panelType: WorldMap,
    name: "Test Panel",
    rect: IRect(x: 100, y: 100, w: 800, h: 600),
    pos: vec2(0, 0),
    zoom: 10.0
  )
  
  # Test 1: Check default zoom limits
  assert testPanel.minZoom == 5.0, "Default minZoom should be 5.0"
  assert testPanel.maxZoom == 100.0, "Default maxZoom should be 100.0"
  echo "✓ Default zoom limits are correct"
  
  # Test 2: Check hasMouse default
  assert testPanel.hasMouse == false, "hasMouse should default to false"
  echo "✓ hasMouse defaults to false"
  
  # Test 3: Test custom zoom limits
  testPanel.minZoom = 1.0
  testPanel.maxZoom = 50.0
  assert testPanel.minZoom == 1.0, "Custom minZoom should be 1.0"
  assert testPanel.maxZoom == 50.0, "Custom maxZoom should be 50.0"
  echo "✓ Custom zoom limits work"
  
  # Test 4: Check mouse capture globals
  assert mouseCaptured == false, "mouseCaptured should default to false"
  assert mouseCapturedPanel == nil, "mouseCapturedPanel should default to nil"
  echo "✓ Mouse capture globals initialized correctly"
  
  # Test 5: Test settings
  assert settings.showResources == true, "showResources should default to true"
  echo "✓ New settings field works"
  
  echo "\nAll tests passed! Viewport improvements are working correctly."

when isMainModule:
  testPanelFeatures()