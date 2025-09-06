## Super simple test - just export one function to verify basic mechanism works

proc test_function*(): cint {.cdecl, exportc, dynlib.} =
  return 42

when isMainModule:
  echo "Test function returns: ", test_function()