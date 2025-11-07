switch("app", "lib")
switch("tlsEmulation", "off")
when defined(windows):
  switch("out", "heuristic_agents.dll")
elif defined(macosx):
  switch("out", "libheuristic_agents.dylib")
else:
  switch("out", "libheuristic_agents.so")
switch("outdir", "bindings/generated")

--define:gennyPython

when not defined(debug):
  --define:release
