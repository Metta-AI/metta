switch("app", "lib")
switch("tlsEmulation", "off")
when defined(windows):
  switch("out", "fast_agents.dll")
elif defined(macosx):
  switch("out", "libfast_agents.dylib")
else:
  switch("out", "libfast_agents.so")
switch("outdir", "bindings/generated")

--define:gennyPython

when not defined(debug):
  --define:release
