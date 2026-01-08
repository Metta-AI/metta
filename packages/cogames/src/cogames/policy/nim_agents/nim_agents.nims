switch("app", "lib")
switch("tlsEmulation", "off")
when defined(windows):
  switch("out", "nim_agents.dll")
elif defined(macosx):
  switch("out", "libnim_agents.dylib")
else:
  switch("out", "libnim_agents.so")
switch("outdir", "bindings/generated")

--define:gennyPython

when not defined(debug):
  --define:release
