--define:gennyPython

switch("app", "lib")
switch("tlsEmulation", "off")
when defined(windows):
  switch("out", "mettascope.dll")
elif defined(macosx):
  switch("out", "libmettascope.dylib")
else:
  switch("out", "libmettascope.so")
switch("outdir", "bindings/generated")

when not defined(release):
  --define:noAutoGLerrorCheck
  --define:release
