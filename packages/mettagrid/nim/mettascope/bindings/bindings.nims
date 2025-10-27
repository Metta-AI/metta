
switch("app", "lib")
--define:fidgetUseCached
switch("tlsEmulation", "off")
when defined(windows):
  switch("out", "mettascope2.dll")
elif defined(macosx):
  switch("out", "libmettascope2.dylib")
else:
  switch("out", "libmettascope2.so")
switch("outdir", "bindings/generated")

when not defined(release):
  --define:noAutoGLerrorCheck
  --define:release
