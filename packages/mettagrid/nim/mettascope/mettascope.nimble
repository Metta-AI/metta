version     = "0.0.3"
author      = "Softmax"
description = "Visualization of the MettaGrid environment."
license     = "MIT"

srcDir = "src"

requires "nim >= 2.2.4"
requires "cligen >= 1.9.0"
requires "fidget2 >= 0.1.2"
requires "genny >= 0.1.1"

task bindings, "Generate bindings":

  proc compile(libName: string) =
    exec "nim c -d:release --app:lib -d:fidgetUseCached=true --tlsEmulation:off --out:" & libName & " --outdir:bindings/generated bindings/bindings.nim"

  when defined(windows):
    compile "mettascope.dll"
  elif defined(macosx):
    compile "libmettascope.dylib"
  else:
    compile "libmettascope.so"
