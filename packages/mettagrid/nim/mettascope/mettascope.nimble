version     = "0.0.1"
author      = "Softmax"
description = "Visualization of the MettaGrid environment."
license     = "MIT"

srcDir = "src"

requires "nim >= 2.2.4"
requires "cligen >= 1.9.0"
requires "fidget2 >= 0.0.12"
requires "genny >= 0.1.1"

task bindings, "Generate bindings":

  proc compile(libName: string) =
    exec "nim c -d:release --app:lib -d:fidgetUseCached=true --tlsEmulation:off --out:" & libName & " --outdir:bindings/generated bindings/bindings.nim"

  when defined(windows):
    compile "mettascope2.dll"
  elif defined(macosx):
    compile "libmettascope2.dylib"
  else:
    compile "libmettascope2.so"
