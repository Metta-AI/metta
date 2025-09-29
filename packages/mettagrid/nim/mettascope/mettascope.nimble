version     = "0.0.1"
author      = "Softmax"
description = "Visualization of the MettaGrid environment."
license     = "MIT"

srcDir = "src"

requires "nim >= 2.2.4"
requires "boxy >= 0.1.4"
requires "cligen >= 1.9.0"
requires "jsony >= 1.1.5"
requires "windy >= 0.1.2"
requires "puppy >= 2.1.2"
requires "https://github.com/treeform/pixie.git"
requires "https://github.com/treeform/fidget2.git"
requires "genny >= 0.1.0"

task bindings, "Generate bindings":

  proc compile(libName: string, flags = "") =
    exec "nim c -f " & flags & " -d:release --app:lib --gc:arc --tlsEmulation:off --out:" & libName & " --outdir:bindings/generated bindings/bindings.nim"

  when defined(windows):
    compile "mettascope2.dll"
  elif defined(macosx):
    compile "libmettascope2.dylib"
  else:
    compile "libmettascope2.so"
