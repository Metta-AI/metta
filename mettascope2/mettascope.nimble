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
requires "fidget2#head"
requires "vmath >= 2.0.0"
requires "bumpy >= 1.1.0"
requires "chroma >= 0.2.7"
requires "genny >= 0.1.0"

task bindings, "Generate Python bindings for tribal environment":
  # Generate the bindings
  exec "nim r bindings/tribal_bindings.nim"
  
  # Create the shared library
  when defined(windows):
    exec "nim c --app:lib --mm:arc --opt:speed --outdir:bindings/generated --out:tribal.dll bindings/tribal_bindings.nim"
  elif defined(macosx):
    exec "nim c --app:lib --mm:arc --opt:speed --outdir:bindings/generated --out:libtribal.dylib bindings/tribal_bindings.nim"
  else:
    exec "nim c --app:lib --mm:arc --opt:speed --outdir:bindings/generated --out:libtribal.so bindings/tribal_bindings.nim"
  
  echo "✅ Generated Python bindings in bindings/generated/"
  echo "✅ Files: tribal.py and libtribal.{so,dylib,dll}"

task run, "Run the tribal environment":
  exec "nim c -r src/tribal.nim"
