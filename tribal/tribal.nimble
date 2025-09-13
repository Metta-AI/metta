version     = "0.1.0"
author      = "Metta Team"
description = "High-performance tribal environment for multi-agent RL"
license     = "MIT"

srcDir = "src"

requires "nim >= 2.2.4"
requires "genny >= 0.1.0"
requires "nimpy >= 0.2.0"
requires "pixie >= 5.0.0"
requires "vmath >= 2.0.0"
requires "chroma >= 0.2.7"
requires "boxy >= 0.1.4"
requires "windy >= 0.1.2"

task bindings, "Generate Python bindings for tribal environment":
  # Generate the bindings
  exec "nim r bindings/tribal_bindings.nim"
  
  # Create the shared library (use -d:danger for maximum speed in production)
  let buildFlags = if existsEnv("TRIBAL_DANGER"): 
    "--app:lib --mm:arc -d:danger" 
  else: 
    "--app:lib --mm:arc --opt:speed"
    
  when defined(windows):
    exec "nim c " & buildFlags & " --outdir:bindings/generated --out:tribal.dll bindings/tribal_bindings.nim"
  elif defined(macosx):
    exec "nim c " & buildFlags & " --outdir:bindings/generated --out:libtribal.dylib bindings/tribal_bindings.nim"
  else:
    exec "nim c " & buildFlags & " --outdir:bindings/generated --out:libtribal.so bindings/tribal_bindings.nim"
  
  echo "✅ Generated Python bindings in bindings/generated/"
  echo "✅ Files: tribal.py and libtribal.{so,dylib,dll}"
  echo ""
  if existsEnv("TRIBAL_DANGER"):
    echo "⚡ DANGER MODE: Maximum speed, no safety checks!"
    echo "⚠️  Errors will segfault - use only for production training"
  else:
    echo "🔧 SAFE MODE: Good performance with safety checks"
    echo "💡 Use TRIBAL_DANGER=1 nimble bindings for maximum speed"

task run, "Run the tribal environment":
  exec "nim c -r src/tribal/environment.nim"

task visualize, "Run the tribal visualization":
  exec "nim c -r src/tribal.nim"

task play_nim, "Run tribal play interface from Nim":
  exec "nim c -r src/tribal_play.nim"

task test, "Run tribal environment tests":
  exec "nim c -r tests/test_python_bindings.nim"

task clean, "Clean build artifacts":
  exec "rm -rf bindings/generated/"
  echo "✅ Cleaned build artifacts"