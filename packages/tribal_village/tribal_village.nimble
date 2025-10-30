version     = "0.1.0"
author      = "Metta Team"
description = "High-performance tribal-village environment for multi-agent RL"
license     = "MIT"

srcDir = "src"

requires "nim >= 2.2.4"
requires "vmath >= 2.0.0"
requires "chroma >= 0.2.7"
requires "boxy"
requires "windy"

import std/[os, strformat, strutils]

proc ensureDir(path: string) =
  if dirExists(path):
    return
  when defined(windows):
    exec "cmd /c mkdir " & quoteShell(path)
  else:
    exec "mkdir -p " & quoteShell(path)

task buildLib, "Build shared library for PufferLib":
  echo "Building Tribal Village shared library (ultra-fast direct buffers)..."

  let ext = when defined(windows): "dll"
            elif defined(macosx): "dylib"
            else: "so"

  exec "nim c --app:lib --mm:arc --opt:speed -d:danger --out:libtribal_village." & ext & " src/tribal_village_interface.nim"
  echo "Built libtribal_village." & ext & " with ultra-fast direct buffers"

task run, "Run the tribal village game":
  exec "nim c -r tribal_village.nim"

task lib, "Build shared library for PufferLib (alias for buildLib)":
  exec "nimble buildLib"

task wasm, "Build Tribal Village WASM demo":
  let
    root = getCurrentDir()
    outDir = root / "build" / "web"
    nimcacheDir = outDir / "nimcache"
    emCacheDir = outDir / "emscripten_cache"
    shellFileRel = "scripts/shell_minimal.html"
    shellFile = root / shellFileRel
    htmlOutRel = "build/web/tribal_village.html"
    htmlOut = root / htmlOutRel
    nimcacheRel = "build/web/nimcache"

  ensureDir(outDir)
  ensureDir(nimcacheDir)
  ensureDir(emCacheDir)

  if not fileExists(shellFile):
    raise newException(OSError, &"Missing Emscripten shell file at {shellFile}.")

  putEnv("EM_CACHE", emCacheDir)

  var cmdParts = @["nim", "c"]

  cmdParts.add("--app:gui")
  cmdParts.add("--threads:off")
  cmdParts.add("--gc:arc")
  cmdParts.add("--exceptions:goto")
  cmdParts.add("--define:noSignalHandler")
  cmdParts.add("--os:linux")
  cmdParts.add("--cpu:wasm32")
  cmdParts.add("--cc:clang")
  cmdParts.add("--nimcache:" & nimcacheRel)
  cmdParts.add("--listCmd")
  cmdParts.add("-d:release")
  cmdParts.add("-d:emscripten")
  cmdParts.add("-d:nimNoDevRandom")
  cmdParts.add("-d:nimNoGetRandom")
  cmdParts.add("-d:nimNoSysrand")
  cmdParts.add("-o:" & htmlOutRel)

  when defined(windows):
    cmdParts.add("--clang.exe:emcc.bat")
    cmdParts.add("--clang.linkerexe:emcc.bat")
    cmdParts.add("--clang.cpp.exe:emcc.bat")
    cmdParts.add("--clang.cpp.linkerexe:emcc.bat")
  else:
    cmdParts.add("--clang.exe:emcc")
    cmdParts.add("--clang.linkerexe:emcc")
    cmdParts.add("--clang.cpp.exe:emcc")
    cmdParts.add("--clang.cpp.linkerexe:emcc")

  let passLFlags = [
    "--shell-file=" & shellFileRel,
    "--preload-file data",
    "-sUSE_GLFW=3",
    "-sUSE_WEBGL2=1",
    "-sASYNCIFY",
    "-sALLOW_MEMORY_GROWTH",
    "-sINITIAL_MEMORY=512MB",
    "-sFULL_ES3=1",
    "-sGL_ENABLE_GET_PROC_ADDRESS=1",
    "-sERROR_ON_UNDEFINED_SYMBOLS=0"
  ]

  for flag in passLFlags:
    cmdParts.add("--passL:\"" & flag & "\"")

  cmdParts.add("tribal_village.nim")

  exec cmdParts.join(" ")

before install:
  exec "nimble buildLib"

after install:
  echo "Tribal Village installation complete with shared library built!"
