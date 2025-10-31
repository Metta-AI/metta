version     = "0.1.0"
author      = "Metta Team"
description = "Standalone scripted Nim agent for CoGames"
license     = "MIT"

srcDir = "."

requires "nim >= 2.2.4"

import std/[os, strformat]

proc libExt(): string =
  when defined(windows):
    ".dll"
  elif defined(macosx):
    ".dylib"
  else:
    ".so"

task buildLib, "Build shared library for the scripted agent":
  let ext = libExt()
  let outName = "libcogames_agent" & ext
  let cmd = fmt"nim c --app:lib --mm:arc --opt:speed -d:danger --out:{outName} scripted_agent.nim"
  echo "Building CoGames scripted agent Nim library: ", cmd
  exec cmd
  echo "Built ", outName

task lib, "Alias for buildLib":
  exec "nimble buildLib"

before install:
  exec "nimble buildLib"

after install:
  echo "CoGames scripted agent library built."
