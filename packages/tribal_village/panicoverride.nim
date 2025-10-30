when defined(emscripten):
  proc consoleLog(msg: cstring) {.importc: "console.log".}
  proc consoleError(msg: cstring) {.importc: "console.error".}

  proc rawOutput(msg: string) =
    consoleLog(msg.cstring)

  proc panic(message: string) =
    consoleError(("Nim panic: " & message).cstring)
    while true:
      discard
else:
  {.warning: "panicoverride only active for emscripten builds".}
