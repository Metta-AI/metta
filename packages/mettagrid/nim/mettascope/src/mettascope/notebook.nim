import
  std/[strutils],
  replays, common, replayloader

when defined(emscripten):
  {.emit: """
  #include <emscripten.h>
  """.}

  {.emit: """
  EM_JS(void, setup_postmessage_replay_handler_internal, (void* userData), {
    // Check if the message origin is from an allowed source.
    function isValidOrigin(origin) {
      // Google Colab uses *.googleusercontent.com domains.
      if (origin.includes('colab') && origin.includes('googleusercontent.com')) {
        return true;
      }
      // Localhost and 127.0.0.1 variants (http and https, any port).
      if (origin.startsWith('http://localhost:') || origin.startsWith('https://localhost:')
        || origin.startsWith('http://127.0.0.1:') || origin.startsWith('https://127.0.0.1:')) {
        return true;
      }
      return false;
    }

    // notify the iframe parent that we are ready to receive replay data.
    window.parent.postMessage({ type: 'mettascopeReady' }, '*');

    // Listen for postMessage events from parent windows (Jupyter notebooks).
    window.addEventListener('message', function(event) {
      if (!isValidOrigin(event.origin) || !event.data || event.data.type !== 'replayData') {
        return;
      }

      const base64Data = event.data.base64;
      if (!base64Data || typeof base64Data !== 'string') {
        return;
      }

      try {
        // Decode base64 to binary string.
        const binaryString = atob(base64Data);
        const binaryLength = binaryString.length;
        const binaryPtr = _malloc(binaryLength);
        if (!binaryPtr) return;

        // Copy binary data to heap.
        for (let i = 0; i < binaryLength; i++) {
          HEAPU8[binaryPtr + i] = binaryString.charCodeAt(i);
        }

        // Allocate and copy filename.
        const fileName = event.data.fileName || 'replay_from_notebook.json.z';
        const fileNameLen = lengthBytesUTF8(fileName) + 1;
        const fileNamePtr = _malloc(fileNameLen);
        stringToUTF8(fileName, fileNamePtr, fileNameLen);

        // Call the Nim callback with the replay data.
        Module._mettascope_postmessage_replay_callback(userData, fileNamePtr, binaryPtr, binaryLength);

        // Free allocated memory.
        _free(fileNamePtr);
        _free(binaryPtr);
      } catch (error) {
        console.error('Error processing postMessage replay data:', error);
      }
    });
  });
  """.}

  proc setup_postmessage_replay_handler_internal*(userData: pointer) {.importc.}
    ## Set up postMessage listener for receiving replay data from Jupyter notebooks.

  proc mettascope_postmessage_replay_callback(userData: pointer, fileNamePtr: cstring, binaryPtr: pointer, binaryLen: cint) {.exportc, cdecl, codegenDecl: "EMSCRIPTEN_KEEPALIVE $# $#$#".} =
    ## Callback to handle postMessage replay data from JavaScript.
    ## EMSCRIPTEN_KEEPALIVE is required to avoid dead code elimination.
    let fileName = $fileNamePtr
    var fileData = newString(binaryLen)
    if binaryLen > 0:
      copyMem(fileData[0].addr, binaryPtr, binaryLen)
    
    if fileName.endsWith(".json.gz") or fileName.endsWith(".json.z"):
      try:
        common.replay = loadReplay(fileData, fileName)
        onReplayLoaded()
        echo "Loaded replay from postMessage: ", fileName
      except:
        echo "Error loading replay: ", getCurrentExceptionMsg()

  proc setupPostMessageReplayHandler*(userData: pointer) =
    ## Set up postMessage handler for receiving replay data from Jupyter notebooks.
    when defined(emscripten):
      setup_postmessage_replay_handler_internal(userData)

