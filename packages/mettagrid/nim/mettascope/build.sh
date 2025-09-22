mkdir -p bindings/generated
nimble install
nim c -d:release --app:lib --gc:arc --tlsEmulation:off --out:libmettascope2.dylib --outdir:bindings/generated bindings/bindings.nim
