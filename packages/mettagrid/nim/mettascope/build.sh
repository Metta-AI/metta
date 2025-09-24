mkdir -p bindings/generated
nimble install
nim c -d:release --app:lib --gc:arc -d:fidgetUseCached=true --tlsEmulation:off --out:libmettascope2.dylib --outdir:bindings/generated bindings/bindings.nim
