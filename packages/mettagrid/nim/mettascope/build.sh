mkdir -p bindings/generated
nimble install -y
nimble c -y -d:release --app:lib --gc:arc -d:fidgetUseCached=true --tlsEmulation:off --out:libmettascope2.dylib --outdir:bindings/generated bindings/bindings.nim
