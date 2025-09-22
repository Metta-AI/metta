# Mettascope2

## Building

Install Nim `2.2.4` if you haven't already.
* [Mac] `brew install nim`
* [Linux] `sudo apt install nim`
* [Windows] [Downloads](https://nim-lang.org/install.html)

Make sure you are using Nim `2.2.4`.
```
nim --version
```

Build the dynamic link library:

```
cd mettascope2
nimble install
nim c -d:release --app:lib --gc:arc --tlsEmulation:off --out:libmettascope2.dylib --outdir:bindings/generated bindings/bindings.nim
```

## Running

```
./tools/run.py experiments.recipes.arena.play mettascope2=true
```
