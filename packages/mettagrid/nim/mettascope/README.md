# Mettascope

## Building

Install Nim `2.2.4` if you haven't already.

- [Mac] `brew install nim`
- [Linux] `sudo apt install nim`
- [Windows] [Downloads](https://nim-lang.org/install.html)

Make sure you are using Nim `2.2.4`.

```
nim --version
```

Build the dynamic link library:

```
cd mettascope
./build.sh
```

## Running

```
./tools/run.py arena.play mettascope=true
```
