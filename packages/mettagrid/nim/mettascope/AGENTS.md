# Nim Coding Guidelines

See root `CLAUDE.md` and `STYLE_GUIDE.md` for general guidance. This file covers Nim-specific conventions.

## Abstractions

Please follow Handmade manifesto ideas of minimal abstraction, simple data structures, and linear straightforward code.
If function is called only one time, just inline it unless it's deeply nested. Use proper meta programming for the right
things.

- Try simple types.
- Only when types are not enough, try generics.
- Only when generics are not enough, try templates.
- Only when templates are not enough, try macros.

## Anatomy of a Nim file

Here is the anatomy of a Nim file:

- Imports
- Constants
- Types
- Variables
- Procedures
- when isMainModule (use rarely, never for tests)

## Imports

Imports should start with std modules then external modules, then local modules. Ideally in 3 lines like this:

```
import
  std/[os, random, strutils],
  fidget2, boxy, windy,
  common, internal, models, widgets.
```

Use plural for modules unless it's common.nim. If a module deals with Player, use `players.nim`. Always try to use
single English words for module names. Some modules will have `test_` or `bench_` prefix.

## Tests

Don't use unit test framework, use doAssert and echos instead. Testing is hard and they should be as simple, almost
stupid simple. Use a single tests/tests.nim file for all tests.

```nim
echo "Testing equality"
doAssert a == b, "a should be equal to b"
```

If it gets too big, split it into multiple files all starting with test\_.

After testing, benchmarking is just as important. Also write bench\_\*.nim files for benchmarks using benchy library.

```nim
import benchy, std/os, std/random

timeIt "number counter":
  var s = 0
  for i in 0 .. 1_000_000:
    s += s
```

## Names

Best names are single English words. Only go to two or three words if absolutely necessary. Use common abbreviations
like HTTP, API, JSON, etc. Use camelCase for variables and functions. Use PascalCase for types, constants, and enums.
Use plural for arrays and maps and other collections. When iterating and only when using integers prefer to use `i`,
`j`, `k` etc...

## Variables

At the top level prefer to use `const` over `let`. Note: in Nim const use CamelCase with capital first letter. Prefer to
use `let` over `var` unless you need to mutate the variable. Merge multiple const, let, and var declarations into a
single block declaration.

## Readme

Fix spelling and grammar, only! Avoid using emoji in the readme, avoid using fancy quotes, mdash, semicolon, and other
fancy characters. Write in a simple, clear, and direct way. Bullet lists or table to show features are good.

## Indentation

Use 2 spaces for indentation. Never use double lines even between types, procs or sections. If breaking a large function
call break it into a line per argument.

```nim
func(
  arg1,
  arg2,
  arg3
)
```

If body of a if or loop is too large, break it into a line per statement, but then indent the body by 4 spaces.

```nim
if condition or
  longCondition or
  anotherLongCondition:
    statement1
    statement2
    statement3
```

Don't indent the body of a case statement. Prefer to use enums and case statements together.

```nim
case expression:
of value1:
  statement1
of value2:
  statement2
else:
  statement4
```

## Comments

Have all comments be complete sentences. Start with a capital letter and end with a period. Make sure all functions have
doc comments. Try to only use a single line per doc comment. Never more than 4 lines. Avoid top level section comments,
especially surround with `=` or `#` characters.

## Error Handling

Its best to let the exception propagate to the top level. Don't silence them with `try/except`. Use error codes where
absolutely necessary. Adding asserts especially at start or end of a procedure is good as they can be compiled out in
release mode. Many errors are not actually errors and can be passed through. Prefer returning nil, "", 0, false, over
raising exceptions or error codes.

## Checking the code

In many projects you can run `nim check` and `nimble test` as you are writing the code to make sure it works. Always do
this after big changes and before committing.

## Block formatting

Some small or repeating functions its ok to be without a doc comment.

```nim
proc toFlatty*(s: var string, x: uint8) = s.addUint8(x)
proc toFlatty*(s: var string, x: int8) = s.addInt8(x)
proc toFlatty*(s: var string, x: uint16) = s.addUint16(x)
proc toFlatty*(s: var string, x: int16) = s.addInt16(x)
proc toFlatty*(s: var string, x: uint32) = s.addUint32(x)
proc toFlatty*(s: var string, x: int32) = s.addInt32(x)
proc toFlatty*(s: var string, x: uint64) = s.addUint64(x)
proc toFlatty*(s: var string, x: int64) = s.addInt64(x)
proc toFlatty*(s: var string, x: float32) = s.addFloat32(x)
proc toFlatty*(s: var string, x: float64) = s.addFloat64(x)
```

Some functions that are just use to interface with external libraries its ok to be without a doc comment and be on long
lines.

```nim
proc WinHttpReceiveResponse*(hRequest: HINTERNET, lpReserved: LPVOID): BOOL {.dynlib: "winhttp".}
proc WinHttpQueryHeaders*(hRequest: HINTERNET, dwInfoLevel: DWORD, pwszName: LPCWSTR, lpBuffer: LPVOID, lpdwBufferLength: LPDWORD, lpdwIndex: LPDWORD): BOOL {.dynlib: "winhttp".}
proc WinHttpReadData*(hFile: HINTERNET, lpBuffer: LPVOID, dwNumberOfBytesToRead: DWORD, lpdwNumberOfBytesRead: LPDWORD): BOOL {.dynlib: "winhttp".}
```

Its ok to add zero to make blocks line up:

```
for i in 0 ..< 10:
  echo data[i * 3 + 0]
  echo data[i * 3 + 1]
  echo data[i * 3 + 2]
```
