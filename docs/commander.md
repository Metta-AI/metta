
# Commander iParse and apply command line arguments.

This function takes a command line arguments and applies them to a possibly very nested and complex tree of objects.

A command line of:

`--foo.bar.baz 3`

Will take the following tree, and apply the command line arguments to it in place.


```json
{
  "foo": {
    "bar": {
      "baz": 3
    }
  }
}
```

But it only works if the tree had the keys already, otherwise it will be an error.

Both `-foo.bar.baz 3`, `--foo.bar.baz=3`, and `--foo.bar.baz:3` are valid command line arguments. As they all have been used for command line arguments in the past.

Strings with spaces are supported for both quotes `"` and `'`.

`--foo.bar "hello world"` will set `{foo: {bar: "hello world"}}`
`--foo.bar 'hello world'` will set `{foo: {bar: "hello world"}}`

Strings with escape characters are supported:

`--foo.bar "hello\nworld"` will set `{foo: {bar: "hello\nworld"}}`

Strings with quotes are supported:

`--foo.bar "hello \"world\""` will set `{foo: {bar: "hello \"world\""}}`

Plain words are treated as strings:

`--foo.bar hello` will set `{foo: {bar: "hello"}}`

But if a strings starts with a non letter character, it needs to be quoted.

`--foo.bar.baz @foo` needs to be quoted: `--foo.bar.baz '@foo'`

Keys without a value are treated as booleans.

`--foo.bar` will set `{foo: {bar: true}}`

Numbers can start with a `+` or `-` sign. Special care needs to be taken for negative numbers as they look like other parameters.

`--foo.bar -1` will set `{foo: {bar: -1}}`
`--foo.bar +1` will set `{foo: {bar: 1}}`

Numbers can have a decimal point and scientific notation.

`--foo.bar 1.23456789` will set `{foo: {bar: 1.23456789}}`
`--foo.bar 1.23456789e10` will set `{foo: {bar: 1.23456789e10}}`

Numbers in the key are treated as indexes so:

`--array.1 7` Can set `{array: [0, 0, 0]}` to `{array: [0, 7, 0]}`

It also checks the the type of the inserted value matches the type that was there before.

The error message are really helpful. It gives you what it was trying to set and why it failed.

Passing a special `--help` or `-h` argument will print out the whole tree with the types and doc comments.
