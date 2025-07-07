# MettaScope – agent quick-start

MettaScope is a browser viewer to watch metta replays.

## Project layout

./mettascope/src/ has the frontend typescript source
./mettascope/server.py is a small python backend for serving replays

The code is plain TypeScript with no framework and is bundled by the TypeScript
compiler itself – no webpack or vite involved.

## Useful commands

All commands are executed from the `mettascope/` directory.

| Task | Command |
|------|---------|
| Build once | `npm run build` |
| Incremental build | `npm run watch` |
| Lint & format | `npm run lint` / `npm run format` |
| Full static checks | `npm run check` |

### Regenerating HTML/CSS

The HTML and CSS in the repo are generated from TS/JS sources.  After changing
UI templates run:

```bash
python tools/gen_html.py
```

## Commit guidelines

Follow the repository-wide rules described in `/AGENTS.md`:

* Short, present-tense commit messages
* Run lint/format before pushing

