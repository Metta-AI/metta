## Mettabook - Research Analysis Notebooks

This is experimental. The hopes are that it makes analysis easier - providing a unified interface for launching training
runs, monitoring them, and analyzing results all in one place.

More tools will be added over time, including integration with Observatory

We expect that this won't be perfect at first but want to gauge its viability. Please post in #researcher-tools for
anything that is painful or that you need!

### Setup

1. **Install notebook support**:

   ```bash
   metta install notebook
   ```

   Then `metta status` to check that it worked.

2. **In Cursor or VS Code, select the correct environment**:

   When you open a notebook file (`.ipynb` file), you'll need to choose a python environment for it. Just pick the same
   environment you see listed in `metta status`, e.g. ".venv (Python 3.11.7)"

### Sharing results

- Feel free to make copies of an existing notebook such as `mettabook-template.ipynb` into your own file in the
  `research/` folder
- Commit relatively freely to the `research/` subfolder for now. Don't hold yourself to a high quality bar: these are
  not intended to be depended on
- Add tools you think others will also want to`utils/` files. These will undergo a more typical dev process
