## Mettabook - Research Analysis Notebooks

This is experimental. The hopes are that it makes analysis easier - providing a unified interface for launching training runs, monitoring them, and analyzing results all in one place.

More tools will be added over time, including integration with Observatory

We expect that this won't be perfect at first but want to gauge its viability. Please post in #researcher-tools for anything that is painful or that you need!

### How to Use

- Copy `mettabook-template.ipynb` to your own file in the research/ folder
- Check you're up to date with `metta install`
- Open your notebook in Cursor. When you open the file or try to run a cell (shift+enter by default), you should be prompted to select a Python kernel. Select the one from your uv-managed `.venv`

### Sharing results
- Commit relatively freely to the `research/` subfolder for now. Don't hold yourself to a high quality bar: these are not intended to b depended on
- Add tools you think others will also want to`utils/` files. These will undergo a more typical dev process
