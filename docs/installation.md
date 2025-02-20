# Installation
## Using `hatch`, `uv` or any pyproject-compatible Python manager

Edit file `pyproject.toml` to add the line

```toml
  "quantum-evolution-kernel"
```

to the list of `dependencies`.

## Using `pip` or `pipx`
To install the `pipy` package using `pip` or `pipx`

1. Create a `venv` if that's not done yet

```sh
$ python -m venv venv

```

2. Enter the venv

```sh
$ . venv/bin/activate
```

3. Install the package

```sh
$ pip install quantum-evolution-kernel
# or
$ pipx install quantum-evolution-kernel
```
