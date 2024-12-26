# Quantum Evolution Kernel

This is a Python library that implements embedding graphs as _quantum kernels_ for
the purpose of machine learning and classification on analog Quantum Processing Units.

For more details about the approach, see https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042615

## Installation

### Using `uv`

To add `pasqal-qek` as a dependency using the `uv` package manager:

```sh
$ uv add pasqal-qek
```

### Using any other pyproject-compatible Python manager

Edit file `pyproject.toml` to add the line

```toml
  "pasqal-qek"
```

to the list of `dependencies`.

### Using `pip` or `pipx`
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
$ pip install pasqal-qek
# or
$ pipx install pasqal-qek
```

## Usage
For the time being, the easiest way to learn how to use this package is to look
at the [example notebook](examples/pipeline.ipynb).

## Going further

If you need professional support, an industrial license or a variant of this library
optimized for your workload, don't hesitate to drop us a line at
[mailto:licensing@pasqal.com](licensing@pasqal.com).
