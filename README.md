# Quantum Evolution Kernel

A Python library that implements Quantum Evolution Kernel, a method for measuring the
similarity between graph-structured data, based on the time-evolution of a quantum system.
It includes an example application in machine learning classification on a bio-chemical dataset.

For more details about the approach, see https://journals.aps.org/pra/abstract/10.1103/PhysRevA.107.042615

## Installation

### Using `hatch`, `uv` or any pyproject-compatible Python manager

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
[licensing@pasqal.com](mailto:licensing@pasqal.com).
