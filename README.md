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

See also the [full API documentation](https://pqs.pages.pasqal.com/quantum-evolution-kernel/).

## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [GitHub Repository](https://github.com/pasqal-io/quantum-evolution-kernel) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)
