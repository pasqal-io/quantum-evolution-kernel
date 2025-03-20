# First contact

## Getting the code

The code is available on GitHub.

To clone it

```sh
$ git clone git@github.com:pasqal-io/quantum-evolution-kernel.git
```

or

```sh
$ git clone https://github.com/pasqal-io/quantum-evolution-kernel
```

## Hatch instructions

We use `hatch` and Python 3.10 for development.

### Setting up

With Python and pip installed, to setup the environment:

```sh
$ pip install hatch
$ hatch -v shell
```

This will open a shell with all the dependencies installed.

### Running tests

To run the unit and integration tests

```sh
$ hatch run test
```

To run linters

```sh
$ hatch run pre-commit run --all-files
```
