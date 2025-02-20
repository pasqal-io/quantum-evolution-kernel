# "Quantum Evolution Kernel"
## Installation
### Using `hatch`, `uv` or any pyproject-compatible Python manager

Edit file `pyproject.toml` to add the line

```toml
  "quantum-evolution-kernel"
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
$ pip install quantum-evolution-kernel
# or
$ pipx install quantum-evolution-kernel
```

## Usage

```python
# Load a dataset
import torch_geometric.datasets as pyg_dataset
og_ptcfm = pyg_dataset.TUDataset(root="dataset", name="PTC_FM")

# Setup a quantum feature extractor for this dataset.
# In this example, we'll use QutipExtractor, to emulate a Quantum Device on our machine.
import qek.data.graphs as qek_graphs
import qek.data.extractors as qek_extractors
extractor = qek_extractors.QutipExtractor(compiler=qek_graphs.PTCFMCompiler())

# Add the graphs, compile them and look at the results.
extractor.add_graphs(graphs=og_ptcfm)
extractor.compile()
processed_dataset = extractor.run().processed_data

# Prepare a machine learning pipeline with Scikit Learn.
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X = [data for data in processed_dataset]  # Features
y = [data.target for data in processed_dataset]  # Targets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size=0.2, random_state=42)

# Train a kernel
from qek.kernel import QuantumEvolutionKernel as QEK
kernel = QEK(mu=0.5)
model = SVC(kernel=kernel, random_state=42)
model.fit(X_train, y_train)
```

## Documentation

::: qek

## Tutorials

We have a two parts tutorial:

1. [Using a Quantum Device to extract machine-learning features](https://github.com/pasqal-io/quantum-evolution-kernel/blob/main/examples/tutorial%201%20-%20Using%20a%20Quantum%20Device%20to%20Extract%20Machine-Learning%20Features.ipynb);
2. [Machine Learning with the Quantum Evolution Kernel](https://github.com/pasqal-io/quantum-evolution-kernel/blob/main/examples/tutorial%202%20-%20Machine-Learning%20with%20the%20Quantum%20EvolutionKernel.ipynb)

## Getting in touch

- [Pasqal Community Portal](https://community.pasqal.com/) (forums, chat, tutorials, examples, code library).
- [GitHub Repository](https://github.com/pasqal-io/quantum-evolution-kernel) (source code, issue tracker).
- [Professional Support](https://www.pasqal.com/contact-us/) (if you need tech support, custom licenses, a variant of this library optimized for your workload, your own QPU, remote access to a QPU, ...)

## Contribute

The GitHub repository is open for contributions!

Don't forget to read the [Contributor License Agreement]().
