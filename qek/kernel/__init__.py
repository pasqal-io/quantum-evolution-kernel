"""
The Quantum Evolution Kernel itself, for use in a machine-learning pipeline.
"""

from .kernel import FastQEK, IntegratedQEK

# Alias, for backwards compatibility.
QuantumEvolutionKernel = FastQEK

__all__ = ["QuantumEvolutionKernel", "FastQEK", "IntegratedQEK"]
