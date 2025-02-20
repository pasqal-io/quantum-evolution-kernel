"""
Exceptions raised within this library.
"""


class CompilationError(Exception):
    """
    An error raised when attempting to compile a graph for an architecture
    that does not support it, e.g. because it requires too many qubits or
    because the physical constraints on the geometry are not satisfied.
    """
