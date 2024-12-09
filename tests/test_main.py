from __future__ import annotations

from qek.main import main

expected_msg = "Welcome to Quantum Evolution Kernel!"


def test_main() -> None:
    msg = main()
    assert msg == expected_msg


def test_main_with_str() -> None:
    str_to_add = "\nExecuted from test file"
    msg = main(str_to_add=str_to_add)
    assert msg == expected_msg + str_to_add
