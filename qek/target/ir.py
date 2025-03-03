"""
Code emitted by compilation.

In practice, this code is a very thin layer around Pulser's representation.
"""

import pulser


class Pulse:
    """
    Specification of a laser pulse to be executed on a quantum device
    """

    def __init__(
        self,
        pulse: pulser.Pulse,
    ):
        """
        Specify a laser pulse
        """
        self.pulse = pulse

    def draw(self) -> None:
        """
        Draw the shape of this laser pulse.
        """
        self.pulse.draw()


class Register:
    """
    Specification of a geometry of atoms to be executed on a quantum device
    """

    def __init__(self, device: pulser.devices.Device, register: pulser.Register):
        """
        Specify a register
        """
        self.register = register
        self._device = device

    def __len__(self) -> int:
        """
        The number of qubits in this register.
        """
        return len(self.register.qubits)

    def draw(self) -> None:
        """
        Draw the geometry of this register.
        """
        self.register.draw(blockade_radius=self._device.min_atom_distance + 0.01)
