class BusyError(RuntimeError):
    """A bounded executor or resource is busy."""


class GateError(RuntimeError):
    """An experiment or configuration boundary prevents this operation."""
