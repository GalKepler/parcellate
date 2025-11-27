from dataclasses import dataclass


@dataclass
class Statistic:
    """Container for a parcellation statistic."""

    name: str
    function: callable
    requires_image: bool = False
