from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Statistic:
    """Container for a parcellation statistic."""

    name: str
    function: Callable[..., Any]
    requires_image: bool = False
