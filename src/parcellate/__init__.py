"""Atlas-based volumetric parcellation of scalar neuroimaging maps.

This package provides tools for extracting regional statistics from volumetric
brain images using atlas-based parcellation. It supports multiple neuroimaging
pipelines (CAT12, QSIRecon) and provides flexible Python and CLI interfaces.
"""

from parcellate.parcellation import VolumetricParcellator

__all__ = ["VolumetricParcellator"]
