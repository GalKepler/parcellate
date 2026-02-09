"""
Utility functions for image handling
"""

from pathlib import Path
from typing import Union

import nibabel as nib


def _load_nifti(img: Union[nib.Nifti1Image, str, Path]) -> nib.Nifti1Image:
    """
    Safe load an image

    Parameters
    ----------
    img : Union[nib.Nifti1Image, str, Path]
        Image to load.

    Returns
    -------
    nib.Nifti1Image
        Loaded image
    """
    if isinstance(img, nib.Nifti1Image):
        return img
    return nib.Nifti1Image.from_filename(str(img))
