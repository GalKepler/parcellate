from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar, Literal

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.datasets import (
    load_mni152_brain_mask,
    load_mni152_gm_mask,
    load_mni152_wm_mask,
)
from nilearn.image import resample_to_img

from parcellate.metrics.base import Statistic
from parcellate.metrics.volume import BUILTIN_STATISTICS
from parcellate.utils import _load_nifti

logger = logging.getLogger(__name__)


class MissingLUTColumnsError(ValueError):
    def __init__(self, missing):
        super().__init__(f"Lookup table is missing required columns: {missing}")


class AtlasShapeError(ValueError):
    def __init__(self, message: str = "Atlas image must be 3D."):
        super().__init__(message)


class MissingStatisticalFunctionError(ValueError):
    def __init__(self, message: str = "At least one statistical function must be provided."):
        super().__init__(message)


class ParcellatorNotFittedError(RuntimeError):
    def __init__(self, message: str = "Parcellator must be fitted before calling transform()."):
        super().__init__(message)


class VolumetricParcellator:
    """Base volumetric parcellator.

    The parcellator assumes an integer-valued atlas where each non-background
    voxel stores the parcel identifier. Parcellation is performed by sampling a
    scalar map image and aggregating values inside each region. Resampling to
    the atlas grid is handled by default to keep atlas boundaries consistent
    across inputs.
    """

    REQUIRED_LUT_COLUMNS: ClassVar[set[str]] = {"index", "label"}
    BUILTIN_STANDARD_MASKS: ClassVar[Mapping[str, str]] = {
        "gm": load_mni152_gm_mask,
        "wm": load_mni152_wm_mask,
        "brain": load_mni152_brain_mask,
    }

    def __init__(
        self,
        atlas_img: nib.Nifti1Image | str | Path,
        labels: Mapping[int, str] | Sequence[str] | None = None,
        lut: pd.DataFrame | str | Path | None = None,
        *,
        mask: nib.Nifti1Image | str | Path | None = None,
        mask_threshold: float = 0.0,
        background_label: int = 0,
        resampling_target: Literal["data", "labels", "atlas", None] = "data",
        stat_functions: Mapping[str, Callable[..., float]] | None = None,
    ) -> None:
        """
        Initialize a volumetric parcellator

        Parameters
        ----------
        atlas_img : nib.Nifti1Image | str | Path
            The atlas image defining the parcellation.
        labels : Mapping[int, str] | Sequence[str] | None, optional
            Region labels mapping or sequence, by default None
        lut : pd.DataFrame | str | Path | None, optional
            Lookup table for region labels, by default None. Must include columns
            "index" and "label" following the BIDS standard.
        mask : nib.Nifti1Image | str | Path | None, optional
            Optional mask to apply to the atlas, by default None
        mask_threshold : float, optional
            Threshold for the mask image. Voxels with mask values strictly greater
            than this threshold are included; all others are excluded. Default is
            ``0.0``, which preserves the original behaviour of including every
            non-zero voxel. Use values in [0, 1] for probability maps (e.g.
            ``0.5`` to keep only voxels with >50 % gray-matter probability).
        background_label : int, optional
            Label value to treat as background, by default 0
        resampling_target : Literal["data", "labels", None], optional
            Resampling target for input maps, by default "data"
        stat_functions : Mapping[str, StatFunction] | None, optional
            Mapping of statistic names to functions, by default None
        """
        self.atlas_img = _load_nifti(atlas_img)
        self.lut = self._load_atlas_lut(lut) if lut is not None else None
        self.mask_threshold = float(mask_threshold)
        self.mask = self._load_mask(mask) if mask is not None else None
        self.background_label = int(background_label)
        self.resampling_target = resampling_target
        self._atlas_data = self._load_atlas_data()
        self._regions = self._build_regions(labels)
        self._stat_functions = self._prepare_stat_functions(stat_functions)
        self._fitted_scalar_id: str | int | None = None

    def _load_mask(self, mask: nib.Nifti1Image | str | Path) -> nib.Nifti1Image:
        """
        Load a mask image, supporting built-in standard masks.

        Parameters
        ----------
        mask : nib.Nifti1Image | str | Path
            Mask image to load.

        Returns
        -------
        nib.Nifti1Image
            Loaded mask image.
        """
        if isinstance(mask, str) and mask in self.BUILTIN_STANDARD_MASKS:
            return self.BUILTIN_STANDARD_MASKS[mask](threshold=self.mask_threshold)
        return _load_nifti(mask)

    def _get_labels(self, labels: Mapping[int, str] | Sequence[str] | None) -> list[int]:
        """
        Get labels from those required by the user and the ones from the lut/image

        Parameters
        ----------
        labels : Mapping[int, str] | Sequence[str] | None
            Labels provided by the user.

        Returns
        -------
        list[int]
            List of labels to use.
        """
        if labels is not None:
            if isinstance(labels, Mapping):
                return list(labels.keys())
            elif isinstance(labels, Sequence):
                return list(labels)
        if self.lut is not None:
            return self.lut["index"].tolist()
        return list(np.unique(self._atlas_data[self._atlas_data != self.background_label]).astype(int))

    def _load_atlas_lut(self, lut: pd.DataFrame | str | Path) -> pd.DataFrame:
        """
        Load atlas lookup table and make sure it contains required columns

        Parameters
        ----------
        lut : pd.DataFrame | str | Path
            Lookup table to load.

        Returns
        -------
        pd.DataFrame
            Loaded lookup table.

        Raises
        ------
        ValueError
            If required columns are missing.
        """
        lut_df = lut if isinstance(lut, pd.DataFrame) else pd.read_csv(lut, sep="\t")
        required_columns = self.REQUIRED_LUT_COLUMNS
        if not required_columns.issubset(lut_df.columns):
            missing = required_columns - set(lut_df.columns)
            raise MissingLUTColumnsError(missing)

        return lut_df

    @property
    def regions(self) -> tuple[int, ...]:
        """Tuple of regions defined in the atlas."""
        return self._regions

    def _apply_mask_to_atlas(self) -> nib.Nifti1Image:
        """
        Apply masking to parcellation atlas.

        Returns
        -------
        nib.Nifti1Image
            Masked atlas image.
        """
        atlas_data = np.asarray(self._prepared_atlas_img.get_fdata())
        mask_data = np.asarray(self._prepared_mask.get_fdata()) > self.mask_threshold
        atlas_data[~mask_data] = self.background_label
        return nib.Nifti1Image(atlas_data, self._prepared_atlas_img.affine, self._prepared_atlas_img.header)

    def _prepare_map(
        self,
        source: nib.Nifti1Image,
        reference: nib.Nifti1Image,
        interpolation: str = "nearest",
    ) -> nib.Nifti1Image:
        """Resample source image to reference image grid if needed.

        Parameters
        ----------
        source : nib.Nifti1Image
            Source image to resample.
        reference : nib.Nifti1Image
            Reference image defining the target grid.

        Returns
        -------
        nib.Nifti1Image
            Resampled image.
        """
        return resample_to_img(
            source,
            reference,
            interpolation=interpolation,
            force_resample=True,
            copy_header=True,
        )

    def _build_regions(self, labels: Mapping[int, str] | Sequence[str] | None) -> tuple[int, ...]:
        """
        Build region definitions from atlas data and optional labels.

        Parameters
        ----------
        labels : Mapping[int, str] | Sequence[str] | None
            Optional labels provided by the user.
        """
        atlas_ids = set(self._get_labels(labels))
        atlas_ids.discard(self.background_label)
        return tuple(sorted(atlas_ids))

    def _load_atlas_data(self) -> np.ndarray:
        atlas_data = np.asarray(self.atlas_img.get_fdata())
        if atlas_data.ndim != 3:
            raise AtlasShapeError()
        return atlas_data.astype(int)

    def _prepare_stat_functions(
        self,
        stat_functions: Mapping[str, Callable[..., float]] | None = None,
        *,
        fallback: Mapping[str, Callable[..., float]] | None = None,
    ) -> list[Statistic]:
        """
        Generate a list of summary statistics to describe each ROI

        Parameters
        ----------
        stat_functions : Mapping[str, Callable[..., float]] | None
            Either a mapping of statistic names to functions or None to use defaults.
        fallback : Mapping[str, Callable[..., float]] | None, optional
            Fallback statistics to use if stat_functions is None, by default None

        Returns
        -------
        list[Statistic]
            List of prepared statistics.

        Raises
        ------
        ValueError
            If no statistical functions are provided.
        """
        if stat_functions is None:
            if fallback is None:
                return BUILTIN_STATISTICS
            return fallback
        if isinstance(stat_functions, Mapping):
            prepared = [Statistic(name, func) for name, func in stat_functions.items()]
        elif isinstance(stat_functions, Sequence):
            if all(isinstance(s, Statistic) for s in stat_functions):
                prepared = list(stat_functions)
            else:
                raise MissingStatisticalFunctionError(message="Statistic sequence must contain Statistic instances.")
        else:
            raise MissingStatisticalFunctionError(
                message=f"stat_functions must be a Mapping or Sequence, got {type(stat_functions).__name__}"
            )

        if not prepared or len(prepared) == 0:
            raise MissingStatisticalFunctionError()
        return prepared

    def fit(self, scalar_img: nib.Nifti1Image | str | Path) -> None:
        """
        Fit the parcellator to a scalar image.

        Parameters
        ----------
        scalar_img : nib.Nifti1Image | str | Path
            Scalar image to fit to.
        """
        self.scalar_img = _load_nifti(scalar_img)
        if self.resampling_target in ("labels", "atlas"):
            ref_img = self.atlas_img
            interpolation = "continuous"
        else:
            ref_img = self.scalar_img
            interpolation = "nearest"

        # Cache atlas resampling by reference image identity (optimization 4.2)
        ref_id = id(ref_img)
        if not hasattr(self, "_cached_atlas_ref_id") or self._cached_atlas_ref_id != ref_id:
            self._prepared_atlas_img = self._prepare_map(self.atlas_img, ref_img, interpolation="nearest")
            self._cached_atlas_ref_id = ref_id
            # Reset mask cache when atlas cache is invalidated
            if hasattr(self, "_cached_mask_ref_id"):
                delattr(self, "_cached_mask_ref_id")

        self._prepared_scalar_img = self._prepare_map(self.scalar_img, ref_img, interpolation=interpolation)

        # Cache mask resampling by reference image identity (optimization 4.1)
        if self.mask is not None:
            if not hasattr(self, "_cached_mask_ref_id") or self._cached_mask_ref_id != ref_id:
                self._prepared_mask = self._prepare_map(self.mask, ref_img, interpolation="nearest")
                self._cached_mask_ref_id = ref_id
            self._prepared_atlas_img = self._apply_mask_to_atlas()

        self.ref_img = ref_img
        if isinstance(scalar_img, (str, Path)):
            self._fitted_scalar_id = str(scalar_img)
        else:
            self._fitted_scalar_id = id(scalar_img)

    def transform(self, scalar_img: str | Path | nib.Nifti1Image) -> pd.DataFrame:
        """
        Apply the parcellation to the fitted scalar image.

        Returns
        -------
        pd.DataFrame
            DataFrame containing parcellation statistics for each region.
        """
        if not hasattr(self, "_prepared_atlas_img") or not hasattr(self, "_prepared_scalar_img"):
            raise ParcellatorNotFittedError()

        current_scalar_id: str | int = str(scalar_img) if isinstance(scalar_img, (str, Path)) else id(scalar_img)

        if self._fitted_scalar_id == current_scalar_id:
            prepared_scalar_img = self._prepared_scalar_img
        else:
            prepared_scalar_img = self._prepare_map(
                _load_nifti(scalar_img),
                self.ref_img,
                interpolation="continuous",
            )

        scalar_data = np.asarray(prepared_scalar_img.get_fdata())
        atlas_data = np.asarray(self._prepared_atlas_img.get_fdata()).astype(int)
        if self.lut is not None:
            result = self.lut.copy()
        else:
            result = pd.DataFrame({"index": self._regions, "label": [str(r) for r in self._regions]})

        valid_region_ids = set(result["index"])
        stats_data = {}

        for region_id in self._regions:
            if region_id not in valid_region_ids:
                logger.warning("Region ID %d not found in LUT; skipping.", region_id)
                continue
            parcel_mask = atlas_data == region_id
            parcel_values = scalar_data[parcel_mask]

            region_stats = {}
            for stat in self._stat_functions:
                stat_name = stat.name
                stat_func = stat.function
                if stat.requires_image:
                    stat_value = stat_func(parcel_values, prepared_scalar_img)
                else:
                    stat_value = stat_func(parcel_values)
                region_stats[stat_name] = stat_value
            stats_data[region_id] = region_stats

        stats_df = pd.DataFrame.from_dict(stats_data, orient="index")
        stats_df.index.name = "index"
        result = result.merge(stats_df, on="index", how="left")

        return result
