from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.image import resample_to_img
from parcellate.utils import _load_nifti

StatFunction = Callable[[np.ndarray], float]


class VolumetricParcellator:
    """Base volumetric parcellator.

    The parcellator assumes an integer-valued atlas where each non-background
    voxel stores the parcel identifier. Parcellation is performed by sampling a
    scalar map image and aggregating values inside each region. Resampling to
    the atlas grid is handled by default to keep atlas boundaries consistent
    across inputs.
    """

    REQUIRED_LUT_COLUMNS: ClassVar[set[str]] = {"index", "label"}

    def __init__(
        self,
        atlas_img: nib.Nifti1Image | str | Path,
        labels: Mapping[int, str] | Sequence[str] | None = None,
        lut: pd.DataFrame | str | Path | None = None,
        *,
        mask: nib.Nifti1Image | str | Path | None = None,
        background_label: int = 0,
        resampling_target: Literal["data", "labels", None] = "data",
        stat_functions: Mapping[str, StatFunction] | None = None,
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
            "index" and "name" following the BIDS standard.
        mask : nib.Nifti1Image | str | Path | None, optional
            Optional mask to apply to the atlas, by default None
        background_label : int, optional
            Label value to treat as background, by default 0
        resampling_target : Literal["data", "labels", None], optional
            Resampling target for input maps, by default "data"
        stat_functions : Mapping[str, StatFunction] | None, optional
            Mapping of statistic names to functions, by default None
        """
        self.atlas_img = _load_nifti(atlas_img)
        self.lut = self._load_atlas_lut(lut) if lut is not None else None
        if mask is not None:
            self.mask = _load_nifti(mask)
        self.background_label = int(background_label)
        self.resampling_target = resampling_target
        self._atlas_data = self._load_atlas_data()
        self._regions = self._build_regions(labels)
        self._stat_functions = self._prepare_stat_functions(stat_functions)

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
            raise ValueError(f"Lookup table is missing required columns: {missing}")

        return lut_df

    @property
    def regions(self) -> tuple[int, ...]:
        """Tuple of regions defined in the atlas."""
        return self._regions

    def parcellate(
        self,
        map_img: nib.Nifti1Image,
        stat_functions: Mapping[str, StatFunction] | None = None,
    ) -> pd.DataFrame:
        """Extract parcel-wise statistics from a scalar map.

        Args:
            map_img: Scalar-valued image to sample.
            stat_functions: Optional mapping of statistics to compute. Defaults
                to the statistics supplied at initialization.

        Returns:
            DataFrame with one row per parcel containing the requested
            statistics. Columns include ``region_id``, ``region_name``,
            ``voxel_count``, and one column per statistic.
        """
        prepared_map = self._prepare_map(map_img)
        map_data = np.asarray(prepared_map.get_fdata())
        stats = self._prepare_stat_functions(stat_functions, fallback=self._stat_functions)

        rows: list[dict[str, float | int | str]] = []
        for region in self._regions:
            mask = self._atlas_data == region.region_id
            if not mask.any():
                continue

            values = map_data[mask]
            region_stats = {
                "region_id": region.region_id,
                "region_name": region.name,
                "voxel_count": int(mask.sum()),
            }
            for name, func in stats.items():
                region_stats[name] = float(func(values))

            rows.append(region_stats)

        return pd.DataFrame(rows, columns=["region_id", "region_name", "voxel_count", *stats.keys()])

    def _prepare_map(self, map_img: nib.Nifti1Image) -> nib.Nifti1Image:
        if map_img.shape != self.atlas_img.shape or not np.allclose(map_img.affine, self.atlas_img.affine):
            if self.resampling == "map_to_atlas":
                return resample_to_img(map_img, self.atlas_img, interpolation="continuous")
            raise ValueError(
                "Input map does not match atlas grid; enable resampling or supply a map in atlas space.",
            )

        return map_img

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
            raise ValueError("Atlas must be a 3D volume.")
        return atlas_data.astype(int)

    def _prepare_stat_functions(
        self,
        stat_functions: Mapping[str, StatFunction] | None,
        *,
        fallback: Mapping[str, StatFunction] | None = None,
    ) -> Mapping[str, StatFunction]:
        if stat_functions is None:
            if fallback is None:
                return {
                    "mean": np.nanmean,
                    "median": np.nanmedian,
                    "std": np.nanstd,
                }
            return fallback

        prepared = {str(name): func for name, func in stat_functions.items()}
        if not prepared:
            raise ValueError("At least one statistic function must be provided.")
        return prepared
