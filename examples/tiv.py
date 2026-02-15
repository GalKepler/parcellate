"""Example: Extract TIV (Total Intracranial Volume) from CAT12 XML reports."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)


def _extract_tiv_from_xml(xml_path: Path) -> float | None:
    """Extract vol_TIV from CAT12 XML <subjectmeasures> section.

    The XML contains multiple vol_TIV elements - some with description text
    and some with actual numeric values. This function finds the first
    valid numeric TIV value from the root-level subjectmeasures section.

    Args:
        xml_path: Path to CAT12 XML file (cat_*.xml)

    Returns:
        TIV value in mL, or None if extraction fails
    """
    try:
        tree = ET.parse(xml_path)  # noqa: S314
        root = tree.getroot()

        # Find vol_TIV in subjectmeasures sections
        # Note: XML may have multiple subjectmeasures - some nested with descriptions,
        # and a root-level one with actual numeric values
        for subj_measures in root.iter("subjectmeasures"):
            vol_tiv = subj_measures.find("vol_TIV")
            if vol_tiv is not None and vol_tiv.text:
                try:
                    return float(vol_tiv.text.strip())
                except ValueError:
                    # This vol_TIV contains description text, not a number
                    continue

        # Alternative: look directly for vol_TIV anywhere in the tree
        # Skip subjectratings which contains normalized values
        for vol_tiv in root.iter("vol_TIV"):
            if vol_tiv.text:
                # Check if parent is subjectratings (we want subjectmeasures)
                # Since ElementTree doesn't have parent access, just try to parse
                try:
                    return float(vol_tiv.text.strip())
                except ValueError:
                    continue

    except ET.ParseError:
        logger.warning("Failed to parse XML %s", xml_path, exc_info=True)
        return None
    except Exception:
        logger.warning("Unexpected error extracting TIV from %s", xml_path, exc_info=True)
        return None

    logger.debug("No numeric vol_TIV found in XML: %s", xml_path)
    return None
