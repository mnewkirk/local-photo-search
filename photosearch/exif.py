"""EXIF metadata extraction for JPEG and ARW files.

Extracts date, GPS, camera info, and image dimensions.
Uses exifread for broad format support including Sony ARW.
"""

import os
from pathlib import Path
from typing import Optional

import exifread


def extract_exif(filepath: str) -> dict:
    """Extract EXIF metadata from a JPEG or ARW file.

    Returns a dict with normalized keys ready for db.add_photo().
    Missing fields are None.
    """
    result = {
        "filepath": str(Path(filepath).resolve()),
        "filename": os.path.basename(filepath),
        "date_taken": None,
        "gps_lat": None,
        "gps_lon": None,
        "camera_make": None,
        "camera_model": None,
        "focal_length": None,
        "exposure_time": None,
        "f_number": None,
        "iso": None,
        "image_width": None,
        "image_height": None,
    }

    try:
        with open(filepath, "rb") as f:
            tags = exifread.process_file(f, details=False)
    except Exception:
        return result

    # Date
    for date_tag in ("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime"):
        if date_tag in tags:
            result["date_taken"] = str(tags[date_tag]).replace(":", "-", 2)
            break

    # Camera info
    if "Image Make" in tags:
        result["camera_make"] = str(tags["Image Make"]).strip()
    if "Image Model" in tags:
        result["camera_model"] = str(tags["Image Model"]).strip()
    if "EXIF FocalLength" in tags:
        result["focal_length"] = str(tags["EXIF FocalLength"])
    if "EXIF ExposureTime" in tags:
        result["exposure_time"] = str(tags["EXIF ExposureTime"])
    if "EXIF FNumber" in tags:
        result["f_number"] = str(tags["EXIF FNumber"])
    if "EXIF ISOSpeedRatings" in tags:
        try:
            result["iso"] = int(str(tags["EXIF ISOSpeedRatings"]))
        except ValueError:
            pass

    # Image dimensions (from EXIF, not pixel decoding — fast)
    for w_tag in ("EXIF ExifImageWidth", "Image ImageWidth"):
        if w_tag in tags:
            try:
                result["image_width"] = int(str(tags[w_tag]))
            except ValueError:
                pass
            break
    for h_tag in ("EXIF ExifImageLength", "Image ImageLength"):
        if h_tag in tags:
            try:
                result["image_height"] = int(str(tags[h_tag]))
            except ValueError:
                pass
            break

    # GPS
    gps_lat = _extract_gps_coord(tags, "GPS GPSLatitude", "GPS GPSLatitudeRef")
    gps_lon = _extract_gps_coord(tags, "GPS GPSLongitude", "GPS GPSLongitudeRef")
    if gps_lat is not None:
        result["gps_lat"] = gps_lat
    if gps_lon is not None:
        result["gps_lon"] = gps_lon

    return result


def _extract_gps_coord(tags: dict, coord_tag: str, ref_tag: str) -> Optional[float]:
    """Convert EXIF GPS DMS (degrees/minutes/seconds) to decimal degrees."""
    if coord_tag not in tags:
        return None
    try:
        values = tags[coord_tag].values
        degrees = _ratio_to_float(values[0])
        minutes = _ratio_to_float(values[1])
        seconds = _ratio_to_float(values[2])
        decimal = degrees + minutes / 60.0 + seconds / 3600.0

        if ref_tag in tags:
            ref = str(tags[ref_tag]).strip().upper()
            if ref in ("S", "W"):
                decimal = -decimal
        return round(decimal, 7)
    except (IndexError, ValueError, ZeroDivisionError, AttributeError):
        return None


def _ratio_to_float(ratio) -> float:
    """Convert an exifread Ratio to float."""
    if hasattr(ratio, "num") and hasattr(ratio, "den"):
        if ratio.den == 0:
            return 0.0
        return ratio.num / ratio.den
    return float(ratio)


def find_raw_pair(jpeg_path: str) -> Optional[str]:
    """Check if a matching ARW file exists alongside a JPEG.

    Given 'DSC04878.JPG', looks for 'DSC04878.ARW' in the same directory.
    Returns the ARW path if found, else None.
    """
    p = Path(jpeg_path)
    for ext in (".ARW", ".arw"):
        raw_path = p.with_suffix(ext)
        if raw_path.exists():
            return str(raw_path)
    return None
