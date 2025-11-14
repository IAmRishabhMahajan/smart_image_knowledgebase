import os, exifread
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

def _to_deg(value):
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)
    return d + (m / 60.0) + (s / 3600.0)

def extract_exif(path: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    try:
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False)
        dt = tags.get("EXIF DateTimeOriginal") or tags.get("Image DateTime")
        if dt:
            # '2024:11:09 18:21:15' -> iso
            meta["datetime"] = datetime.strptime(str(dt), "%Y:%m:%d %H:%M:%S").isoformat()
        # GPS
        gps_lat = tags.get("GPS GPSLatitude")
        gps_lat_ref = tags.get("GPS GPSLatitudeRef")
        gps_lon = tags.get("GPS GPSLongitude")
        gps_lon_ref = tags.get("GPS GPSLongitudeRef")
        if gps_lat and gps_lon and gps_lat_ref and gps_lon_ref:
            lat = _to_deg(gps_lat); lon = _to_deg(gps_lon)
            if str(gps_lat_ref) == "S": lat = -lat
            if str(gps_lon_ref) == "W": lon = -lon
            meta["lat"] = lat; meta["lon"] = lon
        # device
        model = tags.get("Image Model"); make = tags.get("Image Make")
        if model: meta["device"] = str(model)
        if make: meta["make"] = str(make)
    except Exception:
        pass
    return meta
