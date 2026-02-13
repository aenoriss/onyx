import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def parse_srt_data(raw_data: bytes) -> List[Dict[str, float]]:
    """
    Parses SRT subtitle data to extract GPS telemetry.
    
    Expected format (DJI style):
    1
    00:00:00,000 --> 00:00:00,032
    [time: 123456] [latitude: 12.34567] [longitude: 123.45678] [altitude: 50.5] ...
    
    Args:
        raw_data: Raw bytes of the SRT file/stream
        
    Returns:
        List of dictionaries containing:
        - timestamp: float (seconds)
        - lat: float
        - lon: float
        - alt: float
    """
    try:
        text_data = raw_data.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error decoding SRT data: {e}")
        return []

    samples = []
    
    # Split by double newlines to get blocks
    # Normalize line endings first
    text_data = text_data.replace('\r\n', '\n')
    blocks = text_data.strip().split('\n\n')
    
    # Regex patterns
    # Matches: [latitude: 12.345] or [latitude : 12.345]
    lat_pattern = re.compile(r'\[\s*latitude\s*:\s*([-\d.]+)\s*\]', re.IGNORECASE)
    lon_pattern = re.compile(r'\[\s*longitude\s*:\s*([-\d.]+)\s*\]', re.IGNORECASE)
    alt_pattern = re.compile(r'\[\s*altitude\s*:\s*([-\d.]+)\s*\]', re.IGNORECASE)
    
    # Alternative: GPS(lat, lon, alt)
    gps_pattern = re.compile(r'GPS\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)', re.IGNORECASE)

    for block in blocks:
        lines = block.split('\n')
        # We need at least index, time, and text
        if len(lines) < 3:
            continue
            
        time_line = lines[1]
        
        # Combine all subsequent lines as text content
        content = " ".join(lines[2:])
        
        # Parse Timestamp
        # Format: HH:MM:SS,mmm --> HH:MM:SS,mmm
        try:
            if '-->' not in time_line:
                continue
                
            start_str = time_line.split('-->')[0].strip()
            # Parse HH:MM:SS,mmm
            parts = start_str.split(':')
            if len(parts) != 3:
                continue
                
            h = int(parts[0])
            m = int(parts[1])
            s_parts = parts[2].split(',')
            
            if len(s_parts) != 2:
                continue
                
            s = int(s_parts[0])
            ms = int(s_parts[1])
            
            timestamp = h * 3600 + m * 60 + s + ms / 1000.0
            
        except (ValueError, IndexError):
            continue

        # Extract Coordinates
        lat = None
        lon = None
        alt = 0.0
        
        lat_match = lat_pattern.search(content)
        lon_match = lon_pattern.search(content)
        alt_match = alt_pattern.search(content)
        
        if lat_match and lon_match:
            try:
                lat = float(lat_match.group(1))
                lon = float(lon_match.group(1))
                if alt_match:
                    alt = float(alt_match.group(1))
            except ValueError:
                continue
        else:
            # Try fallback GPS pattern
            gps_match = gps_pattern.search(content)
            if gps_match:
                try:
                    lat = float(gps_match.group(1))
                    lon = float(gps_match.group(2))
                    alt = float(gps_match.group(3))
                except ValueError:
                    continue

        if lat is not None and lon is not None:
            samples.append({
                'timestamp': timestamp,
                'lat': lat,
                'lon': lon,
                'alt': alt
            })

    return samples
