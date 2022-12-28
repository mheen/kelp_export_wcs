from dataclasses import dataclass

@dataclass
class LocationInfo:
    name: str
    lon_range: list
    lat_range: list
    meridians: list
    parallels: list
    contour_levels: list

perth = LocationInfo('perth', [115.25, 115.87], [-32.65, -31.49],
                     [115.3, 115.7], [-32.4, -32.0, -31.6],
                     [10, 25, 50, 100, 150, 200])
cwa_perth = LocationInfo('cwa_perth', [113.20, 115.87], [-34.33, -30.85],
                         [113.0, 114.0, 115,0], [-34.0, -33.0, -32.0, -31.0],
                         [20, 50, 100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000, 6000])
all_locations = [perth, cwa_perth]

def get_location_info(location:str) -> LocationInfo:
    for loc in all_locations:
        if loc.name == location:
            return loc
    raise ValueError(f'Unknown location requested')
    pass