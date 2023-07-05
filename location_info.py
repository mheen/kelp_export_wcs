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
perth_wide = LocationInfo('perth_wide', [114.25, 115.87], [-32.65, -31.49],
                     [114.5, 115.0, 115.5], [-32.4, -32.0, -31.6],
                     [10, 25, 50, 100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000])
perth_wide_south = LocationInfo('perth_wide_south', [114.45, 115.87], [-33.8, -31.3],
                                [114.5, 115.0, 115.5], [-33.6, -33.2, -32.8, -32.4, -32.0, -31.6],
                                [10, 25, 50, 100, 200, 600, 1000, 2000, 3000, 4000, 5000, 6000])
cwa_perth = LocationInfo('cwa_perth', [113.0, 115.87], [-34.33, -30.85],
                         [113.0, 114.0, 115,0], [-34.0, -33.0, -32.0, -31.0],
                         [10, 25, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000])
cwa_perth_zoom = LocationInfo('cwa_perth_zoom', [114.0, 115.87], [-33.0, -31.5],
                              [114.0, 114.5, 115.0, 115.5], [-33.0, -32.5, -32.0, -31.5],
                              [10, 25, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000])
cwa_perth_zoom2 = LocationInfo('cwa_perth_zoom2', [114.0, 115.87], [-34.33, -31.0],
                               [114.5, 115.0, 115.5], [-34.0, -33.0, -32.0, -31.0],
                               [10, 20, 50, 100, 200, 500, 1000, 2000])
gsr = LocationInfo('gsr', [112.0, 155.0], [-45.0, -27.5],
                   [120.0, 130.0, 140., 150.0], [-44.0, -40.0, -36.0, -32.0, -28.0],
                   [10, 25, 50, 100, 200, 400, 600, 800, 1000, 1500, 2000, 3000, 4000, 5000, 6000])
wa = LocationInfo('wa', [112.0, 116.0], [-34.0, -28.0],
                  [112.0, 114.0, 116.0], [-34.0, -32.0, -30.0, -28.0], [10, 50, 100, 200, 400, 600, 800, 1000, 2000, 3000, 4000, 5000, 6000])

all_locations = [perth, cwa_perth, cwa_perth_zoom, cwa_perth_zoom2, gsr, wa, perth_wide, perth_wide_south]

def get_location_info(location:str) -> LocationInfo:
    for loc in all_locations:
        if loc.name == location:
            return loc
    raise ValueError(f'Unknown location requested')
    pass