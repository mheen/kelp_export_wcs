from tools import log
from location_info import LocationInfo, get_location_info
from basic_maps import plot_basic_map
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

class BathymetryData:
    def __init__(self, lon, lat, h):
        self.lon = lon
        self.lat = lat
        self.h = h

    def plot_contours(self, location_info:LocationInfo, ax=None, show=True, color='k') -> plt.axes:
        if ax is None:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax = plot_basic_map(ax, location_info)
            
        def _fmt(x):
            s = f'{x:.0f}'
            return s

        cs = ax.contour(self.lon, self.lat, self.h, levels=location_info.contour_levels,
                        colors=color, linewidths=1, transform=ccrs.PlateCarree())
        ax.clabel(cs, cs.levels, fontsize=8, inline=True, fmt=_fmt)

        cs200 = ax.contour(self.lon, self.lat, self.h, [200],
                           colors='k', linewidths=1.5, transform=ccrs.PlateCarree())

        t = ax.text(114.93, -32.10, 'Perth canyon', va='center', ha='right', transform=ccrs.PlateCarree())
        t.set_bbox(dict(facecolor='w', alpha=0.6, edgecolor='w'))

        if show is True:
            plt.show()
        else:
            return ax

    def write_roms_bathymetry_to_csv(self, output_path:str):
        
        df = pd.DataFrame(columns=['lon', 'lat', 'h'])
        df['lon'] = self.lon.flatten()
        df['lat'] = self.lat.flatten()
        df['h'] = self.h.flatten()

        df.to_csv(output_path, index=False)
        log.info(f'Wrote ROMS bathymetry data to: {output_path}')

    @staticmethod
    def read_from_csv(input_path:str):
        log.info(f'Reading bathymetry data from: {input_path}')

        df = pd.read_csv(input_path)
        lon = df['lon'].values
        lat = df['lat'].values
        h = df['h'].values

        return BathymetryData(lon, lat, h)

    @staticmethod
    def read_from_netcdf(input_path:str):
        log.info(f'Reading bathymetry data from: {input_path}')

        netcdf = Dataset(input_path)
        lon = netcdf['lon_rho'][:].filled(fill_value=np.nan)
        lat = netcdf['lat_rho'][:].filled(fill_value=np.nan)
        h = netcdf['h'][:].filled(fill_value=np.nan)
        netcdf.close()

        return BathymetryData(lon, lat, h)

if __name__ == '__main__':
    bathymetry = BathymetryData.read_from_netcdf('input/cwa_roms_grid.nc')
    location_info = get_location_info('cwa_perth')
    bathymetry.plot_contours(location_info)