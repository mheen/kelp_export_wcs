from basic_maps import perth_map
from netCDF4 import Dataset
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from py_tools.files import get_dir_from_json, get_files_in_dir
from py_tools import log

class BathymetryData:
    def __init__(self, lon, lat, h):
        self.lon = lon
        self.lat = lat
        self.h = h

    def plot_contours(self, ax=None, show=True) -> plt.axes:
        if ax is None:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax = perth_map(ax)
            
        def _fmt(x):
            s = f'{x:.0f}'
            return s

        cs = ax.contour(self.lon, self.lat, self.h, levels=[10, 25, 50, 100, 150, 200],
                        colors='k', linewidths=1, transform=ccrs.PlateCarree())
        ax.clabel(cs, cs.levels, fontsize=8, inline=True, fmt=_fmt)

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
    def read_from_csv(input_path='input/perth_roms_bathymetry.csv'):
        log.info(f'Reading bathymetry data from: {input_path}')

        df = pd.read_csv(input_path)
        lon = df['lon'].values
        lat = df['lat'].values
        h = df['h'].values

        return BathymetryData(lon, lat, h)

    @staticmethod
    def read_from_netcdf(input_path='input/perth_roms_grid.nc'):
        log.info(f'Reading bathymetry data from: {input_path}')

        netcdf = Dataset(input_path)
        lon = netcdf['lon_rho'][:].filled(fill_value=np.nan)
        lat = netcdf['lat_rho'][:].filled(fill_value=np.nan)
        h = netcdf['h'][:].filled(fill_value=np.nan)
        netcdf.close()

        return BathymetryData(lon, lat, h)

if __name__ == '__main__':
    bathy = BathymetryData.read_from_netcdf()
    bathy.plot_contours()