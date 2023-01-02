from tools import log
from netCDF4 import Dataset
import numpy as np
import pandas as pd

class BathymetryData:
    def __init__(self, lon, lat, h):
        self.lon = lon
        self.lat = lat
        self.h = h

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
