import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json
from tools.arrays import get_closest_index
from plot_tools.basic_maps import plot_basic_map
from location_info import LocationInfo, get_location_info
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from dataclasses import dataclass
from netCDF4 import Dataset
import numpy as np
import h5py

# using carbon sequestration fraction from Siegel et al. (2021)
# DOI: 10.1088/1748-9326/ac0be0

@dataclass
class CarbonFraction:
    time: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    depth: np.ndarray
    fseq: np.ndarray

def read_carbon_fraction_from_netcdf(input_file=get_dir_from_json('carbon_seq')) -> CarbonFraction:
    nc = Dataset(input_file)
    time = np.arange(0, nc.dimensions['time'].size, 1)
    lon = nc['LON'][0, :, 0].filled(fill_value=np.nan)
    lat = nc['LAT'][0, 0, :].filled(fill_value=np.nan)
    depth = nc['DEPTH'][:, 0, 0].filled(fill_value=np.nan)
    fseq = nc['fseq'][:].filled(fill_value=np.nan)
    mask = nc['MASK'][:].filled(fill_value=np.nan)
    
    fseq_all = np.empty((len(time), len(depth), len(lon), len(lat)))*np.nan
    for t in range(len(time)):
        fseq_all[t, mask==1] = fseq[t, :]
        
    return CarbonFraction(time, lon, lat, depth, fseq_all)

def select_carbon_fraction_at_location_range(carbon_fraction:CarbonFraction,
                                             lon_range:list, lat_range:list) -> CarbonFraction:
    l_lon = np.logical_and(carbon_fraction.lon >= lon_range[0], carbon_fraction.lon <= lon_range[1])
    l_lat = np.logical_and(carbon_fraction.lat >= lat_range[0], carbon_fraction.lat <= lat_range[1])
    
    time = carbon_fraction.time
    lon = carbon_fraction.lon[l_lon]
    lat = carbon_fraction.lat[l_lat]
    depth = carbon_fraction.depth
    fseq = carbon_fraction.fseq[:, :, l_lon, :][:, :, :, l_lat]
    
    return CarbonFraction(time, lon, lat, depth, fseq)

def plot_carbon_sequestration_fraction(carbon_fraction:CarbonFraction,
                                       location_info:LocationInfo,
                                       depth=200, time=100,
                                       cmap='winter', vmin=0, vmax=1):
    
    cf = select_carbon_fraction_at_location_range(carbon_fraction, location_info.lon_range, location_info.lat_range)
    i_depth = get_closest_index(cf.depth, depth)
    i_time = get_closest_index(cf.time, time)
    
    x, y = np.meshgrid(cf.lon, cf.lat)
    
    fig = plt.figure(figsize=(4, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plot_basic_map(ax, location_info)
    
    c = ax.pcolormesh(x, y, cf.fseq[i_time, i_depth, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(c)
    cbar.set_label(f'Fraction carbon remaining after {time} years')
    
    ax.set_title(f'Injection depth {cf.depth[i_depth]} m')
    
    plt.show()

def get_sequestration_fraction_at_depth_location(carbon_fraction:CarbonFraction,
                                                 lon_p:float, lat_p:float, depth_p:float) ->tuple[np.ndarray, np.ndarray]:
    i_lon = get_closest_index(carbon_fraction.lon, lon_p)
    i_lat = get_closest_index(carbon_fraction.lat, lat_p)
    i_depth = get_closest_index(carbon_fraction.depth, depth_p)
    
    fseq = carbon_fraction[:, i_depth, i_lon, i_lat]
    
    return carbon_fraction.time, fseq

if __name__ == '__main__':
    carbon = read_carbon_fraction_from_netcdf()
    
    # location_info = get_location_info('cwa_perth')
    # plot_carbon_sequestration_fraction(carbon, location_info)
    
    lon_perth = 115.5
    lat_perth = -32.
    depth = 200
    
    time, fseq_perth = get_sequestration_fraction_at_depth_location(carbon, lon_perth, lat_perth, depth)
    