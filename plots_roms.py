from roms_data import RomsData, read_roms_data_from_netcdf
from location_info import LocationInfo, get_location_info
from basic_maps import plot_basic_map
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_map_roms_data(roms_data:RomsData, location_info:LocationInfo, parameter:str, t:int, s:int,
                       ax=None, show=True,
                       cmap='RdBu_r', clabel='', vmin=None, vmax=None) -> plt.axes:
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
        if len(values.shape) == 4:
            values = values[t, s, :, :] # [time, s, eta, xi]
        else:
            raise ValueError(f'Map plotting currently only works for 4D variables')
    elif parameter == 'velocity':
        u = roms_data.u[t, s, :, :]
        v = roms_data.v[t, s, :, :]
        values = np.sqrt(u**2+v**2)
    else:
        raise ValueError(f'Unknown parameter {parameter} in RomsData')

    c = ax.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, values, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(c)
    cbar.set_label(clabel)

    if parameter == 'velocity':
        thin = 5
        i = np.arange(0, u.shape[0], thin)
        j = np.arange(0, u.shape[1], thin)
        u_q = u[i][:, j]
        v_q = v[i][:, j]
        lon_q = roms_data.grid.lon[i][:, j]
        lat_q = roms_data.grid.lat[i][:, j]
        ax.quiver(lon_q, lat_q, u_q, v_q, scale=10, transform=ccrs.PlateCarree())

    if show is True:
        plt.show()
    else:
        return ax

