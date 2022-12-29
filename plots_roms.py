from tools.files import get_dir_from_json
from tools.timeseries import get_closest_time_index
from tools.coordinates import get_distance_between_points, get_points_on_line_between_points
from roms_data import RomsGrid, RomsData, read_roms_data_from_multiple_netcdfs
from location_info import LocationInfo, get_location_info
from basic_maps import plot_basic_map
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd

def get_eta_xi_along_transect(grid:RomsGrid, lon1:float, lat1:float, lon2:float, lat2:float, ds:float) -> tuple:
    lons, lats = get_points_on_line_between_points(lon1, lat1, lon2, lat2, ds)
    eta, xi = grid.get_eta_xi_of_lon_lat_point(lons, lats)
    return eta, xi

def get_distance_along_transect(lons:np.ndarray, lats:np.ndarray):
    distance = [0]
    
    for i in range(len(lons)-1):
        d = get_distance_between_points(lons[i], lats[i], lons[i+1], lats[i+1])
        distance.append(d)
    distance = np.array(distance)
    
    return np.cumsum(distance) # distance in meters

def plot_roms_map(roms_data:RomsData, location_info:LocationInfo,
                  parameter:str, time:datetime, s=-1, # default: surface
                  ax=None, show=True,
                  cmap='RdBu_r', clabel='', vmin=None, vmax=None) -> plt.axes:
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    t = get_closest_time_index(roms_data.time, time)

    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
        if len(values.shape) == 4:
            values = values[t, s, :, :] # [time, s, eta, xi]
        elif len(values.shape) == 3:
            values = values[t, :, :] # [time, eta, xi] -> correct? or can also be [s, eta, xi]?
        elif len(values.shape) == 2:
            values = values[:, :] # [eta, xi]
    elif hasattr(roms_data.grid, parameter):
        values = getattr(roms_data.grid, parameter)
        if len(values.shape) == 3:
            values = values[s, :, :] # [s, eta, xi] (because grid data is non-time dependent)
        elif len(values.shape) == 2:
            values = values[:, :] # [eta, xi]
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

def plot_roms_map_with_transect(roms_data:RomsData, location_info:LocationInfo,
                                lon1:float, lat1:float, lon2:float, lat2:float, ds:float,
                                parameter:str, time:datetime, s=-1, # default: surface
                                ax=None, show=True,
                                cmap='RdBu_r', clabel='', vmin=None, vmax=None) -> plt.axes:

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    eta, xi = get_eta_xi_along_transect(roms_data.grid, lon1, lat1, lon2, lat2, ds)
    lon = roms_data.grid.lon[eta, xi]
    lat = roms_data.grid.lat[eta, xi]

    # TEMP: show glider points as test
    df = pd.read_csv('test_glider_coords.csv')
    glider_lon = df['lon'].values
    glider_lat = df['lat'].values

    ax = plot_roms_map(roms_data, location_info, parameter, time, s=s, ax=ax, show=False, cmap=cmap, clabel=clabel, vmin=vmin, vmax=vmax)
    ax.plot(glider_lon, glider_lat, 'xk', label='Glider transect')
    ax.plot(lon, lat, '.k', label='ROMS transect')

    ax.legend(loc='upper left')

    if show is True:
        plt.show()
    else:
        return ax

def plot_roms_transect(roms_data:RomsData,
                       lon1:float, lat1:float, lon2:float, lat2:float, ds:float,
                       parameter:str, time:datetime,
                       ax=None, show=True,
                       cmap='RdBu_r', clabel='', vmin=None, vmax=None) -> plt.axes:
    
    eta, xi = get_eta_xi_along_transect(roms_data.grid, lon1, lat1, lon2, lat2, ds)

    lon = roms_data.grid.lon[eta, xi]
    lat = roms_data.grid.lat[eta, xi]
    distance = get_distance_along_transect(lon, lat)/1000 # distance in km

    z = roms_data.grid.z[:, eta, xi]
    h = roms_data.grid.h[eta, xi]

    t = get_closest_time_index(roms_data.time, time)

    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
        if len(values.shape) == 4:
            values = values[t, :, eta, xi] # [time, s, eta, xi]
        elif len(values.shape) == 3:
            values = values[t, eta, xi] # [time, eta, xi]
        elif len(values.shape) == 2:
            values = values[eta, xi] # [eta, xi]
    elif parameter == 'velocity':
        u = roms_data.u[t, :, eta, xi]
        v = roms_data.v[t, :, eta, xi]
        values = np.sqrt(u**2+v**2)
    else:
        raise ValueError(f'Unknown parameter {parameter} in RomsData to plot transect')

    if ax is None:
        ax = plt.axes()
    distance2d = np.repeat(distance[np.newaxis, :], z.shape[0], axis=0)
    c = ax.pcolormesh(distance2d, z, values.transpose(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.fill_between(distance, -h, np.nanmin(z), edgecolor='k', facecolor='#989898') # ROMS bottom
    
    ax.set_xlabel('Distance along transect (km)')
    ax.set_xlim([0, np.nanmax(distance)])
    ax.set_ylabel('Depth (m)')
    ax.set_ylim([np.nanmin(z), 0])
    
    cbar = plt.colorbar(c)
    cbar.set_label(clabel)
    
    if show is True:
        plt.show()
    else:
        return ax

if __name__ == '__main__':
    location_info = get_location_info('perth')
    input_dir = f'{get_dir_from_json("roms_data")}2022/'
    start_date = datetime(2022, 6, 30)
    end_date = datetime(2022, 7, 2)
    roms = read_roms_data_from_multiple_netcdfs(input_dir, start_date, end_date)
    
    plot_roms_map_with_transect(roms, location_info, 115.61, -31.80, 115.26, -31.95, 5000, 'h', start_date)
    plot_roms_transect(roms, 115.61, -31.80, 115.26, -31.95, 500, 'temp', start_date)