import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json, get_daily_files_in_time_range
from tools.timeseries import get_closest_time_index, get_l_time_range
from tools import log
from data.wind_data import WindData, read_era5_wind_data_from_netcdf, get_daily_mean_wind_data
from plot_tools.plots_bathymetry import plot_contours
from plot_tools.plot_cycler import plot_cycler
from location_info import LocationInfo, get_location_info
from plot_tools.basic_maps import plot_basic_map
from plot_tools.general import add_subtitle
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.units as munits
import numpy as np
from datetime import datetime, date, timedelta
import pandas as pd
import cmocean

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)

def plot_cycling_wind_map(wind_data:WindData, location_info:LocationInfo,
                         t_interval=1,
                         vmin=0., vmax=15., cmap=cmocean.cm.tempo):
    
    def single_plot(fig, req_time):
        t = get_closest_time_index(wind_data.time, req_time)
        time_str = wind_data.time[t].strftime('%d-%m-%Y %H:%M')

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)
        ax.set_title(time_str)
        c = ax.pcolormesh(wind_data.lon, wind_data.lat, wind_data.vel[t, :, :],
                          cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')

        ax.quiver(wind_data.lon, wind_data.lat, wind_data.u[t, :, :], wind_data.v[t, :, :])
        
        u_mean = np.nanmean(wind_data.u[t, :, :])
        v_mean = np.nanmean(wind_data.v[t, :, :])
        ax.quiver(114.5, -32.0, u_mean, v_mean, color='r')

        cbar = plt.colorbar(c)
        cbar.set_label('Wind speed (m/s)')

    t = np.arange(0, len(wind_data.time), t_interval)
    time = wind_data.time[t]

    fig = plot_cycler(single_plot, time)
    plt.show()

def plot_wind_map(wind_data:WindData, location_info:LocationInfo,
                  time:datetime, ax=None, show=True, output_path=None,
                  cmap=cmocean.cm.tempo, vmin=None, vmax=None) -> plt.axes:
    
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)
    
    if hasattr(wind_data, 'time') and time is not None:
        t = get_closest_time_index(wind_data.time, time)
    else:
        t = 0
        
    c = ax.pcolormesh(wind_data.lon, wind_data.lat, wind_data.vel, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(c)
    cbar.set_label('Wind speed (m/s)')
    ax.quiver(wind_data.lon, wind_data.lat, wind_data.u, wind_data.v, scale=5)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax
    
if __name__ == '__main__':
    start_date = datetime(2017, 3, 1)
    end_date = datetime(2017, 9, 30)
    location_info_perth = get_location_info('perth')
    wind_data = read_era5_wind_data_from_netcdf(get_dir_from_json("era5_data"), start_date, end_date,
                                                lon_range=location_info_perth.lon_range,
                                                lat_range=location_info_perth.lat_range)
    wind_data = get_daily_mean_wind_data(wind_data)
    plot_cycling_wind_map(wind_data, get_location_info('perth_wide'))
