from tools import log
from tools.files import get_dir_from_json, get_daily_files_in_time_range
from tools.timeseries import convert_time_to_datetime, get_l_time_range
from plot_tools.basic_maps import plot_basic_map
from location_info import LocationInfo, get_location_info
from plot_tools.plots_particles import plot_timeseries_in_deep_sea, plot_age_in_deep_sea
from particles import Particles
from plot_tools.plots_particles import plot_histogram_arriving_in_deep_sea
from data.kelp_data import KelpProbability
from data.bathymetry_data import BathymetryData
from plot_tools.plots_bathymetry import plot_contours
from data.roms_data import RomsData, RomsGrid, read_roms_data_from_multiple_netcdfs, read_roms_grid_from_netcdf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import cartopy.crs as ccrs
from datetime import date, datetime, timedelta
import numpy as np
from netCDF4 import Dataset
import pandas as pd

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)

def get_coastal_offshore_temperature_timeseries_from_roms_netcdfs(input_dir:str,
                                                            start_date:datetime,
                                                            end_date:datetime,
                                                            lon_c:float, lat_c:float,
                                                            lon_o:float, lat_o:float, s=-1,
                                                            grid_input_path='input/cwa_roms_grid.nc') -> tuple:
    
    grid = read_roms_grid_from_netcdf(grid_input_path)
    eta_c, xi_c = grid.get_eta_xi_of_lon_lat_point(lon_c, lat_c)
    eta_o, xi_o = grid.get_eta_xi_of_lon_lat_point(lon_o, lat_o)
    
    time = []
    temp_c = []
    temp_o = []
    nc_files = get_daily_files_in_time_range(input_dir, start_date, end_date, 'nc')
    for nc_file in nc_files:
        nc = Dataset(nc_file)

        time_org = nc['ocean_time'][:].filled(fill_value=np.nan)
        time_units = nc['ocean_time'].units
        time.append(convert_time_to_datetime(time_org, time_units))

        temp_c.append(nc['temp'][:, s, eta_c, xi_c].filled(fill_value=np.nan))
        temp_o.append(nc['temp'][:, s, eta_o, xi_o].filled(fill_value=np.nan))

    return np.array(time).flatten(), np.array(temp_c).flatten(), np.array(temp_o).flatten()

def get_daily_mean_temperature_difference(time:np.ndarray, temp:np.ndarray) -> tuple:

    daily_time = []
    daily_temp = []

    n_days = (time[-1]-time[0]).days

    for n in range(n_days):
        start_date = time[0]+timedelta(days=n)
        end_date = start_date+timedelta(days=1)
        l_time = get_l_time_range(time, start_date, end_date)
        daily_time.append(start_date)
        daily_temp.append(np.nanmean(temp[l_time]))

    return np.array(daily_time), np.array(daily_temp)

def plot_temperature_gradient_in_time(time: np.ndarray, temp_c:np.ndarray, temp_o:np.ndarray,
                                      color='k', ax=None,
                                      show=True, output_path=None) -> plt.axes:

    dtemp = temp_c-temp_o

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()
        ax.set_ylabel('Temperature difference coastal-offshore ($^o$C)')

    ax.plot(time, dtemp, color=color)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_particles_arriving_with_temperature_gradient(particles:Particles, h_deep_sea:float,
                                                      time:np.ndarray, temp_c:np.ndarray, temp_o:np.ndarray,
                                                      color_temp='k', color_hist='#1b7931', shade_dswc=False,
                                                      show=True, output_path=None):

    dswc_shading = '#e4e4e4'
    alpha = 0.4
    dtemp = temp_c-temp_o
    l_dswc = dtemp < 0

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(3, 1, 3)
    ax = plot_histogram_arriving_in_deep_sea(particles, h_deep_sea, ax=ax, show=False, color=color_hist)
    if shade_dswc is True:
        # add shading where DSWC can occur
        ylim = ax.get_ylim()
        ax.fill_between(time, ylim[1], where=l_dswc, facecolor=dswc_shading, alpha=alpha)
        ax.set_ylim(ylim)

    ax2 = plt.subplot(3, 1, 2)
    ax2 = plot_temperature_gradient_in_time(time, temp_c, temp_o,
                                            ax=ax2, color=color_temp, show=False)
    ax2.plot([time[0], time[-1]], [0, 0], '--k')
    
    ax2.set_ylabel('Temperature difference\ncoastal-offshore ($^o$C)', color=color_temp)
    ax2.set_xticklabels([])
    ax2.set_xlim([time[0], time[-1]])
    ax2.set_ylim([-4, 4])
    # add shading where DSWC can occur
    if shade_dswc is True:
        ax2.fill_between(time, -4, where=l_dswc, facecolor=dswc_shading, alpha=alpha)


    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

if __name__ == '__main__':
    
    # # --- ROMS data ---
    # input_dir = f'{get_dir_from_json("roms_data")}2017/'
    # start_date = datetime(2017, 3, 1)
    # end_date = datetime(2017, 8, 1)
    # lon_c = 115.70 # coastal coordinate
    # lat_c = -31.76
    # lon_o = 114.47 # offshore coordinate
    # lat_o = -31.80

    # time, temp_c, temp_o = get_coastal_offshore_temperature_timeseries_from_roms_netcdfs(
    #                                                                     input_dir, start_date, end_date,
    #                                                                     lon_c, lat_c, lon_o, lat_o)
    # df = pd.DataFrame(columns=['time', 'temp_c', 'temp_o'])
    # df['time'] = time
    # df['temp_c'] = temp_c
    # df['temp_o'] = temp_o
    # df.to_csv('test_roms_temperature_points.csv', index=False)

    df = pd.read_csv('test_roms_temperature_points.csv')
    time = pd.to_datetime(df['time'].values)
    temp_c = df['temp_c'].values
    temp_o = df['temp_o'].values

    daily_time, daily_temp_c = get_daily_mean_temperature_difference(time, temp_c)
    _, daily_temp_o = get_daily_mean_temperature_difference(time, temp_o)

    # --- Particle tracking data ---
    h_deep_sea = 600 # m depth: max Leeuwin Undercurrent depth
    input_path = f'{get_dir_from_json("opendrift")}cwa-perth_2017-Mar-Aug.nc'
    particles = Particles.read_from_netcdf(input_path)

    # -- plots ---
    plot_particles_arriving_with_temperature_gradient(particles, h_deep_sea, daily_time, daily_temp_c, daily_temp_o)

