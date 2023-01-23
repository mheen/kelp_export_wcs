from tools import log
from tools.files import get_dir_from_json, get_daily_files_in_time_range
from tools.timeseries import convert_time_to_datetime, get_l_time_range, get_daily_means
from plot_tools.basic_maps import plot_basic_map
from location_info import LocationInfo, get_location_info
from plot_tools.plots_particles import plot_timeseries_in_deep_sea, plot_age_in_deep_sea
from particles import Particles
from plot_tools.plots_particles import plot_histogram_arriving_in_deep_sea
from data.kelp_data import KelpProbability
from data.bathymetry_data import BathymetryData
from data.wind_data import read_era5_wind_data, get_wind_vel_and_dir_in_point, get_wind_dir_and_text
from plot_tools.plots_bathymetry import plot_contours
from data.roms_data import RomsData, RomsGrid, read_roms_data_from_multiple_netcdfs, read_roms_grid_from_netcdf
import matplotlib
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

def plot_temperature_gradient_timeseries(time: np.ndarray, temp_c:np.ndarray, temp_o:np.ndarray,
                                         color='k', ax=None,
                                         show=True, output_path=None) -> plt.axes:

    daily_time, daily_temp_c = get_daily_means(time, temp_c)
    _, daily_temp_o = get_daily_means(time, temp_o)
    dtemp = daily_temp_c-daily_temp_o

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()
        ax.set_ylabel('Temperature difference coastal-offshore ($^o$C)')

    ax.plot(daily_time, dtemp, color=color)
    ax.set_xlim([daily_time[0], daily_time[-1]])

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_wind_vel_timeseries(time:np.ndarray, dir:np.ndarray,
                             color='k', ax=None, show=True,
                             output_path=None) -> plt.axes:

    daily_time, daily_dir = get_daily_means(time, dir)

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()

    ax.plot(daily_time, daily_dir, color=color)
    ax.set_ylabel('Wind direction')
    ax.set_xlim([daily_time[0], daily_time[-1]])

    dir_ticks, dir_ticklabels = get_wind_dir_and_text()
    ax.set_yticks(dir_ticks)
    ax.set_yticklabels(dir_ticklabels)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_wind_vel_timeseries(time:np.ndarray, vel:np.ndarray,
                            color='k', ax=None, show=True,
                            output_path=None) -> plt.axes:

    daily_time, daily_vel = get_daily_means(time, vel)

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()

    ax.plot(daily_time, daily_vel, color=color)
    ax.set_ylabel('Wind speed (m/s)')
    ax.set_xlim([daily_time[0], daily_time[-1]])

    dir_ticks, dir_ticklabels = get_wind_dir_and_text()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_wind_arrows_timeseries(time:np.ndarray, vel:np.ndarray, dir:np.ndarray, xlim=None,
                                ax=None, show=True, output_path=None, vmin=0, vmax=15):

    colors = ['#008010', '#ffe800', '#C70039', '#652c04'] # green, yellow, red, brown
    color_ranges = np.array([0, 12, 18, 35, 50])/1.94 # knots to m/s, using similar ranges to seabreeze

    daily_time, daily_vel = get_daily_means(time, vel)
    _, daily_dir = get_daily_means(time, dir)

    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()

    if xlim is None:
        ax.set_xlim([daily_time[0], daily_time[-1]])
        l_xlim = np.ones(len(daily_time)).astype(bool)
    else:
        ax.set_xlim(xlim)
        l_xlim = get_l_time_range(daily_time, xlim[0], xlim[1])

    ax.set_ylim([vmin, vmax])
    ax.set_ylabel('Wind speed (m/s)')

    ax.plot(daily_time, daily_vel, ':k')

    i_times = np.where(l_xlim)[0]

    for i in i_times: # WRONG ROTATION! DIR IS IN METEO DIRECTION
        i_color = np.where(daily_vel[i]-color_ranges<0)[0][0]-1
        rotation = np.mod(daily_dir[i]+270, 360) # conversion from meteo wind dir to regular vector dir
        ax.text(mdates.date2num(daily_time[i]), daily_vel[i], '   ', rotation=rotation, bbox=dict(boxstyle='rarrow', fc=colors[i_color], ec='k'), fontsize=4)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_particles_arriving_with_dswc_conditions(particles:Particles, h_deep_sea:float,
                                                 time:np.ndarray, temp_c:np.ndarray, temp_o:np.ndarray,
                                                 time_wind:np.ndarray, wind_vel:np.ndarray, wind_dir:np.ndarray,
                                                 color_temp='k', color_hist='#1b7931', shade_dswc=False,
                                                 color_vel='k', color_dir='r',
                                                 show=True, output_path=None):

    dswc_shading = '#e4e4e4'
    alpha = 0.4
    dtemp = temp_c-temp_o
    l_dswc = dtemp < 0
    ylim2 = [-4, 4] # limits for temperature gradient
    ylim3 = [0, 15] # limits for wind speed (m/s)
    xlim = [particles.time[0], particles.time[-1]]

    fig = plt.figure(figsize=(10, 8))

    # --- particles histogram ---
    ax = plt.subplot(4, 1, (3, 4))
    ax = plot_histogram_arriving_in_deep_sea(particles, h_deep_sea, ax=ax, show=False, color=color_hist)
    if shade_dswc is True:
        # add shading where DSWC can occur
        ylim1 = ax.get_ylim()
        ax.fill_between(time, ylim1[1], where=l_dswc, facecolor=dswc_shading, alpha=alpha)
        ax.set_ylim(ylim1)
    ax.set_xlim(xlim)
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- temperature gradient ---
    ax2 = plt.subplot(4, 1, 2)
    ax2 = plot_temperature_gradient_timeseries(time, temp_c, temp_o,
                                               ax=ax2, color=color_temp, show=False)
    ax2.plot([time[0], time[-1]], [0, 0], '--k')
    ax2.set_ylabel('Temperature\ngradient ($^o$C)', color=color_temp)
    ax2.set_xlim(xlim)
    ax2.set_xticklabels([])
    ax2.set_ylim(ylim2)
    ax2.grid(True, linestyle='--', alpha=0.5)
    # add shading where DSWC can occur
    if shade_dswc is True:
        ax2.fill_between(time, ylim2[0], where=l_dswc, facecolor=dswc_shading, alpha=alpha)

    # --- wind ---
    ax3 = plt.subplot(4, 1, 1)
    ax3 = plot_wind_arrows_timeseries(time_wind, wind_vel, wind_dir, xlim=xlim,
                                      ax=ax3, show=False)
    ax3.set_xlim(xlim)
    ax3.set_xticklabels([])
    ax3.grid(True, linestyle='--', alpha=0.5)
    # add shading where DSWC can occur
    if shade_dswc is True:
        ylim3 = ax3.get_ylim()
        ax3.fill_between(time, ylim3[-1], where=l_dswc, facecolor=dswc_shading, alpha=alpha)
        ax3.set_ylim(ylim3)
    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

if __name__ == '__main__':
    
    # # --- ROMS temperature gradient data ---
    # input_dir = f'{get_dir_from_json("roms_data")}2017/'
    # start_date = datetime(2017, 3, 1)
    # end_date = datetime(2017, 8, 1)
    lon_c = 115.70 # coastal coordinate
    lat_c = -31.76
    lon_o = 114.47 # offshore coordinate
    lat_o = -31.80

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

    # --- Wind data ---
    wind_data = read_era5_wind_data(f'{get_dir_from_json("wind_data")}ERA5_winds_2017.nc')
    wind_vel, wind_dir = get_wind_vel_and_dir_in_point(wind_data, lon_o, lat_o)

    # --- Particle tracking data ---
    h_deep_sea = 600 # m depth: max Leeuwin Undercurrent depth
    input_path = f'{get_dir_from_json("opendrift")}cwa-perth_2017-Mar-Aug.nc'
    particles = Particles.read_from_netcdf(input_path)

    # -- plots ---
    output_dir = f'{get_dir_from_json("plots")}'

    time_str = f'{particles.time[0].year}-{particles.time[0].strftime("%b")}-{particles.time[-1].strftime("%b")}'
    output_path = f'{output_dir}cwa-perth_histogram_dswc_conditions_{time_str}.jpg'
    plot_particles_arriving_with_dswc_conditions(particles, h_deep_sea, time, temp_c, temp_o,
                                                 wind_data.time, wind_vel, wind_dir,
                                                 output_path=output_path, show=False)

