from tools import log
from tools.files import get_dir_from_json, get_daily_files_in_time_range
from tools.timeseries import convert_time_to_datetime, get_l_time_range, get_daily_means, get_closest_time_index
from tools.coordinates import get_transect_lons_lats_ds_from_json

from data.kelp_data import KelpProbability
from data.bathymetry_data import BathymetryData
from data.wind_data import read_era5_wind_data, get_wind_vel_and_dir_in_point, get_wind_dir_and_text, get_wind_data_in_point
from data.roms_data import RomsData, RomsGrid, read_roms_data_from_multiple_netcdfs, read_roms_grid_from_netcdf
from data.roms_data import get_roms_data_for_transect, get_depth_integrated_gradient_along_transect

from plot_tools.basic_maps import plot_basic_map
from plot_tools.plots_particles import plot_timeseries_in_deep_sea, plot_age_in_deep_sea
from plot_tools.plots_particles import plot_histogram_arriving_in_deep_sea
from plot_tools.plots_bathymetry import plot_contours
from plot_tools.plots_roms import plot_depth_integrated_gradient_along_transect

from location_info import LocationInfo, get_location_info
from particles import Particles

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import matplotlib.animation as animation
from matplotlib.lines import Line2D
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
                                                 time:np.ndarray, gradient_values:np.ndarray, gradient_parameter:str,
                                                 time_wind:np.ndarray, wind_vel:np.ndarray, wind_dir:np.ndarray, wind_u:np.ndarray,
                                                 color_temp='k', color_hist='#1b7931', shade_dswc=False,
                                                 color_vel='k', color_dir='r',
                                                 show=True, output_path=None):

    daily_time_wind, daily_mean_wind_u = get_daily_means(time_wind, wind_u)
    daily_time, daily_mean_gradient = get_daily_means(time, gradient_values)

    dswc_shading = '#e4e4e4'
    alpha = 0.4
    l_dswc = gradient_values < 0
    ylim2 = [-4, 4] # limits for density gradient
    ylim3 = [0, 15] # limits for onshore wind speed (m/s)
    ylim4 = [-15, 0] # limits for offshore wind speed (m/s)
    xlim = [particles.time[0], particles.time[-1]]

    fig = plt.figure(figsize=(10, 8))

    # --- particles histogram ---
    ax = plt.subplot(6, 1, (5, 6))
    ax = plot_histogram_arriving_in_deep_sea(particles, h_deep_sea, ax=ax, show=False, color=color_hist)
    if shade_dswc is True:
        # add shading where DSWC can occur
        ylim1 = ax.get_ylim()
        ax.fill_between(time, ylim1[1], where=l_dswc, facecolor=dswc_shading, alpha=alpha)
        ax.set_ylim(ylim1)
    ax.set_xlim(xlim)
    ax.grid(True, linestyle='--', alpha=0.5)

    # --- density gradient ---
    ax2 = plt.subplot(6, 1, 4)
    ax2 = plot_depth_integrated_gradient_along_transect(daily_time, daily_mean_gradient, gradient_parameter, ax=ax2, show=False)
    ax2.plot([time[0], time[-1]], [0, 0], '--k')
    ax2.set_xlim(xlim)
    ax2.set_xticklabels([])
    # ax2.set_ylim(ylim2)
    ax2.grid(True, linestyle='--', alpha=0.5)
    # add shading where DSWC can occur
    if shade_dswc is True:
        ax2.fill_between(time, ylim2[0], where=l_dswc, facecolor=dswc_shading, alpha=alpha)

    # --- onshore wind ---
    ax3 = plt.subplot(6, 1, 3)
    l_onshore = daily_mean_wind_u>0
    ax3.plot(daily_time_wind[l_onshore], daily_mean_wind_u[l_onshore], '-k')
    ax3.set_ylabel('Onshore\nwind (m/s)')
    ax3.set_ylim(ylim3)
    ax3.set_xlim(xlim)
    ax3.set_xticklabels([])
    ax3.grid(True, linestyle='--', alpha=0.5)
    # add shading where DSWC can occur
    if shade_dswc is True:
        ylim3 = ax3.get_ylim()
        ax3.fill_between(time, ylim3[-1], where=l_dswc, facecolor=dswc_shading, alpha=alpha)
        ax3.set_ylim(ylim3)
    
    # --- offshore wind ---
    ax4 = plt.subplot(6, 1, 2)
    l_offshore = ~l_onshore
    ax4.plot(daily_time_wind[l_offshore], daily_mean_wind_u[l_offshore], '-k')
    ax4.set_ylabel('Offshore\nwind (m/s)')
    ax4.set_ylim(ylim4)
    ax4.set_xlim(xlim)
    ax4.set_xticklabels([])
    ax4.grid(True, linestyle='--', alpha=0.5)
    # add shading where DSWC can occur
    if shade_dswc is True:
        ylim4 = ax4.get_ylim()
        ax4.fill_between(time, ylim4[-1], where=l_dswc, facecolor=dswc_shading, alpha=alpha)
        ax4.set_ylim(ylim4)

    # --- wind arrows seabreeze style ---
    ax5 = plt.subplot(6, 1, 1)
    ax5 = plot_wind_arrows_timeseries(time_wind, wind_vel, wind_dir, xlim=xlim,
                                      ax=ax5, show=False)
    ax5.set_ylabel('Wind speed\n(m/s)')
    ax5.set_xlim(xlim)
    ax5.set_xticklabels([])

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

# THIS ANIMATION ONLY WORKS IF THE OUTPUT TIMES OF PARTICLES IS THE SAME AS THAT OF ROMS!!!
def animate_particles_with_roms_field(particles:Particles, roms_input_dir:str,
                                      location_info:LocationInfo, h_deep_sea:float,
                                      parameter='temp', s=0,
                                      cmap='RdBu_r', clabel='Bottom temperature ($^o$C)', vmin=18, vmax=22,
                                      show_bathymetry=True, show_kelp_map=False,
                                      output_path=None, color_p='k', color_ds='#1b7931',
                                      dpi=100, fps=10):
    
    roms_grid = read_roms_grid_from_netcdf('input/cwa_roms_grid.nc')

    l_deep_sea = particles.get_l_deep_sea(h_deep_sea).astype(bool)

    writer = animation.PillowWriter(fps=fps)

    # plot map
    plt.rcParams.update({'font.size' : 15})
    plt.rcParams.update({'font.family': 'arial'})
    plt.rcParams.update({'figure.dpi': dpi})
    fig = plt.figure(figsize=(10,8))
    fig.tight_layout()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plot_basic_map(ax, location_info)

    if show_bathymetry is True:
        bathymetry = BathymetryData.read_from_netcdf('input/cwa_roms_grid.nc')
        ax = plot_contours(bathymetry.lon, bathymetry.lat, bathymetry.h, location_info,
                           highlight_contour=[h_deep_sea], ax=ax, show=False, color='#757575', show_perth_canyon=False)
    
    if show_kelp_map is True:
        kelp_prob = KelpProbability.read_from_tiff('input/perth_kelp_probability.tif')
        ax = kelp_prob.plot(location_info, ax=ax, show=False)

    # animated points
    point = ax.plot([], [], 'o', color=color_p, markersize=2, zorder=2)[0]
    point_ds = ax.plot([], [], 'o', color=color_ds, markersize=2, zorder=3)[0]
    # animated field
    field = ax.pcolormesh(roms_grid.lon, roms_grid.lat, np.zeros(roms_grid.lon.shape), cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(field)
    cbar.set_label(clabel)
    
    # legend
    legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor=color_p, markersize=10,
                       label='Coastal region'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=color_ds, markersize=10,
                       label='Past shelf break')]
    ax.legend(handles=legend_elements, loc='upper left')

    # animated text
    ttl = ax.text(0.5, 1.04, '', transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(facecolor='w', alpha=0.3, edgecolor='w', pad=2))
    ttl.set_animated(True)

    def animate(i):
        x, y = (particles.lon[~l_deep_sea[:, i], i], particles.lat[~l_deep_sea[:, i], i])
        point.set_data(x, y)
        x_ds, y_ds = (particles.lon[l_deep_sea[:, i], i], particles.lat[l_deep_sea[:, i], i])
        point_ds.set_data(x_ds, y_ds)

        roms_data = read_roms_data_from_multiple_netcdfs(roms_input_dir, particles.time[i], particles.time[i])
        values = getattr(roms_data, parameter)[0, s, :, :]
        field.set_array(values.ravel())

        title = particles.time[i].strftime('%d %b %Y %H:%M')
        ttl.set_text(title)

        return point, point_ds, ttl

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(particles.time), blit=True)
    if output_path is not None:
        log.info(f'Saving animation to: {output_path}')
        anim.save(output_path, writer=writer)
    else:
        plt.show()

if __name__ == '__main__':

    location_info = get_location_info('cwa_perth')

    # --- Particle tracking data ---
    h_deep_sea = 600 # m depth: max Leeuwin Undercurrent depth
    input_path = f'{get_dir_from_json("opendrift")}cwa-perth_2017-Mar-Aug.nc'
    particles = Particles.read_from_netcdf(input_path)
    
    # --- ROMS transect and density gradient data ---
    roms_input_dir = f'{get_dir_from_json("roms_data")}2017/'
    start_date = datetime(2017, 3, 1)
    end_date = datetime(2017, 8, 1)
    lon1, lat1, lon2, lat2, ds = get_transect_lons_lats_ds_from_json('two_rocks_glider')
    roms_data = get_roms_data_for_transect(roms_input_dir, start_date, end_date, lon1, lat1, lon2, lat2)
    density_gradient, density, distance, z = get_depth_integrated_gradient_along_transect(roms_data,
                                                                                          'density',
                                                                                          lon1, lat1,
                                                                                          lon2, lat2,
                                                                                          ds)
    
    # --- Wind data ---
    wind_data = read_era5_wind_data(f'{get_dir_from_json("wind_data")}ERA5_winds_2017.nc')
    wind_vel, wind_dir = get_wind_vel_and_dir_in_point(wind_data, lon2, lat2)
    wind_data_p = get_wind_data_in_point(wind_data, lon2, lat2)

    # -- Plots ---
    output_dir = f'{get_dir_from_json("plots")}'

    time_str = f'{particles.time[0].year}-{particles.time[0].strftime("%b")}-{particles.time[-1].strftime("%b")}'
    output_path = f'{output_dir}cwa-perth_histogram_dswc_conditions_{time_str}.jpg'
    plot_particles_arriving_with_dswc_conditions(particles, h_deep_sea, roms_data.time, density_gradient, 'density',
                                                 wind_data.time, wind_vel, wind_dir, wind_data_p.u,
                                                 output_path=output_path, show=False)

    # output_path = f'{output_dir}cwa-perth_animation_with_bottom_temp_2017-Mar-Aug.gif'
    # animate_particles_with_roms_field(particles, roms_input_dir, location_info, h_deep_sea, output_path=output_path)

