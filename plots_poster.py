from tools import log
from tools.files import get_dir_from_json
from tools.timeseries import add_month_to_time, convert_datetime_to_time
from data.kelp_data import KelpProbability
from data.bathymetry_data import BathymetryData
from data.roms_data import read_roms_grid_from_netcdf, read_roms_data_from_multiple_netcdfs, get_subgrid
from data.roms_data import read_roms_data_from_netcdf
from data.roms_data import get_cross_shelf_velocity_component, get_along_shelf_velocity_component, get_eta_xi_along_depth_contour
from data.roms_data import get_lon_lat_along_depth_contour, get_distance_along_transect
from data.glider_data import GliderData
from data.satellite_data import SatelliteSST, read_satellite_sst_from_netcdf
from data.wind_data import WindData, read_era5_wind_data_from_netcdf, get_daily_mean_wind_data, convert_u_v_to_meteo_vel_dir
from data.carbon_sequestration import read_carbon_fraction_from_netcdf, get_sequestration_fraction_at_depth_location
from particles import Particles, get_particle_density, DensityGrid
from plot_tools.basic_maps import plot_basic_map
from plot_tools.general import add_subtitle
from plot_tools.plots_bathymetry import plot_contours
from plot_tools.plots_particles import plot_age_in_deep_sea, plot_particle_age_in_deep_sea_depending_on_depth
from plot_tools.plots_sst import plot_sst
from location_info import LocationInfo, get_location_info
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib.patches import Polygon as mpl_polygon
from matplotlib.collections import PatchCollection
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import cmocean
import os
import shapefile
from shapely.geometry import Polygon, Point
from scipy.stats import pearsonr

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)

kelp_green = '#1b7931'
ocean_blue = '#25419e'
season_color = '#4f5478'

roms_grid = read_roms_grid_from_netcdf('input/cwa_roms_grid.nc')
bathy = BathymetryData.read_from_netcdf(f'{get_dir_from_json("bathymetry")}',
                                        lon_str='lon', lat_str='lat', h_str='z',
                                        h_fac=-1)

# decay rate from Simpkins et al. (in prep)
k = -0.075
k_sd = 0.031

# kelp density and carbon content from Filbee-Dexter & Wernberg (2020)
kelp_density = 6.3 # kelp/m2
carbon_content = 0.3 # fraction dry weight

def figure2(cmap_temp='RdBu_r', vmin_temp=18, vmax_temp=22,
            dz_interp=1, dt_interp=1/60,
            vmin_tempg=19.5, vmax_tempg=22.5,
            cmap_bbp=cmocean.cm.turbid, vmin_bbp=0, vmax_bbp=0.008,
            show=True, output_path=None):
    
    glider_all = GliderData.read_from_netcdf(f'{get_dir_from_json("glider_data")}IMOS_ANFOG_BCEOPSTUV_20220628T064224Z_SL286_FV01_timeseries_END-20220712T082641Z.nc')
    start_glider = datetime(2022, 6, 30, 22, 30)
    end_glider = datetime(2022, 7, 2, 15)
    glider = glider_all.get_data_in_time_frame(start_glider, end_glider)
    
    fig = plt.figure(figsize=(12, 5))
    # fig.set_facecolor('#456281')
    plt.subplots_adjust(wspace=1.1)
    plt.rcParams.update({'font.size': 18})
    
    # (a) Makuru mean SST
    sst = read_satellite_sst_from_netcdf(f'{get_dir_from_json("satellite_sst")}gsr_monthly_mean_makuru.nc')
    
    location_info = get_location_info('perth_wider')
    
    ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info, ymarkers='off', xmarkers='off', facecolor='#9d9d9d')
    ax1, c1, cbar1 = plot_sst(sst, location_info, ax=ax1, show=False, cmap=cmap_temp, vmin=vmin_temp, vmax=vmax_temp)
    ax1 = plot_contours(bathy.lon, bathy.lat, bathy.h, location_info,
                        ax=ax1, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    ax1.plot(glider.lon, glider.lat, '.k', label='Transect')
    cbar1.remove()
    # ax1.legend(loc='lower left')
    
    add_subtitle(ax1, 'June-July SST', alpha=0.5, location='lower left')
    
    # (b) July 2022 glider transect temperature
    depth_ticks = [-150, -100, -50, 0]
    depth_ticklabels = [150, 100, 50, 0]
    
    ax2 = plt.subplot(1, 3, (2, 3))
    ax2, c2, cbar2 = glider.plot_transect(parameter='temp', ax=ax2, show=False,
                                          cmap=cmap_temp, vmin=vmin_tempg, vmax=vmax_tempg,
                                          dz_interp=dz_interp, dt_interp=dt_interp,
                                          fill_color='#9d9d9d')
    ax2.set_yticks(depth_ticks)
    ax2.set_yticklabels(depth_ticklabels)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position('right')
    ax2.set_xticks([])
    ax2.set_xticklabels([])
    
    cbar2.remove()
    # l2, b2, w2, h2 = ax2.get_position().bounds
    # cbax2 = fig.add_axes([l2+w2+0.01, b2, 0.02, h2])
    # cbar2 = plt.colorbar(c2, cax=cbax2)
    # cbar2.set_label('Temperature ($^o$C)')
    
    add_subtitle(ax2, f'Temperature along transect', alpha=0.5, location='lower right')
    
    # reposition ax1
    l1, b1, w1, h1 = ax1.get_position().bounds
    l2, b2, w2, h2 = ax2.get_position().bounds
    ax1.set_position([l1, b2, h1/h2*w2, h2])
    
    l1, b1, w1, h1 = ax1.get_position().bounds
    # add ax1 colorbar
    cbax1 = fig.add_axes([l1-0.03, b1, 0.02, h1])
    cbar1 = plt.colorbar(c1, cax=cbax1)
    cbar1.set_label('Temperature ($^o$C)', labelpad=-70)
    cbar1.ax.yaxis.set_ticks_position('left')
    
    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)

        plt.close()

def figure5(particles:Particles, h_deep_sea=200,
            show=True, output_path=None,
            start_date = datetime(2017, 1, 1),
            end_date = datetime(2017, 12, 31)):
    
    fig = plt.figure(figsize=(17, 6))
    # fig.set_facecolor('#456281')
    plt.subplots_adjust(wspace=0.1)
    plt.rcParams.update({'font.size': 18})
    
    # (b) histogram decomposed particles passing shelf
    t_release = particles.get_release_time_index()
    p_ds, t_ds = particles.get_indices_arriving_in_deep_sea(h_deep_sea)
    
    times_release = particles.time[t_release]
    times_ds = particles.time[t_ds]
    time_bins = []
    for n in range(times_release[-1].month-times_release[0].month+2):
        time_bins.append(add_month_to_time(particles.time[0], n))
    n_releases, _ = np.histogram(times_release, bins=time_bins)
    
    total_particles = particles.lon.shape[0]
    n_releases_norm = n_releases/total_particles*100
    
    center_bins = np.array(time_bins[:-1]+np.diff(np.array(time_bins))/2)
    tick_labels = [center_bin.strftime("%b") for center_bin in center_bins]
    width = 0.8*np.array([dt.days for dt in np.diff(np.array(time_bins))])
    
    dt_ds = np.array([(particles.time[t_ds[i]]-particles.time[t_release[p_ds[i]]]).total_seconds()/(24*60*60) for i in range(len(p_ds))])
    f_ds = np.exp(k*dt_ds)
    f_ds_min = np.exp((k-k_sd)*dt_ds)
    f_ds_max = np.exp((k+k_sd)*dt_ds)
    times_ds_int, _ = convert_datetime_to_time(times_ds)
    time_bins_int, _ = convert_datetime_to_time(time_bins)
    i_bins = np.digitize(times_ds_int, bins=time_bins_int)
    
    f_ds_month = np.array([np.sum(f_ds[i_bins==i]) for i in range(1, len(center_bins)+1)])
    f_ds_month_norm = f_ds_month/total_particles*100
    f_ds_month_min = np.array([np.sum(f_ds_min[i_bins==i]) for i in range(1, len(center_bins)+1)])
    f_ds_month_norm_min = f_ds_month_min/total_particles*100
    f_ds_month_max = np.array([np.sum(f_ds_max[i_bins==i]) for i in range(1, len(center_bins)+1)])
    f_ds_month_norm_max = f_ds_month_max/total_particles*100
    
    yerr = np.abs(np.array([f_ds_month_norm_min, f_ds_month_norm_max])-f_ds_month_norm)
    
    ax1 = plt.subplot(1, 2, 1)
    # ax1.set_facecolor('#456281')
    ax1.bar(center_bins, f_ds_month_norm, tick_label=tick_labels, width=width, color='#dfc64d', yerr=yerr, ecolor='#c09e30')
    ax1.set_ylabel('Particles passing shelf (%)')
    ax1.set_ylim([0, 11.5])
    add_subtitle(ax1, 'Decomposing kelp export', alpha=0.5)
    ax1.set_xlim([time_bins[0], time_bins[-1]])
    
    # ax2 = ax1.twinx()
    # ax2.plot(center_bins, n_releases_norm, 'xk', label='Particles released')
    # ax2.set_ylabel('Particles released (%)')
    # ax2.set_ylim([0, 28.75])
    # ax2.set_xlim([time_bins[0], time_bins[-1]])
    
    # (a) histogram dswc occurrence
    location_info_perth = get_location_info('perth')
    wind_data = read_era5_wind_data_from_netcdf(get_dir_from_json("era5_data"), start_date, end_date,
                                                lon_range=location_info_perth.lon_range,
                                                lat_range=location_info_perth.lat_range)
    wind_data = get_daily_mean_wind_data(wind_data)
    u_mean = np.nanmean(np.nanmean(wind_data.u, axis=1), axis=1)
    v_mean = np.nanmean(np.nanmean(wind_data.v, axis=1), axis=1)
    vel_mean, dir_mean = convert_u_v_to_meteo_vel_dir(u_mean, v_mean)

    def read_dswc_components(csv_gw='temp_data/gravitational_wind_components_in_time.csv'):
        if not os.path.exists(csv_gw):
            raise ValueError(f'''Gravitational vs wind components file does not yet exist: {csv_gw}
                                Please create it first by running write_gravitation_wind_components_to_csv (in dswc_detector.py)''')
        df = pd.read_csv(csv_gw)
        time_gw = [datetime.strptime(t, '%Y-%m-%d') for t in df['time'].values][:-1]
        grav_c = df['grav_component'].values[:-1]
        wind_c = df['wind_component'].values[:-1]
        drhodx = df['drhodx'].values[:-1]
        phi = df['phi'].values[:-1]
        return time_gw, grav_c, wind_c, drhodx, phi

    def determine_l_time_dwswc_conditions(dir_mean):
        time, g, w, drhodx, phi = read_dswc_components()
        l_drhodx = drhodx < 0
        l_phi = phi > 5
        l_prereq = np.logical_and(l_drhodx, l_phi)
        l_components = g > w
        l_onshore = np.logical_and(225 < dir_mean, dir_mean < 315)
        l_wind = np.logical_or(l_components, l_onshore)
        l_dswc = np.logical_and(l_prereq, l_components)
        return l_drhodx, l_phi, l_components, l_wind, l_onshore, l_dswc
        
    time_gw, grav_c, wind_c, drhodx, phi = read_dswc_components()
    l_drhodx, l_phi, l_components, l_wind, l_onshore, l_dswc = determine_l_time_dwswc_conditions(dir_mean)
    
    month_dswc = []
    p_dswc = []
    for n in range(time_gw[0].month, time_gw[-1].month+1):
        l_time = [t.month == n for t in time_gw]
        month_dswc.append(datetime(time_gw[0].year, n, 1))
        p_dswc.append(np.sum(l_dswc[l_time])/np.sum(l_time))
        
    month_dswc = np.array(month_dswc)
    month_dswc_extra_month_added = np.append(month_dswc, add_month_to_time(month_dswc[-1], 1))
    center_month_dswc = np.array(month_dswc_extra_month_added[:-1]+np.diff(month_dswc_extra_month_added)/2)
    str_month_dswc = np.array([t.strftime('%b') for t in month_dswc])
    p_dswc = np.array(p_dswc)
    width_dswc = 0.8*np.array([dt.days for dt in np.diff(month_dswc_extra_month_added)])
        
    ax4 = plt.subplot(1, 2, 2)
    # ax4.set_facecolor('#456281')
    ax4.bar(center_month_dswc, p_dswc*100, color='#456281', tick_label=str_month_dswc, width=width_dswc)
    ax4.set_ylabel('Suitable conditions (% of time)')
    # ax4.yaxis.set_label_position("right")
    # ax4.yaxis.tick_right()
    ax4.set_ylim([0, 100])
    # ax4.tick_params(axis='y', colors=ocean_blue)
    # ax4.yaxis.label.set_color(ocean_blue)
    add_subtitle(ax4, 'Dense shelf water transport', alpha=0.5)
    ax4.set_xlim([time_bins[0], time_bins[-1]])
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position('right')

    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)

        plt.close()

def plot_global_map(output_path:str):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, edgecolor='k', facecolor='#9d9d9d')
    ax.add_feature(cfeature.COASTLINE)
    ax.axis('off')
    
    log.info(f'Saving figure to: {output_path}')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)

    plt.close()

def plot_australia_map(output_path:str):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, edgecolor='k', facecolor='#9d9d9d')
    ax.add_feature(cfeature.COASTLINE)
    ax.plot([115.0, 116.0, 116.0, 115.0, 115.0],
            [-32.6, -32.6, -31.5, -31.5, -32.6], '-', color='#dfc64d', linewidth=5.0)
    ax.set_extent([113.0, 155.0, -45.0, -10.0], ccrs.PlateCarree())
    ax.axis('off')
    
    log.info(f'Saving figure to: {output_path}')
    plt.savefig(output_path, bbox_inches='tight', dpi=300, transparent=True)

    plt.close() 

if __name__ == '__main__':

    plot_dir = 'plots/poster/'

    figure2(output_path=f'{plot_dir}dswt.png', show=False)
    
    particle_path = f'{get_dir_from_json("opendrift_output")}cwa_perth_JanFeb2018_baseline-fy.nc'
    particles = Particles.read_from_netcdf(particle_path)
    figure5(particles, output_path=f'{plot_dir}export.png', show=False)
    
    plot_global_map(f'{plot_dir}global_map.png')
    plot_australia_map(f'{plot_dir}australia_map.png')
