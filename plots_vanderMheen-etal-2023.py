from tools import log
from tools.files import get_dir_from_json
from tools.timeseries import add_month_to_time, convert_datetime_to_time
from tools.coordinates import get_index_closest_point
from data.kelp_data import KelpProbability
from data.roms_data import read_roms_grid_from_netcdf, read_roms_data_from_multiple_netcdfs, get_subgrid
from data.roms_data import get_cross_shelf_velocity_component, get_along_shelf_velocity_component, get_eta_xi_along_depth_contour
from data.roms_data import get_lon_lat_along_depth_contour, get_distance_along_transect
from data.glider_data import GliderData
from data.satellite_data import SatelliteSST, read_satellite_sst_from_netcdf
from data.wind_data import WindData, read_era5_wind_data_from_netcdf, get_daily_mean_wind_data, convert_u_v_to_meteo_vel_dir
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
import numpy as np
import pandas as pd
import cmocean
import os
import shapefile
from shapely.geometry import Polygon, Point

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

k = -0.075
k_sd = 0.031

def save_bottom_cross_shelf_velocities(location_name='perth_wide',
                                       h_level=100,
                                       max_s_layer=3):
    location_info = get_location_info(location_name)

    roms_dir = f'{get_dir_from_json("roms_data")}2017/'
    date0 = datetime(2017, 1, 1)
    n_months = 12
    time_monthly = []
    u_cross_monthly_spatial = []
    v_along_monthly_spatial = []
    for n in range(n_months):
        start_date = add_month_to_time(date0, n)
        end_date = add_month_to_time(date0, n+1)-timedelta(days=1)
        roms = read_roms_data_from_multiple_netcdfs(roms_dir, start_date, end_date,
                                                    lon_range=location_info.lon_range,
                                                    lat_range=location_info.lat_range)
        u_cross = get_cross_shelf_velocity_component(roms)
        v_along = get_along_shelf_velocity_component(roms)
        eta, xi = get_eta_xi_along_depth_contour(roms.grid, h_level=h_level)
        # monthly means
        time_monthly.append(start_date)
        u_cross_monthly_spatial.append(np.nanmean(np.nanmean(u_cross[:, 0:max_s_layer, eta, xi], axis=0), axis=0))
        v_along_monthly_spatial.append(np.nanmean(np.nanmean(v_along[:, 0:max_s_layer, eta, xi], axis=1), axis=0))

    df_s = pd.DataFrame(np.array(u_cross_monthly_spatial).transpose(), columns=time_monthly)
    df_s.to_csv(f'temp_data/{location_name}_monthly_mean_u_cross_{h_level}m.csv', index=False)
    
    df_sv = pd.DataFrame(np.array(v_along_monthly_spatial).transpose(), columns=time_monthly)
    df_sv.to_csv(f'temp_data/{location_name}_monthly_mean_v_along_{h_level}m.csv', index=False)

def save_distance_along_depth_contour(location_name='perth_wide', h_level=100):
    location_info = get_location_info(location_name)
    grid = get_subgrid(roms_grid, lon_range=location_info.lon_range,
                                  lat_range=location_info.lat_range, s_range=None)
    lons, lats = get_lon_lat_along_depth_contour(grid)
    distance = get_distance_along_transect(lons, lats)
    
    df = pd.DataFrame(np.array([distance, lons, lats]).transpose(),
                      columns=['distance', 'lons', 'lats'])
    df.to_csv(f'temp_data/{location_name}_distance_{h_level}m.csv', index=False)

def figure1(show=True, output_path=None):

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(wspace=0.5)

    # (a) kelp probability map
    location_info_p = get_location_info('perth')
    kelp_prob = KelpProbability.read_from_tiff('input/perth_kelp_probability.tif')
    ax1 = plt.subplot(3, 3, (1, 4), projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info_p)
    ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_p, ax=ax1, show=False, show_perth_canyon=False, color='k', linewidths=0.7)
    ax1, cbar1, c1 = kelp_prob.plot(location_info_p, ax=ax1, show=False)
    cbar1.remove()
    l1, b1, w1, h1 = ax1.get_position().bounds
    cbax1 = fig.add_axes([l1+w1+0.01, b1, 0.02, h1])
    cbar1 = plt.colorbar(c1, cax=cbax1)
    cbar1.set_label('Probability of kelp')
    add_subtitle(ax1, '(a) Perth kelp forests')

    # (c) bathymetry and oceanography overview
    location_info_pw_mc = get_location_info('perth_wide_more_contours')
    ax3 = plt.subplot(3, 3, (2, 6), projection=ccrs.PlateCarree())
    ax3 = plot_basic_map(ax3, location_info_pw_mc, ymarkers='off')
    ax3 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_pw_mc, ax=ax3, show=False, show_perth_canyon=True, color='k', linewidths=0.7)
    c3 = ax3.pcolormesh(roms_grid.lon, roms_grid.lat, roms_grid.h, vmin=0, vmax=2000, cmap=cmocean.cm.deep)
    l3, b3, w3, h3 = ax3.get_position().bounds
    cbax3 = fig.add_axes([l3+w3+0.01, b3, 0.02, h3])
    cbar3 = plt.colorbar(c3, cax=cbax3)
    cbar3.set_label('Bathymetry (m)')
    add_subtitle(ax3, '(c) Main oceanographic features')
    # ax3.set_position([l3+0.02, b3+0.05, w3, h3])

    # (b) detritus production from de Bettignies et al. (2013)
    time_detritus = ['Mar-May', 'Jun-Aug', 'Sep-Nov', 'Dec-Feb']
    detritus = [4.8, 2.12, 0.70, 0.97]
    detritus_sd = [1.69, 0.84, 0.45, 0.81]

    ax2 = plt.subplot(3, 3, 7)
    ax2.bar(np.arange(len(time_detritus)), detritus, color=kelp_green, tick_label=time_detritus, yerr=detritus_sd)
    ax2.set_ylim([0, 8.0])
    ax2.set_ylabel('Detritus production\n(g/kelp/day)')
    add_subtitle(ax2, '(b) Seasonal detritus production')
    l2, b2, w2, h2 = ax2.get_position().bounds
    ax2.set_position([l1, b2, w1+0.03, h2])

    # (d) cross-shore transport histogram
    csv_ucross = 'temp_data/perth_wide_monthly_mean_u_cross_100m.csv'
    if not os.path.exists(csv_ucross):
        raise ValueError(f'''Mean cross-shelf velocity file does not yet exist: {csv_ucross}
                         Please create it first by running save_bottom_cross_shelf_velocities''')
    df = pd.read_csv(csv_ucross)
    time_cross = [datetime.strptime(d, '%Y-%m-%d') for d in df.columns.values]
    str_time_cross = [d.strftime('%b') for d in time_cross]
    ucross = -np.array([np.nanmean(df.iloc[:, i].values) for i in range(len(df.columns))])
    # minus sign so cross-shelf velocity follows oceanographic conventions again
    # ucross is calculated as perpendicular to a depth contour, and the depth contour axis points offshore

    csv_valong = 'temp_data/perth_wide_monthly_mean_v_along_100m.csv'
    if not os.path.exists(csv_valong):
        raise ValueError(f'''Mean along-shelf velocity file does not yet exist: {csv_valong}
                         Please create it first by running save_bottom_cross_shelf_velocities''')
    df_v = pd.read_csv(csv_valong)
    valong = [np.nanmean(df_v.iloc[:, i].values) for i in range(len(df_v.columns))]

    ax4 = plt.subplot(3, 3, (8, 9))
    ax4.bar(np.arange(len(time_cross)), ucross, color=ocean_blue, tick_label=str_time_cross)
    xlim = ax4.get_xlim()
    ax4.plot([-1, 12], [0, 0], '-k')
    ax4.set_xlim(xlim)
    ax4.set_ylim([-0.05, 0.05])
    ax4.set_ylabel('Cross-shelf transport (m/s)')
    ax4.yaxis.label.set_color(ocean_blue)
    ax4.tick_params(axis='y', colors=ocean_blue)
    ax4.spines['right'].set_color(ocean_blue)
    add_subtitle(ax4, '(d) Monthly mean bottom cross-shelf transport')
    l4, b4, w4, h4 = ax4.get_position().bounds
    ax4.set_position([l3, b4, w3, h4])
    
    ax5 = ax4.twinx()
    ax5.plot(np.arange(len(time_cross)), ucross/valong*100, 'xk')
    ax5.set_xlim(xlim)
    ax5.set_ylim([-20, 20])
    ax5.set_yticks([-16, -8, 0, 8, 16])
    ax5.set_ylabel('Cross-shelf transport\n(% along-shelf)')
    
    # swap y-axes
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    ax5.yaxis.set_label_position("left")
    ax5.yaxis.tick_left()
    
    # season texts
    ax4.text(0.5/13, -0.2, 'Birak', ha='center', color=season_color, transform=ax4.transAxes)
    ax4.text(2.5/13, -0.2, 'Bunuru', ha='center', color=season_color, transform=ax4.transAxes)
    ax4.text(4.5/13, -0.2, 'Djeran', ha='center', color=season_color, transform=ax4.transAxes)
    ax4.text(6.5/13, -0.2, 'Makuru', ha='center', color=season_color, transform=ax4.transAxes)
    ax4.text(8.5/13, -0.2, 'Djilba', ha='center', color=season_color, transform=ax4.transAxes)
    ax4.text(10.5/13, -0.2, 'Kambarang', ha='center', color=season_color, transform=ax4.transAxes)
    ax4.text(12.5/13, -0.2, 'Birak', ha='center', color=season_color, transform=ax4.transAxes)

    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

        plt.close()

def figure2(cmap_temp='RdBu_r', vmin_temp=18, vmax_temp=22,
            dz_interp=1, dt_interp=1/60,
            vmin_tempg=19.5, vmax_tempg=22.5,
            cmap_bbp=cmocean.cm.turbid, vmin_bbp=0, vmax_bbp=0.008,
            show=True, output_path=None):
    
    glider_all = GliderData.read_from_netcdf(f'{get_dir_from_json("glider_data")}IMOS_ANFOG_BCEOPSTUV_20220628T064224Z_SL286_FV01_timeseries_END-20220712T082641Z.nc')
    start_glider = datetime(2022, 6, 30, 22, 30)
    end_glider = datetime(2022, 7, 2, 15)
    glider = glider_all.get_data_in_time_frame(start_glider, end_glider)
    
    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(wspace=1.8)
    
    # (a) Makuru mean SST
    sst = read_satellite_sst_from_netcdf(f'{get_dir_from_json("satellite_sst")}gsr_monthly_mean_makuru.nc')
    
    location_info = get_location_info('perth_wider')
    
    ax1 = plt.subplot(3, 5, (1, 12), projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info, ymarkers='right')
    ax1, c1, cbar1 = plot_sst(sst, location_info, ax=ax1, show=False, cmap=cmap_temp, vmin=vmin_temp, vmax=vmax_temp)
    ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info,
                        ax=ax1, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    ax1.plot(glider.lon, glider.lat, '.k', label='Ocean glider transect')
    cbar1.remove()
    ax1.legend(loc='lower left')
    l1, b1, w1, h1 = ax1.get_position().bounds
    
    add_subtitle(ax1, '(a) Makuru (JJ) mean SST')
    
    # (b) July 2022 glider transect temperature
    depth_ticks = [-150, -100, -50, 0]
    depth_ticklabels = [150, 100, 50, 0]
    
    ax2 = plt.subplot(3, 5, (8, 10))
    ax2, c2, cbar2 = glider.plot_transect(parameter='temp', ax=ax2, show=False,
                                          cmap=cmap_temp, vmin=vmin_tempg, vmax=vmax_tempg,
                                          dz_interp=dz_interp, dt_interp=dt_interp)
    ax2.set_yticks(depth_ticks)
    ax2.set_yticklabels(depth_ticklabels)
    
    cbar2.remove()
    l2, b2, w2, h2 = ax2.get_position().bounds
    cbax2 = fig.add_axes([l2+w2+0.01, b2, 0.02, h2])
    cbar2 = plt.colorbar(c2, cax=cbax2)
    cbar2.set_label('Temperature ($^o$C)')
    
    add_subtitle(ax2, f'(b) Ocean glider temperatures: {start_glider.strftime("%d %b %Y")} - {end_glider.strftime("%d %b %Y")}',
                 location='lower right')
    
    # (c) July 2022 glider transect backscatter
    ax3 = plt.subplot(3, 5, (13, 15))
    ax3, c3, cbar3 = glider.plot_transect(parameter='bbp', ax=ax3, show=False,
                                          cmap=cmap_bbp, vmin=vmin_bbp, vmax=vmax_bbp,
                                          dz_interp=dz_interp, dt_interp=dt_interp)
    ax3.set_yticks(depth_ticks)
    ax3.set_yticklabels(depth_ticklabels)
    
    cbar3.remove()
    l3, b3, w3, h3 = ax3.get_position().bounds
    cbax3 = fig.add_axes([l3+w3+0.01, b3, 0.02, h3])
    cbar3 = plt.colorbar(c3, cax=cbax3)
    cbar3.set_label('Particle backscatter (m$^{-1}$)')
    
    add_subtitle(ax3, f'(c) Ocean glider backscatter: {start_glider.strftime("%d %b %Y")} - {end_glider.strftime("%d %b %Y")}',
                 location='lower right')
    
    # move ax1
    ax1.set_position([l1, b3, w1, h1])
    
    # add ax1 colorbar
    cbax1 = fig.add_axes([l1-0.03, b3, 0.02, h1])
    cbar1 = plt.colorbar(c1, cax=cbax1)
    cbar1.set_label('Satellite mean sea surface temperature ($^o$C)', labelpad=-70)
    cbar1.ax.yaxis.set_ticks_position('left')
    
    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

        plt.close()

def figure3(particles:Particles, dx=0.02,
            cmap='summer',
            show=True, output_path=None):
    
    t_release = particles.get_release_time_index()
    
    location_info = get_location_info('cwa_perth_less_contours')
    grid = DensityGrid(location_info.lon_range, location_info.lat_range, dx)
    x, y = np.meshgrid(grid.lon, grid.lat)
    
    vmin = 0
    vmax = 3
    tick_labels = ['1', '10', '100', '10$^3$']#, '10$^4$', '10$^5$', '10$^6$']
    fig = plt.figure(figsize=(12, 6))
    
    # (a) Bunuru (end of March)
    l_bunuru = np.array([particles.time[t_release[i]].month == 3 for i in range(len(t_release))])
    t_bunuru = np.where(np.array([particles.time[i].date()==date(2017, 3, 31) for i in range(len(particles.time))]))[0]
    lon_bunuru = particles.lon[l_bunuru, :][:, t_bunuru]
    lat_bunuru = particles.lat[l_bunuru, :][:, t_bunuru]
    density_bunuru = get_particle_density(grid, lon_bunuru, lat_bunuru)
    density_bunuru[density_bunuru==0.] = np.nan
    
    ax3 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax3 = plot_basic_map(ax3, location_info)
    ax3 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info,
                        ax=ax3, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    
    c3 = ax3.pcolormesh(x, y, np.log10(density_bunuru), vmin=vmin, vmax=vmax, cmap=cmap)
    add_subtitle(ax3, '(a) Bunuru (end of March)')
    
    # (b) Djeran (end of May)
    l_djeran = np.array([particles.time[t_release[i]].month in [4, 5] for i in range(len(t_release))])
    t_djeran = np.where(np.array([particles.time[i].date()==date(2017, 5, 31) for i in range(len(particles.time))]))[0]
    lon_djeran = particles.lon[l_djeran, :][:, t_djeran]
    lat_djeran = particles.lat[l_djeran, :][:, t_djeran]
    density_djeran = get_particle_density(grid, lon_djeran, lat_djeran)
    density_djeran[density_djeran==0.] = np.nan
    
    ax1 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info)
    ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info,
                        ax=ax1, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    
    c1 = ax1.pcolormesh(x, y, np.log10(density_djeran), vmin=vmin, vmax=vmax, cmap=cmap)
    add_subtitle(ax1, '(b) Djeran (end of May)')
    
    # (c) Makuru (end of July)
    l_makuru = np.array([particles.time[t_release[i]].month in [6, 7] for i in range(len(t_release))])
    t_makuru = np.where(np.array([particles.time[i].date()==date(2017, 7, 31) for i in range(len(particles.time))]))[0]
    lon_makuru = particles.lon[l_makuru, :][:, t_makuru]
    lat_makuru = particles.lat[l_makuru, :][:, t_makuru]
    density_makuru = get_particle_density(grid, lon_makuru, lat_makuru)
    density_makuru[density_makuru==0.] = np.nan
    
    ax2 = plt.subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax2 = plot_basic_map(ax2, location_info)
    ax2 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info,
                        ax=ax2, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    
    c2 = ax2.pcolormesh(x, y, np.log10(density_makuru), vmin=vmin, vmax=vmax, cmap=cmap)
    add_subtitle(ax2, '(c) Makuru (end of July)')
    
    # colorbar
    l2, b2, w2, h2 = ax2.get_position().bounds
    cbax2 = fig.add_axes([l2+w2+0.01, b2, 0.02, h2])
    cbar2 = plt.colorbar(c2, cax=cbax2)
    cbar2.set_label(f'Particle density (# per {dx}$^o$ grid cell)')
    cbar2.set_ticks(np.arange(vmin, vmax+1, 1))
    cbar2.set_ticklabels(tick_labels)
    
    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

        plt.close()

def figure4(particles:Particles, h_deep_seas=[200, 400, 600, 800, 1000],
            colors=[kelp_green, kelp_green, kelp_green, kelp_green, 'k'],
            linestyles=['-', '--', ':', '-.', '--'],
            show=True, output_path=None):
    
    if output_path is not None:
        output_path_csv_p = f'{os.path.splitext(output_path)[0]}a.csv'
        output_path_csv_d = f'{os.path.splitext(output_path)[0]}b.csv'
    else:
        output_path_csv_p = None
        output_path_csv_d = None
    
    fig = plt.figure(figsize=(11, 6))
    
    xlim = [0, 120]
    
    # (a) age in deep sea for different depths
    ax1 = plt.subplot(1, 2, 1)
    ax1, l1 = plot_particle_age_in_deep_sea_depending_on_depth(particles, h_deep_sea_sensitivity=h_deep_seas,
                                                               linestyles=linestyles,
                                                               colors=colors,
                                                               ax=ax1, show=False,
                                                               output_path_csv=output_path_csv_p)
    ax1.set_ylabel('Particles past depth range (%)')
    ax1.set_xlim(xlim)
    ax1.set_ylim([0, 70])
    add_subtitle(ax1, '(a) Particle export')
    
    l1.remove()
    
    # (b) age in deep sea with decomposition
    ax2 = plt.subplot(1, 2, 2)
    
    for i, h_deep_sea in enumerate(h_deep_seas):
        _, age_arriving_ds, matrix_arriving_ds = particles.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
        n_deep_sea_per_age = np.sum(matrix_arriving_ds, axis=0)
        n_deep_sea_decomposed = n_deep_sea_per_age*np.exp(k*age_arriving_ds)
        total_particles = particles.lon.shape[0]
        f_deep_sea_decomposed = n_deep_sea_decomposed/total_particles*100
        f_decomposed = np.cumsum(f_deep_sea_decomposed)

        ax2.plot(age_arriving_ds, f_decomposed, color=colors[i], linestyle=linestyles[i], label=h_deep_sea)
        
        if i == 0:
            df = pd.DataFrame(np.array([age_arriving_ds, f_decomposed]).transpose(),
                              columns=['age (days)', f'fraction past {h_deep_sea}'])
        else:
            df[f'fraction past {h_deep_sea}'] = f_decomposed
        
        if h_deep_sea == 200:
            n_deep_sea_decomposed_min = n_deep_sea_per_age*np.exp((k-k_sd)*age_arriving_ds)
            n_deep_sea_decomposed_max = n_deep_sea_per_age*np.exp((k+k_sd)*age_arriving_ds)
            f_decomposed_min = np.cumsum(n_deep_sea_decomposed_min/total_particles*100)
            f_decomposed_max = np.cumsum(n_deep_sea_decomposed_max/total_particles*100)
            ax2.fill_between(age_arriving_ds, f_decomposed_min, f_decomposed_max, color=colors[i], alpha=0.5)
            print(f'Final percentage past shelf accounting for decomposition: minimum {f_decomposed_min[-1]}, mean {f_decomposed[-1]}, maximum {f_decomposed_max[-1]}')
    
    if output_path_csv_d is not None:
        df.to_csv(output_path_csv_d, index=False)
     
    ax2.set_xlabel('Particle age (days)')
    ax2.set_ylabel('Particles past depth range\naccounting for decomposition (%)')
    ax2.set_ylim([0, 50])
    ax2.set_xlim(xlim)
    ax2.grid(True, linestyle='--', alpha=0.5)
    add_subtitle(ax2, '(b) Decomposed particle export')
    
    l2 = ax2.legend(title='Depth (m)', loc='upper left', bbox_to_anchor=(1.01, 1.01))
    
    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

        plt.close()

def figure5(particles:Particles, h_deep_sea=200,
            show=True, output_path=None):
    
    fig = plt.figure(figsize=(8, 5))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # (a) histogram decomposed particles passing shelf
    t_release = particles.get_release_time_index()
    p_ds, t_ds = particles.get_indices_arriving_in_deep_sea(h_deep_sea)
    
    times_release = particles.time[t_release]
    times_ds = particles.time[t_ds]
    time_bins = []
    for n in range(particles.time[-1].month-particles.time[0].month+2):
        time_bins.append(add_month_to_time(particles.time[0], n))
    n_releases, _ = np.histogram(times_release, bins=time_bins)
    
    total_particles = particles.lon.shape[0]
    n_releases_norm = n_releases/total_particles*100
    
    center_bins = np.array(time_bins[:-1]+np.diff(np.array(time_bins))/2)
    tick_labels = [center_bin.strftime("%b") for center_bin in center_bins]
    width = 0.8*np.array([dt.days for dt in np.diff(np.array(time_bins))])
    
    dt_ds = np.array([(particles.time[t_ds[i]]-particles.time[t_release[p_ds[i]]]).total_seconds()/(24*60*60) for i in range(len(p_ds))])
    f_ds = np.exp(k*dt_ds)
    times_ds_int, _ = convert_datetime_to_time(times_ds)
    time_bins_int, _ = convert_datetime_to_time(time_bins)
    i_bins = np.digitize(times_ds_int, bins=time_bins_int)
    
    f_ds_month = np.array([np.sum(f_ds[i_bins==i]) for i in range(1, 8)])
    f_ds_month_norm = f_ds_month/total_particles*100
    
    ax1 = plt.subplot(1, 2, 1)
    ax1.bar(center_bins, f_ds_month_norm, tick_label=tick_labels, width=width, color=kelp_green)
    ax1.set_ylabel('Particles passing shelf edge\naccounting for decomposition (%)')
    ax1.set_ylim([0, 10.8])
    ax1.spines['left'].set_color(kelp_green)
    ax1.tick_params(axis='y', colors=kelp_green)
    ax1.yaxis.label.set_color(kelp_green)
    add_subtitle(ax1, '(a) Decomposed particle export')
    
    ax2 = ax1.twinx()
    ax2.plot(center_bins, n_releases_norm, 'xk', label='Particles released')
    ax2.set_ylabel('Particles released (%)')
    ax2.set_ylim([0, 27])
    
    # season texts
    ax1.text(0.5/8, -0.1, 'Bunuru', ha='center', color=season_color, transform=ax1.transAxes)
    ax1.text(2.5/8, -0.1, 'Djeran', ha='center', color=season_color, transform=ax1.transAxes)
    ax1.text(4.5/8, -0.1, 'Makuru', ha='center', color=season_color, transform=ax1.transAxes)
    ax1.text(6.8/8, -0.1, 'Djilba', ha='center', color=season_color, transform=ax1.transAxes)
    
    # (b) histogram dswc occurrence
    start_date = datetime(2017, 3, 1)
    end_date = datetime(2017, 9, 30)
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
    str_month_dswc = np.array([t.strftime('%b') for t in month_dswc])
    p_dswc = np.array(p_dswc)
        
    ax4 = plt.subplot(1, 2, 2)
    ax4.bar(month_dswc, p_dswc*100, color=ocean_blue, tick_label=str_month_dswc, width=width)
    ax4.set_ylabel('Occurrence of suitable conditions for\ndense shelf water transport\n(% of time)')
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    ax4.set_ylim([0, 100])
    ax4.spines['left'].set_color(ocean_blue)
    ax4.tick_params(axis='y', colors=ocean_blue)
    ax4.yaxis.label.set_color(ocean_blue)
    add_subtitle(ax4, '(b) Dense shelf water transport')
    
    # season texts
    ax4.text(0.5/8, -0.1, 'Bunuru', ha='center', color=season_color, transform=ax4.transAxes)
    ax4.text(2.5/8, -0.1, 'Djeran', ha='center', color=season_color, transform=ax4.transAxes)
    ax4.text(4.5/8, -0.1, 'Makuru', ha='center', color=season_color, transform=ax4.transAxes)
    ax4.text(6.8/8, -0.1, 'Djilba', ha='center', color=season_color, transform=ax4.transAxes)

    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

        plt.close()

def figure6(particles:Particles, h_deep_sea=200, filter_kelp_prob=0.7,
            dx=0.01, vmin=0, vmax=0.2, scale_z=10, cmap='plasma',
            dx_c=0.02, lon_c=115.43,
            int_t=8,
            show=True, output_path=None):

    lon_examples = np.array([115.50, 115.31, 115.65, 115.64, 115.30, 115.56])
    lat_examples = np.array([-31.90, -31.73, -31.78, -32.43, -32.38, -32.17])

    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(hspace=0.1, wspace=0.4)

    location_info = get_location_info('perth')

    # (a) contribution to offshore export from initial locations
    t_release = particles.get_release_time_index()
    p_ds, t_ds = particles.get_indices_arriving_in_deep_sea(h_deep_sea)
    
    dt_ds = np.array([(particles.time[t_ds[i]]-particles.time[t_release[p_ds[i]]]).total_seconds()/(24*60*60) for i in range(len(p_ds))])
    
    # fraction of decomposed particles first arriving in deep sea
    f_ds = np.exp(k*dt_ds)
    lon0_ds = particles.lon0[p_ds]
    lat0_ds = particles.lat0[p_ds]
    
    grid = DensityGrid(location_info.lon_range, location_info.lat_range, dx)
    density_ds0 = get_particle_density(grid, lon0_ds, lat0_ds, values=f_ds)
    density_ds0_norm = density_ds0/np.sum(f_ds)*100
    
    x, y = np.meshgrid(grid.lon, grid.lat)
    z = np.copy(density_ds0_norm)*scale_z
    z[z==0.] = np.nan

    ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info)
    ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info,
                        ax=ax1, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    
    c1 = ax1.pcolormesh(x, y, z, vmin=vmin, vmax=vmax*scale_z, cmap=cmap)
    l1, b1, w1, h1 = ax1.get_position().bounds
    cbax1 = fig.add_axes([l1+w1+0.01, b1, 0.02, h1])
    cbar1 = plt.colorbar(c1, cax=cbax1)
    if scale_z == 10:
        cbar_label = 'Origin of particles passing shelf break (per mille)'
    elif scale_z == 1:
        cbar_label = 'Origin of particles passing shelf break (%)'
    else:
        cbar_label = f'Origin of particles passing shelf break (% x {scale_z})'
    cbar1.set_label(cbar_label)
    add_subtitle(ax1, '(a) Contribution to export')
    ax1.plot(lon_examples, lat_examples, 'ow', linewidth=5)
    ax1.plot(lon_examples, lat_examples, 'xk')

    # (b) map of mean cross-shelf transport in Perth region
    csv_ucross = 'temp_data/perth_wide_monthly_mean_u_cross_100m.csv'
    if not os.path.exists(csv_ucross):
        raise ValueError(f'''Mean cross-shelf velocity file does not yet exist: {csv_ucross}
                         Please create it first by running save_bottom_cross_shelf_velocities''')
    df = pd.read_csv(csv_ucross)
    time_cross = [datetime.strptime(d, '%Y-%m-%d') for d in df.columns.values]
    i_makuru = np.where([np.logical_or(t.month==6, t.month==7) for t in time_cross])[0]
    ucross = np.array([df.iloc[:, i].values for i in i_makuru])
    ucross = np.nanmean(ucross, axis=0)

    csv_dist = 'temp_data/perth_wide_distance_100m.csv'
    if not os.path.exists(csv_dist):
        raise ValueError(f'''Distance along depth contour file does not yet exist: {csv_dist}
                         Please create it first by running save_distance_along_depth_contour''')
    df_d = pd.read_csv(csv_dist)
    distance = df_d['distance'].values
    lats = df_d['lats'].values

    lat_bins = np.arange(location_info.lat_range[0], location_info.lat_range[-1]+dx_c, dx_c)
    ucross_bins = []
    for i in range(len(lat_bins)-1):
        l_lat = np.logical_and(lats >= lat_bins[i], lats < lat_bins[i+1])
        ucross_bins.append(np.sum(ucross[l_lat]*distance[l_lat])/np.sum(distance[l_lat]))
    ucross_bins = np.array(ucross_bins)
    lat_coords = lat_bins[:-1]+np.diff(lat_bins)
    lon_coords = np.ones(len(lat_coords))*lon_c

    ax2 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax2 = plot_basic_map(ax2, location_info, zorder_c=1, ymarkers='right')
    q = ax2.quiver(lon_coords, lat_coords, -ucross_bins, np.zeros(len(ucross_bins)))
    ax2.quiverkey(q, 0.88, 0.02, 0.1, label='0.1 m/s', labelpos='W')
    # ax2.set_yticklabels([])
    add_subtitle(ax2, '(b) Makuru cross-shelf transport')

    l2, b2, w2, h2 = ax2.get_position().bounds
    ax2.set_position([l2+0.02, b2, w2, h2])

    # (c) example particle tracks from different reefs (particle with median time to cross shelf)
    i_ex, j_ex = grid.get_index(lon_examples, lat_examples)
    lon0 = particles.lon0[p_ds]
    lat0 = particles.lat0[p_ds]
    p_ex = []
    for i in range(len(lon_examples)):
        l_lon = np.logical_and(lon0 >= grid.lon[i_ex[i].astype(int)], lon0 <= grid.lon[i_ex[i].astype(int)+1])
        l_lat = np.logical_and(lat0 >= grid.lat[j_ex[i].astype(int)], lat0 <= grid.lat[j_ex[i].astype(int)+1])
        ps_ex = np.where(np.logical_and(l_lon, l_lat))[0]
        i_sort = np.argsort(dt_ds[ps_ex]) # sort time ascending
        i_med = i_sort[np.floor(len(ps_ex)/2).astype(int)] # take middle (or halfway when even) value as median
        p_ex.append(ps_ex[i_med])
    
    location_info_w = get_location_info('perth_wide_south')
    ax4 = plt.subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax4 = plot_basic_map(ax4, location_info_w, ymarkers='right')
    ax4 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_w,
                        ax=ax4, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    
    lon = particles.lon[p_ds, :]
    lat = particles.lat[p_ds, :]
    # cm = mpl.colormaps['summer']
    colors = ['#2e2d4d', '#5d8888', '#c88066', '#4f5478', '#ebc08b', '#c15251']
    for i in range(len(p_ex)):
        # color = cm(i/(len(p_ex)-1))
        color = colors[i]
        ax4.plot(lon[p_ex[i], :t_ds[p_ex[i]]], lat[p_ex[i], :t_ds[p_ex[i]]], '-', color=color)
        ax4.plot(lon[p_ex[i], :t_ds[p_ex[i]]:int_t], lat[p_ex[i], :t_ds[p_ex[i]]:int_t], '.', color=color)
        ax4.plot(lon0[p_ex[i]], lat0[p_ex[i]], 'xk')
        ax4.plot(lon[p_ex[i], t_ds[p_ex[i]]], lat[p_ex[i], t_ds[p_ex[i]]], 'ok')
        
    add_subtitle(ax4, '(c) Example particle trajectories')
    
    l4, b4, w4, h4 = ax4.get_position().bounds
    ax4.set_position([l4, b2, h2/h4*w4, h2])

    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

        plt.close()

if __name__ == '__main__':
    if not os.path.exists('temp_data/perth_wide_monthly_mean_u_cross_100m.csv'):# or not os.path.exists('temp_data/perth_wide_monthly_mean_v_along_100m.csv'):
        save_bottom_cross_shelf_velocities()
    if not os.path.exists('temp_data/perth_wide_distance_100m.csv'):
        save_distance_along_depth_contour()

    plot_dir = get_dir_from_json("plots")

    # figure1(output_path=f'{plot_dir}fig1.jpg', show=False)
    
    # figure2(output_path=f'{plot_dir}fig2.jpg', show=False)

    particle_path = f'{get_dir_from_json("opendrift_output")}cwa_perth_MarSep2017_baseline.nc'
    particles = Particles.read_from_netcdf(particle_path)
    
    # figure3(particles, output_path=f'{plot_dir}fig3.jpg', show=False)
    
    # figure4(particles, output_path=f'{plot_dir}fig4.jpg', show=False)
    # # percentages past shelf:
    # # 59% particles, 19-33% accounting for decomposition (25% mean)
    
    figure5(particles, output_path=f'{plot_dir}fig5.jpg', show=False)
    
    # figure6(particles, output_path='fig6.jpg', show=False)
