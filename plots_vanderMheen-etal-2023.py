from tools import log
from tools.files import get_dir_from_json
from tools.timeseries import add_month_to_time
from tools.coordinates import get_index_closest_point
from data.kelp_data import KelpProbability
from data.roms_data import read_roms_grid_from_netcdf, read_roms_data_from_multiple_netcdfs, get_subgrid
from data.roms_data import get_cross_shelf_velocity_component, get_eta_xi_along_depth_contour
from data.roms_data import get_lon_lat_along_depth_contour, get_distance_along_transect
from particles import Particles, get_particle_density, DensityGrid
from plot_tools.basic_maps import plot_basic_map
from plot_tools.general import add_subtitle
from plot_tools.plots_bathymetry import plot_contours
from plot_tools.plots_particles import plot_age_in_deep_sea, plot_particle_age_in_deep_sea_depending_on_depth
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
    for n in range(n_months):
        start_date = add_month_to_time(date0, n)
        end_date = add_month_to_time(date0, n+1)-timedelta(days=1)
        roms = read_roms_data_from_multiple_netcdfs(roms_dir, start_date, end_date,
                                                    lon_range=location_info.lon_range,
                                                    lat_range=location_info.lat_range)
        u_cross = get_cross_shelf_velocity_component(roms)
        eta, xi = get_eta_xi_along_depth_contour(roms.grid, h_level=h_level)
        # monthly means
        time_monthly.append(start_date)
        u_cross_monthly_spatial.append(np.nanmean(np.nanmean(u_cross[:, 0:max_s_layer, eta, xi], axis=0), axis=0))

    df_s = pd.DataFrame(np.array(u_cross_monthly_spatial).transpose(), columns=time_monthly)
    df_s.to_csv(f'temp_data/{location_name}_monthly_mean_u_cross_{h_level}m.csv', index=False)

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
    plt.subplots_adjust(wspace=0.35)

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
    add_subtitle(ax1, '(a) Perth kelp reefs')

    # (c) bathymetry and oceanography overview
    location_info_pw = get_location_info('perth_wide')
    ax3 = plt.subplot(3, 3, (2, 6), projection=ccrs.PlateCarree())
    ax3 = plot_basic_map(ax3, location_info_pw, ymarkers='off')
    ax3 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_pw, ax=ax3, show=False, show_perth_canyon=True, color='k', linewidths=0.7)
    c3 = ax3.pcolormesh(roms_grid.lon, roms_grid.lat, roms_grid.h, vmin=0, vmax=4000, cmap=cmocean.cm.deep)
    l3, b3, w3, h3 = ax3.get_position().bounds
    cbax3 = fig.add_axes([l3+w3+0.01, b3, 0.02, h3])
    cbar3 = plt.colorbar(c3, cax=cbax3)
    cbar3.set_label('Bathymetry (m)')
    add_subtitle(ax3, '(c) Main oceanographic features')
    # ax3.set_position([l3+0.02, b3+0.05, w3, h3])

    # (b) detritus production from de Bettignies et al. (2013)
    time_detritus = ['Mar-May', 'Jun-Jul', 'Sep-Oct', 'Dec-Feb']
    detritus = [4.8, 2.12, 0.70, 0.97]
    detritus_sd = [1.69, 0.84, 0.45, 0.81]

    ax2 = plt.subplot(3, 3, 7)
    ax2.bar(np.arange(len(time_detritus)), detritus, color=kelp_green, tick_label=time_detritus, yerr=detritus_sd)
    ax2.set_ylim([0, 7.5])
    ax2.set_ylabel('Detritus production\n(g/kelp/day)')
    add_subtitle(ax2, '(b) Seasonal detritus production')
    l2, b2, w2, h2 = ax2.get_position().bounds
    ax2.set_position([l1, b2, w1+0.03, h2])

    # (c) cross-shore transport histogram
    csv_ucross = 'temp_data/perth_wide_monthly_mean_u_cross_100m.csv'
    if not os.path.exists(csv_ucross):
        raise ValueError(f'''Mean cross-shelf velocity file does not yet exist: {csv_ucross}
                         Please create is first by running save_bottom_cross_shelf_velocities''')
    df = pd.read_csv(csv_ucross)
    time_cross = [datetime.strptime(d, '%Y-%m-%d') for d in df.columns.values]
    str_time_cross = [d.strftime('%b') for d in time_cross]
    ucross = [np.nanmean(df.iloc[:, i].values) for i in range(len(df.columns))]

    ax4 = plt.subplot(3, 3, (8, 9))
    ax4.bar(np.arange(len(time_cross)), ucross, color=ocean_blue, tick_label=str_time_cross)
    xlim = ax4.get_xlim()
    ax4.plot([-1, 12], [0, 0], '-k')
    ax4.set_xlim(xlim)
    ax4.set_ylim([-0.05, 0.05])
    ax4.set_ylabel('Offshore transport (m/s)')
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    add_subtitle(ax4, '(d) Monthly mean offshore transport')
    l4, b4, w4, h4 = ax4.get_position().bounds
    ax4.set_position([l3, b4, w3, h4])

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
    
    fig = plt.figure(figsize=(6, 8))
    
    xlim = [0, 120]
    
    # (a) age in deep sea for different depths
    ax1 = plt.subplot(2, 1, 1)
    ax1, l1 = plot_particle_age_in_deep_sea_depending_on_depth(particles, h_deep_sea_sensitivity=h_deep_seas,
                                                               linestyles=linestyles,
                                                               colors=colors,
                                                               ax=ax1, show=False)
    l1.remove()
    ax1.set_ylabel('Particles past depth range (%)')
    ax1.set_xlim(xlim)
    add_subtitle(ax1, '(a) Particle export')
    
    # (b) age in deep sea with composition
    ax2 = plt.subplot(2, 1, 2)
    
    for i, h_deep_sea in enumerate(h_deep_seas):
        _, age_arriving_ds, matrix_arriving_ds = particles.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
        n_deep_sea_per_age = np.sum(matrix_arriving_ds, axis=0)
        total_particles = particles.lon.shape[0]
        f_deep_sea_per_age = n_deep_sea_per_age/total_particles*100 # divided by total # particles
        f_cumulative_per_age = np.cumsum(f_deep_sea_per_age)
        f_decomposed = f_cumulative_per_age*np.exp(k*age_arriving_ds)    

        ax2.plot(age_arriving_ds, f_decomposed, color=colors[i], linestyle=linestyles[i], label=h_deep_sea)
        
        if h_deep_sea == 200:
            f_decomposed_min = f_cumulative_per_age*np.exp((k-k_sd)*age_arriving_ds)
            f_decomposed_max = f_cumulative_per_age*np.exp((k+k_sd)*age_arriving_ds)
            ax2.fill_between(age_arriving_ds, f_decomposed_min, f_decomposed_max, color=colors[i], alpha=0.5)
        
    ax2.set_xlabel('Particle age (days)')
    ax2.set_ylabel('Particles past depth range\naccounting for decomposition (%)')
    ax2.set_ylim([0, 30])
    ax2.set_xlim(xlim)
    ax2.grid(True, linestyle='--', alpha=0.5)
    add_subtitle(ax2, '(b) Decomposed particle export')
    
    l2 = ax2.legend(title='Depth (m)', loc='upper right', bbox_to_anchor=(0.99, 1.01))
    
    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

        plt.close()

def figure6(particles:Particles, h_deep_sea=200, filter_kelp_prob=0.7,
            dx=0.02, vmin=0, vmax=0.2, cmap='plasma',
            dx_c=0.02, lon_c=115.43,
            int_t=8,
            show=True, output_path=None):

    lon_examples = [115.52, 115.59, 115.32, 115.32, 115.55, 115.63]
    lat_examples = [-32.25, -32.6, -32.4, -31.8, -31.83, -31.72]

    fig = plt.figure(figsize=(12, 6))
    plt.subplots_adjust(hspace=0.1, wspace=0.2)

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
    z = np.copy(density_ds0_norm)
    z[z==0.] = np.nan

    ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info)
    ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info,
                        ax=ax1, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    
    c1 = ax1.pcolormesh(x, y, z, vmin=vmin, vmax=vmax, cmap=cmap)
    l1, b1, w1, h1 = ax1.get_position().bounds
    cbax1 = fig.add_axes([l1+w1+0.01, b1, 0.02, h1])
    cbar1 = plt.colorbar(c1, cax=cbax1)
    cbar1.set_label(f'Origin of particles passing shelf break (%)')
    add_subtitle(ax1, '(a) Contribution to export')
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
    ax2 = plot_basic_map(ax2, location_info, zorder_c=1)
    q = ax2.quiver(lon_coords, lat_coords, -ucross_bins, np.zeros(len(ucross_bins)))
    ax2.quiverkey(q, 0.88, 0.02, 0.1, label='0.1 m/s', labelpos='W')
    ax2.set_yticklabels([])
    add_subtitle(ax2, '(b) Makuru cross-shelf transport')

    l2, b2, w2, h2 = ax2.get_position().bounds

    # (c) example particle tracks from different reefs
    p_ex = []
    lon0 = particles.lon0[p_ds]
    lat0 = particles.lat0[p_ds]
    for i in range(len(lon_examples)):
        p_ex.append(get_index_closest_point(lon0, lat0, lon_examples[i], lat_examples[i])[0])
    
    location_info_w = get_location_info('perth_wide_south')
    ax4 = plt.subplot(1, 3, 3, projection=ccrs.PlateCarree())
    ax4 = plot_basic_map(ax4, location_info_w)
    ax4 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_w,
                        ax=ax4, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    
    lon = particles.lon[p_ds, :]
    lat = particles.lat[p_ds, :]
    cm = mpl.colormaps['summer']
    for i in range(len(p_ex)):
        color = cm(i/(len(p_ex)-1))
        ax4.plot(lon[p_ex[i], :t_ds[p_ex[i]]], lat[p_ex[i], :t_ds[p_ex[i]]], '-', color=color)
        ax4.plot(lon[p_ex[i], :t_ds[p_ex[i]]:int_t], lat[p_ex[i], :t_ds[p_ex[i]]:int_t], '.', color=color)
        ax4.plot(lon0[p_ex[i]], lat0[p_ex[i]], 'xk')
        ax4.plot(lon[p_ex[i], t_ds[p_ex[i]]], lat[p_ex[i], t_ds[p_ex[i]]], 'ok')
        
    add_subtitle(ax4, '(c) Example particle trajectories')

    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

        plt.close()

if __name__ == '__main__':
    if not os.path.exists('temp_data/perth_wide_monthly_mean_u_cross_100m.csv'):
        save_bottom_cross_shelf_velocities()
    if not os.path.exists('temp_data/perth_wide_distance_100m.csv'):
        save_distance_along_depth_contour()

    # figure1(output_path='fig1.jpg')

    particle_path = f'{get_dir_from_json("opendrift_output")}cwa_perth_MarAug2017_baseline.nc'
    particles = Particles.read_from_netcdf(particle_path)
    
    # figure4(particles, output_path='fig4.jpg', show=False)
    
    figure6(particles, output_path='fig6.jpg', show=False)
