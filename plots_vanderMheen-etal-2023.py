from tools import log
from tools.files import get_dir_from_json
from tools.timeseries import add_month_to_time
from data.kelp_data import KelpProbability
from data.roms_data import read_roms_grid_from_netcdf, read_roms_data_from_multiple_netcdfs, get_subgrid
from data.roms_data import get_cross_shelf_velocity_component, get_eta_xi_along_depth_contour
from data.roms_data import get_lon_lat_along_depth_contour, get_distance_along_transect
from particles import Particles, get_particle_density, DensityGrid
from plot_tools.basic_maps import plot_basic_map
from plot_tools.general import add_subtitle
from plot_tools.plots_bathymetry import plot_contours
from location_info import LocationInfo, get_location_info
from datetime import datetime, date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import cartopy.crs as ccrs
import numpy as np
import pandas as pd
import cmocean
import os

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)

kelp_green = '#1b7931'
ocean_blue = '#25419e'

roms_grid = read_roms_grid_from_netcdf('input/cwa_roms_grid.nc')

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
    
    df = pd.DataFrame(distance, columns=['distance'])
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

def figure6(particles:Particles, h_deep_sea=200, filter_kelp_prob=None, dx=0.01,
            vmin=0, vmax=0.05, cmap='plasma',
            show=True, output_path=None):

    fig = plt.figure(figsize=(9, 11))
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    location_info = get_location_info('perth')

    # (a) map of original location as % making it past shelf
    l_deep_sea = particles.get_l_deep_sea(h_deep_sea)
    l_deep_sea_any_time = np.any(l_deep_sea, axis=1).astype('bool')
    n_deep_sea = np.sum(l_deep_sea_any_time)

    lon0_ds = particles.lon0[l_deep_sea_any_time]
    lat0_ds = particles.lat0[l_deep_sea_any_time]

    if filter_kelp_prob is not None:
        kelp_prob = KelpProbability.read_from_tiff('input/perth_kelp_probability.tif')
        kelp_prob_ds = kelp_prob.get_kelp_probability_at_point(lon0_ds, lat0_ds)
        l_kelp_ds = kelp_prob_ds>=filter_kelp_prob
        lon0_ds = lon0_ds[l_kelp_ds]
        lat0_ds = lat0_ds[l_kelp_ds]

    grid = DensityGrid(location_info.lon_range, location_info.lat_range, dx)
    density_ds = get_particle_density(grid, lon0_ds, lat0_ds)
    density_dsn = density_ds/n_deep_sea*100
    
    x, y = np.meshgrid(grid.lon, grid.lat)
    density_dsn[density_dsn==0.] = np.nan

    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info)
    ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info,
                        ax=ax1, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)
    
    c1 = ax1.pcolormesh(x, y, density_dsn, vmin=vmin, vmax=vmax, cmap=cmap)
    l1, b1, w1, h1 = ax1.get_position().bounds
    cbax1 = fig.add_axes([l1+w1+0.01, b1, 0.02, h1])
    cbar1 = plt.colorbar(c1, cax=cbax1)
    cbar1.set_label(f'(%/{dx}x{dx} grid cells)')
    add_subtitle(ax1, '(a) Origin of particles\nmaking it past shelf')

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
    # !!! NOT GOOD ENOUGH: NEED TO GET COORDINATES AND BIN TO REGULAR INTERVALS !!!
    distance = df_d['distance'].values

    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2 = plot_basic_map(ax2, location_info)
    # ax2 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info,
    #                     ax=ax2, show=False, show_perth_canyon=False,
    #                     color='k', linewidths=0.7)
    ax2.set_yticklabels([])

    l2, b2, w2, h2 = ax2.get_position().bounds
    w_t = w2/3
    xlim = [-0.3, 0.3]

    ax3 = fig.add_axes([l2, b2, w_t, h2])
    # ax3.plot(-ucross, distance, ':', linewidth=2, color='#322992', label='200')
    ax3.barh(distance, -ucross, color=ocean_blue) # preferably quiver instead of bar
    ax3.plot([0, 0], [distance[0], distance[-1]], '-', color='k', alpha=0.5)
    ax3.set_xlim(xlim)
    ax3.set_ylim([distance[0], distance[-1]])
    ax3.axis('off')

    # (c) example particle tracks from different reefs
    location_info_w = get_location_info('perth_wide')
    ax4 = plt.subplot(2, 2, (3, 4), projection=ccrs.PlateCarree())
    ax4 = plot_basic_map(ax4, location_info_w)
    ax4 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_w,
                        ax=ax4, show=False, show_perth_canyon=False,
                        color='k', linewidths=0.7)

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
    figure6(particles, output_path='fig6.jpg', show=False)

    
