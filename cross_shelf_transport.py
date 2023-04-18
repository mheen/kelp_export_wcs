from data.roms_data import RomsData, read_roms_data_from_multiple_netcdfs, get_cross_shelf_velocity_component
from data.roms_data import RomsGrid, read_roms_grid_from_netcdf
from data.roms_data import get_distance_along_transect
from tools.timeseries import get_closest_time_index, add_month_to_time
from tools.files import get_dir_from_json
from tools import log
from plot_tools.plots_bathymetry import plot_contours
from location_info import LocationInfo, get_location_info
from plot_tools.basic_maps import plot_basic_map
from plot_tools.plot_cycler import plot_cycler
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from datetime import datetime
import pickle

roms_grid = read_roms_grid_from_netcdf('input/cwa_roms_grid.nc')
plots_dir = f'{get_dir_from_json("plots")}'

def plot_cycling_cross_shelf_vectors_with_temperature(roms_data:RomsData,
                                                      location_info:LocationInfo,
                                                      s=0, t_interval=1):

    u_cross = get_cross_shelf_velocity_component(roms_data)

    def single_plot(fig, req_time):
        t = get_closest_time_index(roms_data.time, req_time)
        
        title = roms_data.time[t].strftime('%d-%m-%Y %H:%M')

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)
        ax.set_title(title)
        c = ax.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, roms_data.temp[t, s, :, :],
                          cmap='RdBu_r', vmin=20, vmax=22, shading='nearest')
        cbar = plt.colorbar(c)
        cbar.set_label(f'Temperature ($^o$), s={s}')

        n_thin = 2
        u = -u_cross[t, s, ::n_thin, ::n_thin]
        v = np.zeros(u.shape)
        lon = roms_data.grid.lon[::n_thin, ::n_thin]
        lat = roms_data.grid.lat[::n_thin, ::n_thin]
        ax.quiver(lon, lat, u, v, scale=2)

    t = np.arange(0, len(roms_data.time), t_interval)
    time = roms_data.time[t]

    fig = plot_cycler(single_plot, time)
    plt.show()

def plot_cycling_cross_shelf_transport_with_velocity_vectors(roms_data:RomsData,
                                                             location_info:LocationInfo,
                                                             s=0, t_interval=1):

    u_cross = get_cross_shelf_velocity_component(roms_data)

    def single_plot(fig, req_time):
        t = get_closest_time_index(roms_data.time, req_time)
        
        title = roms_data.time[t].strftime('%d-%m-%Y %H:%M')

        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)
        ax.set_title(title)
        c = ax.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, u_cross[t, s, :, :],
                          cmap='RdBu_r', vmin=-0.2, vmax=0.2, shading='nearest')
        cbar = plt.colorbar(c)
        cbar.set_label(f'Cross-shelf velocity (m/s), s={s}')

        n_thin = 2
        u = roms_data.u_east[t, s, ::n_thin, ::n_thin]
        v = roms_data.v_north[t, s, ::n_thin, ::n_thin]
        lon = roms_data.grid.lon[::n_thin, ::n_thin]
        lat = roms_data.grid.lat[::n_thin, ::n_thin]
        ax.quiver(lon, lat, u, v, scale=2)

    t = np.arange(0, len(roms_data.time), t_interval)
    time = roms_data.time[t]

    fig = plot_cycler(single_plot, time)
    plt.show()

def plot_cross_shelf_transport_along_transect_in_map(lats_50m:np.ndarray,
                                                     lats_100m:np.ndarray,
                                                     lats_200m:np.ndarray,
                                                     distance_50m:np.ndarray,
                                                     distance_100m:np.ndarray,
                                                     distance_200m:np.ndarray,
                                                     u_cross_50m:np.ndarray,
                                                     u_cross_100m:np.ndarray,
                                                     u_cross_200m:np.ndarray,
                                                     location_info:LocationInfo,
                                                     ax=None,
                                                     show=True,
                                                     output_path=None):
    
    l_lat50 = np.logical_and(lats_50m>=location_info.lat_range[0], lats_50m<=location_info.lat_range[1])
    l_lat100 = np.logical_and(lats_100m>=location_info.lat_range[0], lats_100m<=location_info.lat_range[1])
    l_lat200 = np.logical_and(lats_200m>=location_info.lat_range[0], lats_200m<=location_info.lat_range[1])

    if ax is None:
        fig = plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
    
    ax = plot_basic_map(ax, location_info)
    ax = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info, ax=ax, show=False,
                       highlight_contour=None, show_perth_canyon=False)

    l, b, w, h = ax.get_position().bounds
    w_t = w/3
    xlim = [-0.3, 0.3]

    ax2 = fig.add_axes([l, b, w_t, h])
    ax2.plot(-u_cross_200m[l_lat200], distance_200m[l_lat200], ':', linewidth=2, color='#322992', label='200')
    ax2.plot([0, 0], [distance_200m[l_lat200][0], distance_200m[l_lat200][-1]], '-', color='k', alpha=0.5)
    ax2.set_xlim(xlim)
    ax2.set_ylim([distance_200m[l_lat200][0], distance_200m[l_lat200][-1]])
    ax2.axis('off')

    ax3 = fig.add_axes([l+w_t/3, b, w_t, h])
    ax3.plot(-u_cross_100m[l_lat100], distance_100m[l_lat100], '--', linewidth=2, color='#322992', label='100')
    ax3.plot([0, 0], [distance_100m[l_lat100][0], distance_100m[l_lat100][-1]], '-', color='k', alpha=0.5)
    ax3.set_xlim(xlim)
    ax3.set_ylim([distance_100m[l_lat100][0], distance_100m[l_lat100][-1]])
    ax3.axis('off')

    ax4 = fig.add_axes([l+2*w_t/3, b, w_t, h])
    ax4.plot(-u_cross_50m[l_lat50], distance_50m[l_lat50], '-', linewidth=2, color='#322992', label='50')
    ax4.plot([0, 0], [distance_50m[l_lat50][0], distance_50m[l_lat50][-1]], '-', color='k', alpha=0.5)
    ax4.set_xlim(xlim)
    ax4.set_ylim([distance_50m[l_lat50][0], distance_50m[l_lat50][-1]])
    ax4.axis('off')

    fig.legend(loc='lower right', title='Depth levels (m)', bbox_to_anchor=(l+w, b))

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def get_coordinates_of_depth_contour(roms_grid:RomsGrid, h_level=50) -> tuple[np.ndarray, np.ndarray]:
    levels = [0, 10, 20, 50, 100, 200, 400, 600, 1000]

    try:
        i_level = levels.index(h_level)
    except:
        raise ValueError(f'Levels does not contain {h_level}. Please request a valid value: {levels}')

    ax = plt.axes()
    contour_set = ax.contour(roms_grid.lon, roms_grid.lat, roms_grid.h, levels=levels)

    contour_line = contour_set.collections[i_level].get_paths()[0]
    contour_line_coords = contour_line.vertices
    lons = contour_line_coords[:, 0]
    lats = contour_line_coords[:, 1]

    plt.close()

    return lons, lats

def get_time_depth_mean_cross_shelf_transport_at_bottom(u_cross:np.ndarray, h_level:float) -> tuple:
    
    lons, lats = get_coordinates_of_depth_contour(roms_grid, h_level=h_level)
    eta, xi = roms_grid.get_eta_xi_of_lon_lat_point(lons, lats)
    distance = get_distance_along_transect(lons, lats)
    mean_bottom_u_cross = np.nanmean(np.nanmean(u_cross[:, 0:3, eta, xi], axis=0), axis=0)

    return lons, lats, distance, mean_bottom_u_cross

def save_mean_cross_shelf_per_month(input_dir:str, start_date:datetime, end_date:datetime):
    h_levels = [50, 100, 200]
    distances = {}
    lats = {}
    u_cross_means = {}
    
    n_months = end_date.month-start_date.month
    for n in range(n_months+1):
        start = add_month_to_time(start_date, n)
        end = add_month_to_time(start_date, n+1)
        month = start.month
        roms_data = read_roms_data_from_multiple_netcdfs(input_dir, start, end)
        u_cross = get_cross_shelf_velocity_component(roms_data)

        distances[month] = {}
        lats[month] = {}
        u_cross_means[month] = {}
        for h_level in h_levels:
            _, lat, distance, u_cross_level = get_time_depth_mean_cross_shelf_transport_at_bottom(u_cross, h_level)
            distances[month][h_level] = distance
            lats[month][h_level] = lat
            u_cross_means[month][h_level] = u_cross_level

        output_path = f'{plots_dir}mean_cross_shelf_bottom_transport_{datetime(2017, month, 1).strftime("%b%Y")}.jpg'
        plot_cross_shelf_transport_along_transect_in_map(lats[month][50], lats[month][100], lats[month][200],
                                                         distances[month][50], distances[month][100], distances[month][200],
                                                         u_cross_means[month][50], u_cross_means[month][100], u_cross_means[month][200],
                                                         get_location_info('perth'), show=False, output_path=output_path)

    with open('temp_u_cross_shelf_monthly_means.pickle', 'wb') as f:
        pickle.dump((lats, distances, u_cross_means), f)

    return lats, distances, u_cross_means

if __name__ == '__main__':
    start_date = datetime(2017, 8, 15)
    end_date = datetime(2017, 8, 15)
    roms_dir = f'{get_dir_from_json("roms_data")}cwa/2017/'
    location_info = get_location_info('perth_wide')

    roms_data = read_roms_data_from_multiple_netcdfs(roms_dir, start_date, end_date)
    u_cross = get_cross_shelf_velocity_component(roms_data)

    # mean cross-shelf transport along a depth contour
    _, lats_50m, distance_50m, u_cross_50m = get_time_depth_mean_cross_shelf_transport_at_bottom(u_cross, 50)
    _, lats_100m, distance_100m, u_cross_100m = get_time_depth_mean_cross_shelf_transport_at_bottom(u_cross, 100)
    _, lats_200m, distance_200m, u_cross_200m = get_time_depth_mean_cross_shelf_transport_at_bottom(u_cross, 200)
    
    # lats, distances, u_cross_means = save_mean_cross_shelf_per_month(roms_dir, start_date, end_date)

    output_path = f'{plots_dir}cross_shelf_transport_{start_date.strftime("%d%b%Y")}.jpg'
    plot_cross_shelf_transport_along_transect_in_map(lats_50m, lats_100m, lats_200m,
                                                     distance_50m, distance_100m, distance_200m,
                                                     u_cross_50m, u_cross_100m, u_cross_200m,
                                                     location_info, show=False, output_path=output_path)


