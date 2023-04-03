from data.roms_data import RomsData, RomsGrid, read_roms_data_from_multiple_netcdfs, read_roms_grid_from_netcdf, get_subgrid
from data.roms_data import get_eta_xi_along_transect, get_distance_along_transect
from data.roms_data import TransectData, get_transect_data
from particles import Particles
from plot_tools.plot_cycler import plot_cycler
from plot_tools.basic_maps import plot_basic_map
from plot_tools.plots_bathymetry import plot_contours
from tools import log
from tools.timeseries import get_closest_time_index
from tools.files import get_dir_from_json
from tools.coordinates import get_transect_lons_lats_ds_from_json
from location_info import LocationInfo, get_location_info
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import numpy as np
from datetime import datetime
from dataclasses import dataclass
import json
import warnings

roms_grid = read_roms_grid_from_netcdf('input/cwa_roms_grid.nc')
transect_file = 'input/transects_dswc_detection.json' # transects in this file need to be ordered correctly!

def get_transects_from_json(transect_file:str) -> list:
    with open(transect_file, 'r') as f:
        all_transects = json.load(f)
    return all_transects

def plot_transects_on_map(roms_grid:RomsGrid, location_info:LocationInfo,
                          transect_file:str, show=True):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plot_basic_map(ax, location_info)
    ax = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info, ax=ax, show=False, show_perth_canyon=False, color='#757575')
    c = ax.pcolormesh(roms_grid.lon, roms_grid.lat, roms_grid.h, cmap='viridis', vmin=0, vmax=400)
    cbar = plt.colorbar(c)
    cbar.set_label('Bathymetry (m)')

    transects = get_transects_from_json(transect_file)
    for transect in transects:
        eta, xi = get_eta_xi_along_transect(roms_grid, transect['lon1'], transect['lat1'],
                                            transect['lon2'], transect['lat2'], transect['ds'])
        lon = roms_grid.lon[eta, xi]
        lat = roms_grid.lat[eta, xi]
        ax.plot(lon, lat, '-', color='k')

    if show is True:
        plt.show()
    else:
        return ax

def plot_cycling_roms_transect(transect_data:TransectData, parameter:str, l_dswc=None,
                               t_interval=1, vmin=24.8, vmax=25.6, cmap='RdYlBu_r'):
    
    if hasattr(transect_data, parameter):
        values = getattr(transect_data, parameter)
    else:
        raise ValueError(f'Unknown parameter requested.')

    def single_plot(fig, req_time):
        t = get_closest_time_index(roms_data.time, req_time)
        title = transect_data.time[t].strftime('%d-%m-%Y %H:%M')

        if len(values.shape) == 3:
            v = values[t, :, :]
        else:
            raise ValueError(f'Requested parameter does not have a depth component.')

        ax = plt.axes()
        c = ax.pcolormesh(transect_data.distance, transect_data.z, v, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        ax.fill_between(transect_data.distance[0, :], -transect_data.h, np.nanmin(transect_data.z), edgecolor='k', facecolor='#989898') # ROMS bottom

        if l_dswc is not None:
            ax.plot(transect_data.distance[0, :], l_dswc[t, :]*transect_data.h-transect_data.h, '-k', linewidth=5)
        
        ax.quiver(transect_data.distance, transect_data.z, transect_data.u_down[t, :],
                  np.zeros(transect_data.u_down[0, :].shape), scale=5, color='k')

        ax.set_xlabel('Distance along transect (km)')
        ax.set_xlim([0, np.nanmax(transect_data.distance)])
        ax.set_ylabel('Depth (m)')
        ax.set_ylim([np.nanmin(transect_data.z), 0])
        
        cbar = plt.colorbar(c)
        cbar.set_label(parameter)

        ax.set_title(title)
    
    t = np.arange(0, len(transect_data.time), t_interval)
    time = transect_data.time[t]

    fig = plot_cycler(single_plot, time)
    plt.show()  

def plot_cycling_roms_map(roms_data:RomsData, parameter:str, s:int,
                          location_info:LocationInfo, t_interval=1,
                          vmin=24.8, vmax=25.6, cmap='RdYlBu_r',
                          dswc_polygons=None, l_dswc=None):
    
    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
        if len(values.shape) == 4:
            values = values[:, s, :, :]
    
    def single_plot(fig, req_time):
        t = get_closest_time_index(roms_data.time, req_time)
        z = values[t, :, :]
        title = roms_data.time[t].strftime('%d-%m-%Y %H:%M')
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)
        ax.set_title(title)
        c = ax.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, z,
                          cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        cbar = plt.colorbar(c)
        cbar.set_label(parameter)

        if l_dswc is not None:
            cmap_dswc = ListedColormap([(1, 1, 1, 0.5), (1, 1, 1, 0)])
            ax.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, l_dswc[t, :, :], cmap=cmap_dswc)

        n_thin = 2
        u = roms_data.u_east[t, s, ::n_thin, ::n_thin]
        v = roms_data.v_north[t, s, ::n_thin, ::n_thin]
        lon = roms_data.grid.lon[::n_thin, ::n_thin]
        lat = roms_data.grid.lat[::n_thin, ::n_thin]
        ax.quiver(lon, lat, u, v, scale=3)

        if dswc_polygons is not None:
            p_collections = PatchCollection(dswc_polygons[t], alpha=0.5)
            ax.add_collection(p_collections)

    t = np.arange(0, len(roms_data.time), t_interval)
    time = roms_data.time[t]

    fig = plot_cycler(single_plot, time)
    plt.show()

def detect_dswc_in_transect(transect_data:RomsData) -> np.ndarray[bool]:
    '''Detect dense shelf water outflows in a transect based on:
    1. maximum depth to look for them
    2. cooler bottom water than surface water
    3. cooler water along the coast at the surface than offshore
    4. faster flowing offshore water in bottom layer than in mid-layer'''

    # parameters to play with:
    max_depth = 50 # m
    temp_diff_surface_bottom = 0. # positive for colder water at bottom
    temp_diff_offshore_coast = 0.1 # positive for colder water onshore
    minimum_bottom_flow = 0.05 # m/s mean flow in bottom 5 layers (needs to be 0 as minimum)

    # maximum depth for which to search for dswc
    l_h = transect_data.h <= max_depth

    # shape of temp and other variables: [time, depth, distance along transect]
    # bottom is at depth=0, surface is at depth=-1
    # coast is at distance=0, offshore is at distance=-1 (assuming transects are defined in this direction!)
    
    # bottom temperature is lower than surface temperature
    l_bottom = transect_data.temp[:, -1, :]-transect_data.temp[:, 0, :] > temp_diff_surface_bottom
    # surface temperature at coast is lower than offshore
    l_coast = transect_data.temp[:, -1, -1]-transect_data.temp[:, -1, 0] > temp_diff_offshore_coast
    l_coast = np.repeat(l_coast[:, np.newaxis], l_bottom.shape[1], axis=1)

    # mean flow in bottom layers > mean flow in mid layers
    mean_flow_bl = np.nanmean(transect_data.u_down[:, :5, :], axis=1)
    # l_flow = mean_flow_bl > 1.1*np.nanmean(transect_data.u_down[:, 10:15, :], axis=1)
    # and mean flow in bottom layer needs to be positive
    # l_flow = np.logical_and(l_flow, mean_flow_bl > minimum_bottom_flow)
    l_flow = mean_flow_bl > minimum_bottom_flow

    l_temp = np.logical_and(l_coast, l_bottom)
    l_dswc = np.logical_and(np.logical_and(l_temp, l_flow), l_h)

    return l_dswc

@dataclass
class DswcTransect:
    index: int
    time: datetime
    lon_coast: float
    lat_coast: float
    lon_offshore: float
    lat_offshore: float

def find_dswc_in_transects(roms_data:RomsData, transects:list[dict]) -> list[DswcTransect]:
    dswc_transects = []
    for i, transect in enumerate(transects):
        log.info(f'Detecting dense water outflows in transect {i+1}/{len(transects)}')
        transect_data = get_transect_data(roms_data, transect['lon1'], transect['lat1'],
                                          transect['lon2'], transect['lat2'], transect['ds'])
        l_dswc = detect_dswc_in_transect(transect_data)
        for t, time in enumerate(transect_data.time):
            s = np.where(l_dswc[t, :])[0]
            if s.any():
                lon_coast = transect_data.lon[s[0]]
                lat_coast = transect_data.lat[s[0]]
                lon_offshore = transect_data.lon[s[-1]]
                lat_offshore = transect_data.lat[s[-1]]
                dswc_transect = DswcTransect(i, time, lon_coast, lat_coast, lon_offshore, lat_offshore)
                dswc_transects.append(dswc_transect)

    return np.array(dswc_transects)

def group_dswc_transects(times:np.ndarray,
                         dswc_transects:list[DswcTransect]) -> dict[datetime, list[DswcTransect]]:
    dswc_transect_groups = {}
    
    for time in times:
        l_time = [tt==time for tt in [dswc_transect.time for dswc_transect in dswc_transects]]
        dswc_transects_t = dswc_transects[l_time]
        transect_indices = [dswc_transect.index for dswc_transect in dswc_transects_t]
        i_order = np.argsort(transect_indices)
        dswc_transects_t = dswc_transects_t[i_order]
        dswc_transect_groups_t = []
        new_group = True
        i0 = dswc_transects_t[0].index
        for dswc_transect in dswc_transects_t:
            if dswc_transect.index == i0:
                # first time in loop: start new group
                new_group = True
            elif dswc_transect.index == i0+1:
                # transect follows previous one: append to current group
                new_group = False
                group.append(dswc_transect)
            elif dswc_transect.index > i0+1:
                # transect does not follow previous one:
                # start new group and append old group to grouped_dswc_transects_t list
                new_group = True
                dswc_transect_groups_t.append(group)
            else:
                raise ValueError(f"You shouldn't end up here.")
            if new_group is True:
                group = []
                group.append(dswc_transect)
                i0 = dswc_transect.index

        dswc_transect_groups[time] = dswc_transect_groups_t

    return dswc_transect_groups

def create_dswc_polygons(dswc_transect_groups:dict[datetime, list[DswcTransect]]) -> dict[datetime, list[Polygon]]:
    polygon_groups = {}

    for time in dswc_transect_groups.keys():
        polygon_groups_t = []

        for dswc_group in dswc_transect_groups[time]:
            lons = [dswc.lon_coast for dswc in dswc_group]
            lats = [dswc.lat_coast for dswc in dswc_group]
            lons.extend([dswc.lon_offshore for dswc in dswc_group[::-1]])
            lats.extend([dswc.lat_offshore for dswc in dswc_group[::-1]])

            polygon_groups_t.append(Polygon(np.array([lons, lats]).transpose(), closed=True))

        polygon_groups[time] = polygon_groups_t

    return polygon_groups

def get_grid_locations_with_dswc(roms_data:RomsData, dswc_polygons:dict[datetime, list[Polygon]]) -> np.ndarray:
    lons = roms_data.grid.lon.flatten()
    lats = roms_data.grid.lat.flatten()

    l_dswc = np.zeros((len(roms_data.time), roms_data.grid.lon.shape[0], roms_data.grid.lon.shape[1]))

    for t, time in enumerate(roms_data.time):
        if time not in dswc_polygons.keys():
            warnings.warn(f'Did not search for dense water outflows on {time}, skipping.')
            continue
        for p in dswc_polygons[time]:
            l_dswc_1d = p.contains_points(np.array([lons, lats]).transpose())
            l_dswc[t, :, :] = np.reshape(l_dswc_1d, roms_data.grid.lon.shape)

    return l_dswc

if __name__ == '__main__':
    start_date = datetime(2017, 5, 2)
    end_date = datetime(2017, 5, 2)
    roms_dir = f'{get_dir_from_json("roms_data")}cwa/2017/'
    location_info = get_location_info('perth')
    
    roms_data = read_roms_data_from_multiple_netcdfs(roms_dir, start_date, end_date)
    all_transects = get_transects_from_json(transect_file)

    # plot_transects_on_map(roms_grid, location_info, transect_file)
    
    # --- manual checks ---
    transect = all_transects[2]
    transect_data = get_transect_data(roms_data, transect['lon1'], transect['lat1'],
                                      transect['lon2'], transect['lat2'], transect['ds'])
    l_dswc = detect_dswc_in_transect(transect_data)
    plot_cycling_roms_transect(transect_data, 'temp', vmin=20, vmax=22, l_dswc=l_dswc)
    # ---

    dswc_transects = find_dswc_in_transects(roms_data, all_transects)
    dswc_transect_groups = group_dswc_transects(roms_data.time, dswc_transects)
    polygon_groups = create_dswc_polygons(dswc_transect_groups)
    l_dswc = get_grid_locations_with_dswc(roms_data, polygon_groups)

    plot_cycling_roms_map(roms_data, 'temp', 0, location_info, vmin=20., vmax=22.,
                          l_dswc=l_dswc)
    

