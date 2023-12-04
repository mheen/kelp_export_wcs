from data.roms_data import RomsData, RomsGrid, read_roms_data_from_multiple_netcdfs, read_roms_grid_from_netcdf
from data.wind_data import WindData, read_era5_wind_data_from_netcdf
from plot_tools.plot_cycler import plot_cycler
from plot_tools.basic_maps import plot_basic_map
from plot_tools.plots_bathymetry import plot_contours
from tools import log
from tools.timeseries import get_closest_time_index, get_l_time_range
from tools.files import get_dir_from_json
from location_info import LocationInfo, get_location_info
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.dates as mdates
import matplotlib.units as munits
import cartopy.crs as ccrs
import numpy as np
from datetime import datetime, date, timedelta
import json
import warnings
import pandas as pd
from scipy.interpolate import interp1d

# bottom is at depth=0, surface is at depth=-1
depth_limit = 50 # m

def exclude_roms_data_past_depth(roms_data:RomsData, depth_limit=depth_limit) -> RomsData:
    l_too_deep = roms_data.grid.h > depth_limit
    roms_data.temp[:, :, l_too_deep] = np.nan
    roms_data.density[:, :, l_too_deep] = np.nan
    roms_data.salt[:, :, l_too_deep] = np.nan
    roms_data.sigma_t[:, :, l_too_deep] = np.nan
    roms_data.u_east[:, :, l_too_deep] = np.nan
    roms_data.v_north[:, :, l_too_deep] = np.nan
    roms_data.grid.h[l_too_deep] = np.nan
    
    return roms_data

def calculate_mean_density(roms_data:RomsData) -> np.ndarray:
    delta_z = np.diff(roms_data.grid.z_w, axis=0)
    mean_density = np.sum(roms_data.density*delta_z, axis=1)/roms_data.grid.h
    return mean_density

def calculate_potential_energy_anomaly(roms_data:RomsData) -> np.ndarray:
    mean_density = calculate_mean_density(roms_data)
    g = 9.81 # m/s2
    delta_z = np.diff(roms_data.grid.z_w, axis=0)
    phi = g/roms_data.grid.h*np.sum((np.repeat(mean_density[:, np.newaxis, :, :], roms_data.density.shape[1], axis=1)-roms_data.density)*roms_data.grid.z*delta_z, axis=1)
    return phi

def calculate_potential_energy_anomaly_per_latitude(roms_data:RomsData) -> np.ndarray:
    phi = calculate_potential_energy_anomaly(roms_data)
    mean_phi = np.nanmean(phi, axis=2)
    return mean_phi

def calculate_density_gradient_per_latitude(roms_data:RomsData) -> np.ndarray:
    depth_mean_density = calculate_mean_density(roms_data)
    drho = np.diff(depth_mean_density, axis=2)
    dx = -1/roms_data.grid.pm[:, :-1] # m - negative sign because moving away from the coast in westwards direction (doesn't apply for other coastlines!)
    drho_dlon = drho/dx
    mean_density_gradient = np.nanmean(drho_dlon, axis=2)
    return mean_density_gradient

# --- numbers that indicate conditions for dswc ---
def calculate_horizontal_richardson_number(roms_data:RomsData, wind_data:WindData) -> float:
    '''Calculates the horizontal Richardson number (or Simpson number) as specified by
    Mahjabin et al. (2020, Scientific Reports).
    Note: Tanziha gets Richardson numbers on the order of 1.'''
    
    g = 9.81 # gravitational acceleration (m/s^2)
    rho_w = 1025 # sea water density (kg/m^3): using a standard value for this, but could also use a mean value from roms
    rho_a = 1.2 # air density (kg/m^3)
    mean_h = np.nanmean(roms_data.grid.h)
    drhodx = calculate_density_gradient_per_latitude(roms_data)
    mean_drhodx = np.nanmean(drhodx)
        
    mean_w = np.nanmean(wind_data.vel)
    ks = 0.03*(0.63+0.066*mean_w**(1/3))/1000 # surface drag coefficient according to Pugh (1987)
    u = np.sqrt(ks*rho_a/rho_w)*mean_w

    Ri = g/rho_w*mean_h**2/u**2*mean_drhodx # still factor 100 too large and negative (although that seems logical if drhodx is negative)
    
    return Ri

def write_horizontal_richardson_number_to_csv(start_date:datetime, end_date:datetime,
                                              roms_dir:str, era5_dir:str, location_info:LocationInfo,
                                              output_path:str):
    
    n_days = (end_date-start_date).days+1
    time = []
    Ri = []
    for n in range(n_days):
        load_day = start_date+timedelta(days=n)
        roms_data = read_roms_data_from_multiple_netcdfs(roms_dir, load_day, load_day,
                                                         lon_range=location_info.lon_range,
                                                         lat_range=location_info.lat_range)
        roms_data = exclude_roms_data_past_depth(roms_data)
        wind_data = read_era5_wind_data_from_netcdf(era5_dir, load_day, load_day,
                                                    lon_range=location_info.lon_range,
                                                    lat_range=location_info.lat_range)
        
        Ri.append(calculate_horizontal_richardson_number(roms_data, wind_data))
        time.append(load_day)
        
    time = np.array(time).flatten()
    Ri = np.array(Ri).flatten()
    df = pd.DataFrame(np.array([time, Ri]).transpose(), columns=['time', 'Ri'])
    log.info(f'Writing horizontal Richardson number timeseries to: {output_path}')
    df.to_csv(output_path, index=False)

def calculate_gravitational_versus_wind_components(roms_data:RomsData,
                                             wind_data:WindData) -> tuple[float, float]:
    '''Based on Hetzel et al. (2013, Continental Shelf Research). Also used in Mahjabin et al. (2020, SR).
    Note: Yasha gets wind and gravitational components on the order of 10**-7 [J m-3 s-1].'''
    g = 9.81
    rho_w = 1025
    rho_a = 1.2
    delta = 3*10**-3
    K_mz = 1*10**-2
    mean_h = np.nanmean(roms_data.grid.h)

    drhodx = calculate_density_gradient_per_latitude(roms_data)
    mean_drhodx = np.nanmean(drhodx)
    grav_component = 1/320*g**2*mean_h**4/(rho_w*K_mz)*mean_drhodx**2
    
    mean_w = np.nanmean(wind_data.vel)
    ks = 0.03*(0.63+0.066*mean_w**(1/3))/1000 # surface drag coefficient according to Pugh (1987)
    wind_component = delta*ks*rho_a*mean_w**3/mean_h
    
    return grav_component, wind_component

def write_gravitation_wind_components_to_csv(start_date:datetime, end_date:datetime,
                                             roms_dir:str, era5_dir:str, location_info:LocationInfo,
                                             output_path:str):
    
    n_days = (end_date-start_date).days+1
    time = []
    grav_c = []
    wind_c = []
    drhodx = []
    phi = []
    for n in range(n_days):
        load_day = start_date+timedelta(days=n)
        roms_data = read_roms_data_from_multiple_netcdfs(roms_dir, load_day, load_day,
                                                         lon_range=location_info.lon_range,
                                                         lat_range=location_info.lat_range)
        roms_data = exclude_roms_data_past_depth(roms_data)
        wind_data = read_era5_wind_data_from_netcdf(era5_dir, load_day, load_day,
                                                    lon_range=location_info.lon_range,
                                                    lat_range=location_info.lat_range)
        
        time.append(load_day)
        g, w = calculate_gravitational_versus_wind_components(roms_data, wind_data)
        grav_c.append(g)
        wind_c.append(w)
        drhodx_all = calculate_density_gradient_per_latitude(roms_data)
        drhodx.append(np.nanmean(drhodx_all))
        phi_all = calculate_potential_energy_anomaly(roms_data)
        phi.append(np.nanmean(phi_all))
        
    time = np.array(time).flatten()
    grav_c = np.array(grav_c).flatten()
    wind_c = np.array(wind_c).flatten()
    drhodx = np.array(drhodx).flatten()
    phi = np.array(phi).flatten()
    df = pd.DataFrame(np.array([time, grav_c, wind_c, drhodx, phi]).transpose(), columns=['time', 'grav_component', 'wind_component', 'drhodx', 'phi'])
    log.info(f'Writing gravitational and wind component timeseries to: {output_path}')
    df.to_csv(output_path, index=False)

# --- detection ---
def find_dswc_per_latitude(roms_data:RomsData) -> np.ndarray[bool]:
    
    temp_diff_surface_bottom = 0.5 # positive for colder water at bottom
    
    dpdx = calculate_density_gradient_per_latitude(roms_data)
    phi = calculate_potential_energy_anomaly_per_latitude(roms_data)
    
    l_prereq = np.logical_and(dpdx < 0, phi > 1)
    
    l_temp_s_b = roms_data.temp[:, -1, :, :] - roms_data.temp[:, 0, :, :] > temp_diff_surface_bottom
    
    l_dswc = np.zeros((len(roms_data.time), roms_data.grid.lat.shape[0], roms_data.grid.lon.shape[1])).astype(bool)
    for t in range(len(roms_data.time)):
        for i in range(roms_data.grid.lon.shape[1]):
            if l_prereq[t, i] == True:
                l_dswc[t, i, :] = l_temp_s_b[t, i, :]
    
    return l_dswc

def plot_cycling_transect(roms_data:RomsData, parameter='temp', l_dswc=None,
                          i_lat=25,
                          t_interval=1, vmin=20., vmax=22., cmap='RdYlBu_r'):
    
    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
        values = values[:, :, i_lat, :]
    else:
        raise ValueError(f'Unknown parameter requested.')

    lon = roms_data.grid.lon[i_lat, :]
    z = roms_data.grid.z[:, i_lat, :]
    h = roms_data.grid.h[i_lat, :]

    def single_plot(fig, req_time):
        t = get_closest_time_index(roms_data.time, req_time)
        title = roms_data.time[t].strftime('%d-%m-%Y %H:%M')

        if len(values.shape) == 3:
            v = values[t, :, :]
        else:
            raise ValueError(f'Requested parameter does not have a depth component.')

        ax = plt.axes()
        c = ax.pcolormesh(lon, z, v, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        ax.fill_between(lon, -h, np.nanmin(z), edgecolor='k', facecolor='#989898') # ROMS bottom

        if l_dswc is not None:
            ax.plot(lon, l_dswc[t, i_lat, :]*h-h, '-k', linewidth=5)
        
        ax.set_ylabel('Depth (m)')
        ax.set_ylim([np.nanmin(z), 0])
        
        cbar = plt.colorbar(c)
        cbar.set_label(parameter)

        ax.set_title(title)
    
    t = np.arange(0, len(roms_data.time), t_interval)
    time = roms_data.time[t]

    fig = plot_cycler(single_plot, time)
    plt.show()

def plot_cycling_map(roms_data:RomsData, l_dswc=None,
                     parameter='temp', t_interval=1,
                     location_info=get_location_info('perth'),
                     vmin=20., vmax=22., cmap='RdYlBu_r'):
    # for parameter='sigma_t': vmin=24.8, vmax=25.6
    # for parameter='salt': vmin=35.5, vmax=35.9 (in summer at least)
    
    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
        if len(values.shape) == 4:
            values_b = values[:, 0, :, :]
            values_s = values[:, -1, :, :]
    
    def single_plot(fig, req_time):
        t = get_closest_time_index(roms_data.time, req_time)
        
        z_b = values_b[t, :, :]
        z_s = values_s[t, :, :]
        
        time_str = roms_data.time[t].strftime('%d-%m-%Y %H:%M')
        
        n_thin = 2
        lon = roms_data.grid.lon[::n_thin, ::n_thin]
        lat = roms_data.grid.lat[::n_thin, ::n_thin]
        
        # --- Bottom ---
        # colormap parameter
        title = f'Bottom: {time_str}'
        ax = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)
        ax.set_title(title)
        c = ax.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, z_b,
                          cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')

        # velocities
        u_b = roms_data.u_east[t, 0, ::n_thin, ::n_thin]
        v_b = roms_data.v_north[t, 0, ::n_thin, ::n_thin]
        ax.quiver(lon, lat, u_b, v_b, scale=3)
        
        # dswc
        if l_dswc is not None:
            cmap_dswc = ListedColormap([(1, 1, 1, 0.5), (1, 1, 1, 0)])
            ax.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, l_dswc[t, :, :], cmap=cmap_dswc)
        
        # --- Surface ---
        # colormap parameter
        title2 = f'Surface: {time_str}'
        ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
        ax2 = plot_basic_map(ax2, location_info)
        ax2.set_title(title2)
        c2 = ax2.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, z_s,
                            cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest')
        
        # colorbar
        l, b, w, h = ax2.get_position().bounds
        cbax = fig.add_axes([l+w+0.01, b, 0.02, h])
        cbar = plt.colorbar(c2, cax=cbax)
        cbar.set_label(parameter)
        
        # velocities
        u_s = roms_data.u_east[t, -1, ::n_thin, ::n_thin]
        v_s = roms_data.v_north[t, -1, ::n_thin, ::n_thin]
        ax2.quiver(lon, lat, u_s, v_s, scale=3)
        
        # dswc
        if l_dswc is not None:
            cmap_dswc = ListedColormap([(1, 1, 1, 0.5), (1, 1, 1, 0)])
            ax2.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, l_dswc[t, :, :], cmap=cmap_dswc)

    t = np.arange(0, len(roms_data.time), t_interval)
    time = roms_data.time[t]

    fig = plot_cycler(single_plot, time)
    plt.show()

def write_fraction_cells_dswc_in_time_to_csv(start_date:datetime, end_date:datetime,
                                             roms_dir:str, location_info:LocationInfo,
                                             output_path:str,):
    
    n_days = (end_date-start_date).days
    time = []
    f_dswc = []
    for n in range(n_days+1):
        load_day = start_date+timedelta(days=n)
        roms_data = read_roms_data_from_multiple_netcdfs(roms_dir, load_day, load_day,
                                                         lon_range=location_info.lon_range,
                                                         lat_range=location_info.lat_range)
        roms_data = exclude_roms_data_past_depth(roms_data)
        l_dswc = find_dswc_per_latitude(roms_data)
        
        time.append(roms_data.time)
        f_dswc.append(np.sum(np.sum(l_dswc, axis=1), axis=1)/np.sum(roms_data.grid.h <= depth_limit))
        
    time = np.array(time).flatten()
    f_dswc = np.array(f_dswc).flatten()
    df = pd.DataFrame(np.array([time, f_dswc]).transpose(), columns=['time', 'f_dswc'])
    log.info(f'Writing fraction of time dswc to: {output_path}')
    df.to_csv(output_path, index=False)
    
if __name__ == '__main__':
    start_date = datetime(2017, 1, 1)
    end_date = datetime(2017, 12, 31)
    
    write_fraction_dswc = True
    write_gravitational_wind_components = True
    
    roms_dir = f'{get_dir_from_json("roms_data")}2017/'
    era5_dir = f'{get_dir_from_json("era5_data")}'
    location_info = get_location_info('perth')
    
    if write_fraction_dswc == True:
        output_path_fdswc = 'temp_data/fraction_cells_dswc_in_time.csv'
        write_fraction_cells_dswc_in_time_to_csv(start_date, end_date, roms_dir, location_info, output_path_fdswc)
    
    if write_gravitational_wind_components == True:
        output_path_gw = 'temp_data/gravitational_wind_components_in_time.csv'
        write_gravitation_wind_components_to_csv(start_date, end_date, roms_dir, era5_dir, location_info, output_path_gw)

    # # --- test and manual checks ---
    # roms_data = read_roms_data_from_multiple_netcdfs(roms_dir, start_date, end_date, lon_range=location_info.lon_range, lat_range=location_info.lat_range)
    # roms_data = exclude_roms_data_past_depth(roms_data)
    
    # l_dswc = find_dswc_per_latitude(roms_data)
    
    # # plot_cycling_transect(roms_data, l_dswc=l_dswc)
    
    # plot_cycling_map(roms_data, l_dswc=l_dswc)