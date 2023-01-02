from tools.files import get_dir_from_json
from tools.timeseries import get_closest_time_index, get_l_time_range
from tools.coordinates import get_bearing_between_points
from tools import log
from roms_data import RomsGrid, RomsData, read_roms_data_from_multiple_netcdfs
from roms_data import get_distance_along_transect, get_eta_xi_along_transect, get_gradient_along_transect
from plots_bathymetry import plot_contours
from location_info import LocationInfo, get_location_info
from basic_maps import plot_basic_map
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from datetime import datetime, timedelta

def plot_roms_map(roms_data:RomsData, location_info:LocationInfo,
                  parameter:str, time:datetime, s=-1, # default: surface
                  ax=None, show=True, output_path=None,
                  cmap='RdBu_r', clabel='', vmin=None, vmax=None) -> plt.axes:
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    t = get_closest_time_index(roms_data.time, time)

    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
        if len(values.shape) == 4:
            values = values[t, s, :, :] # [time, s, eta, xi]
        elif len(values.shape) == 3:
            values = values[t, :, :] # [time, eta, xi] -> correct? or can also be [s, eta, xi]?
        elif len(values.shape) == 2:
            values = values[:, :] # [eta, xi]
    elif hasattr(roms_data.grid, parameter):
        values = getattr(roms_data.grid, parameter)
        if len(values.shape) == 3:
            values = values[s, :, :] # [s, eta, xi] (because grid data is non-time dependent)
        elif len(values.shape) == 2:
            values = values[:, :] # [eta, xi]
    elif parameter == 'velocity':
        u = roms_data.u_east[t, s, :, :]
        v = roms_data.v_north[t, s, :, :]
        values = np.sqrt(u**2+v**2)
    else:
        raise ValueError(f'Unknown parameter {parameter} in RomsData')

    c = ax.pcolormesh(roms_data.grid.lon, roms_data.grid.lat, values, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(c)
    cbar.set_label(clabel)

    ax.set_title(time.strftime('%d %b %Y'))

    if parameter == 'velocity':
        thin = 5
        i = np.arange(0, u.shape[0], thin)
        j = np.arange(0, u.shape[1], thin)
        u_q = u[i][:, j]
        v_q = v[i][:, j]
        lon_q = roms_data.grid.lon[i][:, j]
        lat_q = roms_data.grid.lat[i][:, j]
        ax.quiver(lon_q, lat_q, u_q, v_q, scale=10, transform=ccrs.PlateCarree())

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_roms_map_with_transect(roms_data:RomsData, location_info:LocationInfo,
                                lon1:float, lat1:float, lon2:float, lat2:float, ds:float,
                                parameter:str, time:datetime, s=-1, # default: surface
                                ax=None, show=True, output_path=None, color='k',
                                cmap='RdBu_r', clabel='', vmin=None, vmax=None) -> plt.axes:

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    eta, xi = get_eta_xi_along_transect(roms_data.grid, lon1, lat1, lon2, lat2, ds)
    lon = roms_data.grid.lon[eta, xi]
    lat = roms_data.grid.lat[eta, xi]

    ax = plot_roms_map(roms_data, location_info, parameter, time, s=s, ax=ax, show=False, cmap=cmap, clabel=clabel, vmin=vmin, vmax=vmax)
    ax.plot(lon, lat, '-', color=color, label='ROMS transect')

    ax.legend(loc='upper left')

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def get_down_transect_velocity_component(u:np.ndarray, v:np.ndarray,
                                         lon1:float, lat1:float, lon2:float, lat2:float) -> np.ndarray:
    alpha = get_bearing_between_points(lon1, lat1, lon2, lat2)
    alpha_rad = np.deg2rad(alpha)
    down_transect = u*np.cos(alpha_rad)+v*np.sin(alpha_rad)
    return down_transect

def plot_roms_transect(roms_data:RomsData,
                       lon1:float, lat1:float, lon2:float, lat2:float, ds:float,
                       parameter:str, time:datetime,
                       ax=None, show=True, output_path=None,
                       cmap='RdBu_r', clabel='', vmin=None, vmax=None) -> plt.axes:
    
    eta, xi = get_eta_xi_along_transect(roms_data.grid, lon1, lat1, lon2, lat2, ds)

    lon = roms_data.grid.lon[eta, xi]
    lat = roms_data.grid.lat[eta, xi]
    distance = get_distance_along_transect(lon, lat)/1000 # distance in km

    z = roms_data.grid.z[:, eta, xi]
    h = roms_data.grid.h[eta, xi]

    t = get_closest_time_index(roms_data.time, time)

    if hasattr(roms_data, parameter):
        values = getattr(roms_data, parameter)
        if len(values.shape) == 4:
            values = values[t, :, eta, xi] # [time, s, eta, xi]
        elif len(values.shape) == 3:
            values = values[t, eta, xi] # [time, eta, xi]
        elif len(values.shape) == 2:
            values = values[eta, xi] # [eta, xi]
    elif parameter == 'velocity':
        u = roms_data.u_east[t, :, eta, xi]
        v = roms_data.v_north[t, :, eta, xi]
        values = np.sqrt(u**2+v**2)
    else:
        raise ValueError(f'Unknown parameter {parameter} in RomsData to plot transect')

    if ax is None:
        fig = plt.figure(figsize=(8, 3))
        ax = plt.axes()
    distance2d = np.repeat(distance[np.newaxis, :], z.shape[0], axis=0)
    c = ax.pcolormesh(distance2d, z, values.transpose(), cmap=cmap, vmin=vmin, vmax=vmax)
    ax.fill_between(distance, -h, np.nanmin(z), edgecolor='k', facecolor='#989898') # ROMS bottom
    
    ax.set_xlabel('Distance along transect (km)')
    ax.set_xlim([0, np.nanmax(distance)])
    ax.set_ylabel('Depth (m)')
    ax.set_ylim([np.nanmin(z), 0])
    
    cbar = plt.colorbar(c)
    cbar.set_label(clabel)
    
    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def animate_roms_transect(roms_data:RomsData,
                          lon1:float, lat1:float, lon2:float, lat2:float, ds:float,
                          parameter:str, start_time:datetime, end_time:datetime,
                          show_quivers=True, dpi=200, fps=10, output_path=None,
                          cmap='RdBu_r', clabel='', vmin=None, vmax=None):

    writer = animation.PillowWriter(fps=fps)

    # intialise plot and load non-changing data
    plt.rcParams.update({'font.size' : 15})
    plt.rcParams.update({'font.family': 'arial'})
    plt.rcParams.update({'figure.dpi': dpi})
    fig = plt.figure(figsize=(8, 3))
    ax = plt.axes()
    
    eta, xi = get_eta_xi_along_transect(roms_data.grid, lon1, lat1, lon2, lat2, ds)

    lon = roms_data.grid.lon[eta, xi]
    lat = roms_data.grid.lat[eta, xi]
    z = roms_data.grid.z[:, eta, xi]
    h = roms_data.grid.h[eta, xi]
    distance = get_distance_along_transect(lon, lat)/1000 # distance in km
    distance2d = np.repeat(distance[np.newaxis, :], z.shape[0], axis=0)

    l_time = get_l_time_range(roms_data.time, start_time, end_time)
    time = roms_data.time[l_time]

    if parameter == 'velocity':
        u = roms_data.u_east[l_time, :, :, :][:, :, eta, xi]
        v = roms_data.v_north[l_time, :, :, :][:, :, eta, xi]
        values = np.sqrt(u**2+v**2)
    else:
        values = getattr(roms_data, parameter)[l_time, :, :, :][:, :, eta, xi]
    
    if show_quivers is True:
        # velocity in transect direction
        u = roms_data.u_east[l_time, :, :, :][:, :, eta, xi]
        v = roms_data.v_north[l_time, :, :, :][:, :, eta, xi]
        s_layer = 2
        index_shallow = h<=75
        thin_h = 5
        index_thin = (np.empty(index_shallow.shape)*0).astype('bool')
        index_thin[::thin_h] = True
        index_h = np.logical_and(index_shallow, index_thin)
        scale = 50
        n_multiply = 10
        vel = get_down_transect_velocity_component(u[:, s_layer, index_h], v[:, s_layer, index_h], lon1, lat1, lon2, lat2)*n_multiply

    # animated data
    transect = ax.pcolormesh(distance2d, z, values[0, :, :], cmap=cmap, vmin=vmin, vmax=vmax)
    if show_quivers is True:
        quiver = ax.quiver(distance2d[s_layer, index_h], z[s_layer, index_h], vel[0, :], np.zeros(vel[0, :].shape), scale=scale, color='k')

    # fixed data
    ax.fill_between(distance, -h, np.nanmin(z), edgecolor='k', facecolor='#989898') # ROMS bottom
    if show_quivers is True:
        ax.quiverkey(quiver, 0.2, 0.2, 0.5*n_multiply, 'Along transect velocity (0.5 m/s)', labelpos='E', coordinates='figure')
    ax.set_xlabel('Distance along transect (km)')
    ax.set_xlim([0, np.nanmax(distance)])
    ax.set_ylabel('Depth (m)')
    ax.set_ylim([np.nanmin(z), 0])
    
    cbar = plt.colorbar(transect)
    cbar.set_label(clabel)

    # animated text
    ttl = ax.text(0.5, 1.04,'', transform=ax.transAxes,
                  ha='center', va='bottom',
                  bbox=dict(facecolor='w', alpha=0.3, edgecolor='w', pad=2))
    ttl.set_animated(True)

    def animate(i):
        transect.set_array(values[i, :, :].ravel())
        title = time[i].strftime('%d %b %Y %H:%M')
        ttl.set_text(title)
        if show_quivers is True:
            quiver.set_UVC(vel[i, :], np.zeros(vel[i, :].shape))
            return transect, quiver, ttl
        return transect, ttl

    fig.tight_layout()

    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(time), blit=True)
    if output_path is not None:
        log.info(f'Saving animation to: {output_path}')
        anim.save(output_path, writer=writer)
    else:
        plt.show()

def plot_depth_gradient(roms_data:RomsData, location_info:LocationInfo,
                        lon1:float, lat1:float, lon2:float, lat2:float, ds=5000,
                        show=True, output_path=None,
                        cmap='RdBu_r', vmin=None, vmax=None):
    
    dhdx, h, distance = get_gradient_along_transect(roms_data, 'h', 0, roms_data.time[0], lon1, lat1, lon2, lat2, ds)

    fig = plt.figure(figsize=(10, 8))
    ax1 = plt.subplot(2, 5, (1, 3))
    ax1.plot(distance[1:], dhdx, '-k')
    ax1.set_xlim(distance[0], distance[-1])
    ax1.set_xticklabels([])
    ax1.set_xlim([0, np.nanmax(distance)])
    ax1.set_ylabel('Depth gradient along transect (m/km)')

    ax2 = plt.subplot(1, 5, (4, 10), projection=ccrs.PlateCarree())
    ax2 = plot_basic_map(ax2, location_info)
    ax2 = plot_roms_map_with_transect(roms_data, location_info, lon1, lat1, lon2, lat2, ds, 'h', roms_data.time[0],
                                      ax=ax2, show=False, clabel='Bathymetry (m)', cmap=cmap, vmin=vmin, vmax=vmax,
                                      color='#e4e4e4')
    ax2 = plot_contours(roms_data.grid.lon, roms_data.grid.lat, roms_data.grid.h, location_info,
                        ax=ax2, show=False)
    
    ax3 = plt.subplot(2, 5, (6, 8))
    ax3.fill_between(distance, -h, -np.nanmax(h), edgecolor='k', facecolor='#989898')
    ax3.set_xlim([0, np.nanmax(distance)])
    ax3.set_ylim([-np.nanmax(h), 0])
    ax3.set_xlabel('Distance along transect (km)')
    ax3.set_ylabel('')

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()

if __name__ == '__main__':
    location_info = get_location_info('perth')
    input_dir = f'{get_dir_from_json("roms_data")}2022/'
    start_date = datetime(2022, 7, 1)
    end_date = datetime(2022, 7, 10)
    roms = read_roms_data_from_multiple_netcdfs(input_dir, start_date, end_date)
    
    lon1 = 115.70
    lat1 = -31.76
    lon2 = 115.26
    lat2 = -31.95
    ds = 500

    mid_date = start_date+timedelta(days=(end_date-start_date).days/2)
    output_path_map = f'{get_dir_from_json("plots")}roms_bathymetry_with_transect.jpg'
    plot_roms_map_with_transect(roms, location_info, lon1, lat1, lon2, lat2, ds, 'temp', mid_date,
                                vmin=18, vmax=22, clabel='Temperature ($^o$C)', output_path=output_path_map, show=False)
    
    # output_path_animation = f'{get_dir_from_json("plots")}roms_dswc_temperature_animation_{start_date.strftime("%b-%Y")}.gif'
    # animate_roms_transect(roms, lon1, lat1, lon2, lat2, ds, 'temp', start_date, end_date+timedelta(days=1), output_path=output_path_animation,
    #                       vmin=18, vmax=22, clabel='Temperature ($^o$C)', show_quivers=False)


    # lon1 = 115.70
    # lat1 = -31.76
    # lon2 = 114.47
    # lat2 = -31.80

    # location_info = get_location_info('cwa_perth_zoom')
    # input_dir = f'{get_dir_from_json("roms_data")}2017/'
    # start_date = datetime(2017, 5, 1)
    # end_date = datetime(2017, 5, 2)
    # roms = read_roms_data_from_multiple_netcdfs(input_dir, start_date, end_date)
    # plot_depth_gradient(roms, location_info, lon1, lat1, lon2, lat2)
