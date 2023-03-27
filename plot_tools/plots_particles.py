import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.timeseries import get_closest_time_index, get_l_time_range
from tools.files import get_dir_from_json
from tools import log
from particles import Particles, DensityGrid, get_particle_density
from data.bathymetry_data import BathymetryData
from plot_tools.plots_bathymetry import plot_contours
from data.kelp_data import KelpProbability
from location_info import LocationInfo, get_location_info
from plot_tools.basic_maps import plot_basic_map
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import cartopy.crs as ccrs
from warnings import warn
import numpy as np
from datetime import date, datetime, timedelta

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)
formatter.formats = ['%y',  # ticks are mostly years
                    '%b',       # ticks are mostly months
                    '%d',       # ticks are mostly days
                    '%H:%M',    # hrs
                    '%H:%M',    # min
                    '%S.%f', ]  # secs
# these are mostly just the level above...
formatter.zero_formats = [''] + formatter.formats[:-1]
# ...except for ticks that are mostly hours, then it is nice to have
# month-day:
formatter.zero_formats[3] = '%d-%b'

formatter.offset_formats = ['',
                            '%Y',
                            '%b %Y',
                            '%d %b %Y',
                            '%d %b %Y',
                            '%d %b %Y %H:%M', ]

def animate_particles(particles:Particles, location_info:LocationInfo, h_deep_sea:float,
                      show_bathymetry=True, show_kelp_map=False,
                      output_path=None, color_p='k', color_ds='#1b7931',
                      dpi=100, fps=25):

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
                           highlight_contour=[h_deep_sea], ax=ax, show=False, color='#757575')
    
    if show_kelp_map is True:
        kelp_prob = KelpProbability.read_from_tiff('input/perth_kelp_probability.tif')
        ax = kelp_prob.plot(location_info, ax=ax, show=False)

    # animated points
    point = ax.plot([], [], 'o', color=color_p, markersize=2, zorder=2)[0]
    point_ds = ax.plot([], [], 'o', color=color_ds, markersize=2, zorder=3)[0]
    
    # legend
    legend_elements = [Line2D([0],[0], marker='o', color='w', markerfacecolor=color_p, markersize=10,
                       label='Coastal region'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor=color_ds, markersize=10,
                       label='Past shelf break')]
    ax.legend(handles=legend_elements, loc='upper left')

    # animated text
    ttl = ax.text(0.5, 1.04,'', transform=ax.transAxes,
                ha='center', va='top',
                bbox=dict(facecolor='w', alpha=0.3, edgecolor='w', pad=2))
    ttl.set_animated(True)
    
    def init():
        point.set_data([], [])
        point_ds.set_data([], [])
        ttl.set_text('')
        return point, point_ds, ttl

    def animate(i):
        x, y = (particles.lon[~l_deep_sea[:, i], i], particles.lat[~l_deep_sea[:, i], i])
        point.set_data(x, y)
        x_ds, y_ds = (particles.lon[l_deep_sea[:, i], i], particles.lat[l_deep_sea[:, i], i])
        point_ds.set_data(x_ds, y_ds)
        title = particles.time[i].strftime('%d %b %Y %H:%M')
        ttl.set_text(title)
        return point, point_ds, ttl

    anim = animation.FuncAnimation(plt.gcf(), animate, init_func=init, frames=len(particles.time), blit=True)
    if output_path is not None:
        log.info(f'Saving animation to: {output_path}')
        anim.save(output_path, writer=writer)
    else:
        plt.show()

def plot_matrix_arriving_in_deep_sea(particles:Particles, h_deep_sea:float,
                                     ax=None, show=True, cmap='plasma',
                                     output_path=None) -> plt.axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()
    
    time_release, age_arriving_ds, matrix_arriving_ds = particles.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
    n_dim_matrix = len(matrix_arriving_ds.shape)
    if n_dim_matrix == 1:
        warn(f'''There is only a single release available in this simulation;
                matrix_arriving_ds is {n_dim_matrix}D, so this plot cannot be made. Skipping.''')
        return ax

    n_particles_in_simulation = particles.get_n_particles_in_simulation()

    y, x = np.meshgrid(age_arriving_ds, time_release)
    z = matrix_arriving_ds/n_particles_in_simulation[0]
    z[z==0] = np.nan

    c = ax.pcolormesh(x, y, z, cmap=cmap)
    cbar = plt.colorbar(c)
    cbar.set_label('Fraction of released particles arriving in deep sea')

    def _create_ticklabels_at_interval(tick_labels:np.ndarray, n_int:int) -> list:
        tick_labels_int = []
        for i in range(len(tick_labels)):
            if i%n_int == 0: # (% is modulo)
                tick_labels_int.append(str(tick_labels[i]))
            else:
                tick_labels_int.append('')
        return tick_labels_int

    # myFmt = mdates.DateFormatter('%d-%m-%Y')
    # ax.set_xticks(time_release[::5])
    # ax.set_xticklabels(time_release[::5], rotation=90)
    # ax.xaxis.set_major_formatter(myFmt)
    ax.set_xlabel('Release date')
    ax.set_xlim([time_release[0], time_release[-1]])

    ax.set_yticks(age_arriving_ds)
    ax.set_yticklabels(_create_ticklabels_at_interval(age_arriving_ds.astype(int), 5))
    ax.set_ylabel('Particle age (days)')
    ax.set_ylim([0, np.nanmax(age_arriving_ds)])

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_histogram_arriving_in_deep_sea(particles:Particles, h_deep_sea:float,
                                        ax=None, show=True, output_path=None,
                                        color='#1b7931', edgecolor='none'):

    _, t_arriving_ds = particles.get_indices_arriving_in_deep_sea(h_deep_sea)
    
    n_days = (particles.time[-1]-particles.time[0]).days
    time_days = np.array([particles.time[0]+timedelta(days=n) for n in range(n_days)])

    n_arriving, _ = np.histogram(particles.time[t_arriving_ds], bins=time_days)
    
    if ax is None:
        fig = plt.figure(figsize=(10, 4))
        ax = plt.axes()

    ax.bar(time_days[:-1], n_arriving, color=color, edgecolor=edgecolor)

    ax.set_xlim([time_days[0], time_days[-1]])
    ax.set_ylabel(f'Particles passing\n {h_deep_sea} m (#)')

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_percentage_in_deep_sea_depending_on_depth(particles:Particles, h_deep_sea_sensitivity=np.arange(200, 6000, 100),
                                                   color_age='#1b7931', color_p='k',
                                                   ax=None, show=True, output_path=None) -> plt.axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()
        ax.set_xlabel('Depth (m)')
        ax.set_ylabel('Total particles\npassing depth range (%)')

    p_deep_sea = []
    age_80_percent = []
    for h_deep_sea in h_deep_sea_sensitivity:
        _, age_arriving_ds, matrix_arriving_ds = particles.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
        n_deep_sea_per_age = np.sum(matrix_arriving_ds, axis=0)
        total_particles = particles.lon.shape[0]
        f_deep_sea_per_age = n_deep_sea_per_age/total_particles*100 # divided by total # particles
        f_cumulative_per_age = np.cumsum(f_deep_sea_per_age)

        p_deep_sea.append(f_cumulative_per_age[-1])

        if f_cumulative_per_age[-1] != 0:
            i_80 = np.where(f_cumulative_per_age>=f_cumulative_per_age[-1]*0.8)[0][0]
            age_80_percent.append(age_arriving_ds[i_80])
        else:
            age_80_percent.append(np.nan)

    ax.plot(h_deep_sea_sensitivity, p_deep_sea, color=color_p)
    ax.set_xticks([200, 600, 1000, 2000, 3000, 4000, 5000, 6000])
    ax.set_xlim([h_deep_sea_sensitivity[0], h_deep_sea_sensitivity[-1]])
    ax.set_ylim([0, 100])
    ax.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax.twinx()
    ax2.set_ylabel('Age when >80% particles\npass depth range (days)', color=color_age)
    ax2.plot(h_deep_sea_sensitivity, age_80_percent, color=color_age)
    ax2.set_yticks([0, 30, 60, 90, 120, 150])
    ax2.tick_params(axis='y', colors=color_age)
    ax2.set_ylim([0, 150])
    
    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def _plot_age_in_deep_sea_cumulative_only(ax, particles:Particles, h_deep_sea:float,
                                         color='k', linestyle='-', label='') -> plt.axes:

    _, age_arriving_ds, matrix_arriving_ds = particles.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
    n_deep_sea_per_age = np.sum(matrix_arriving_ds, axis=0)
    total_particles = particles.lon.shape[0]
    f_deep_sea_per_age = n_deep_sea_per_age/total_particles*100 # divided by total # particles
    f_cumulative_per_age = np.cumsum(f_deep_sea_per_age)

    ax.plot(age_arriving_ds, f_cumulative_per_age, color=color, linestyle=linestyle, label=label)

    return ax

def plot_particle_age_in_deep_sea_depending_on_depth(particles:Particles,
                                                     h_deep_sea_sensitivity=[200, 400, 600, 800, 1000, 2500, 5000],
                                                     colors = ['#1b7931', '#1c642a', '#1a5023', '#183d1d', '#142a16', '#0e190e', '#000000'],
                                                     linestyles = ['--', ':', '-', '-.', '--', ':', '-'],
                                                     output_path=None, show=True) -> plt.axes:
    
    labels = h_deep_sea_sensitivity

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes()
    for i, h in enumerate(h_deep_sea_sensitivity):
        ax = _plot_age_in_deep_sea_cumulative_only(ax, particles, h, color=colors[i], linestyle=linestyles[i], label=labels[i])
    ax.set_xlim([0, 150])
    ax.set_xlabel('Particle age (days)')
    ax.set_ylim([0, 100])
    ax.set_ylabel('Cumulative particles\npassing depth range (%)')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax.legend(title='Depth (m)', loc='upper left', bbox_to_anchor=(1.01, 1.01))
    
    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_age_in_deep_sea(particles:Particles, h_deep_sea:float, total_particles=None,
                         color='#1b7931', age_lim=None, color_cumulative='k',
                         ax=None, show=True, output_path=None) -> plt.axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()
        ax.set_xlabel('Particle age (days)')
        ax.set_ylabel('Particles passing shelf break (%)')
        if age_lim is not None:
            ax.set_xlim([0, age_lim])

    _, age_arriving_ds, matrix_arriving_ds = particles.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
    n_dim_matrix = len(matrix_arriving_ds.shape)
    if n_dim_matrix == 2:
        n_deep_sea_per_age = np.sum(matrix_arriving_ds, axis=0)
    elif n_dim_matrix == 1:
        n_deep_sea_per_age = matrix_arriving_ds
    else:
        raise ValueError(f'You should not be able to get here: matrix_arriving_ds is {n_dim_matrix}D')
    
    if total_particles is None:
        total_particles = particles.lon.shape[0]
    f_deep_sea_per_age = n_deep_sea_per_age/total_particles*100 # divided by total # particles
    f_cumulative_per_age = np.cumsum(f_deep_sea_per_age)

    # histogram age arriving
    ax.bar(age_arriving_ds, f_deep_sea_per_age, color=color, width=1)
    ax.spines['left'].set_color(color)
    ax.tick_params(axis='y', colors=color)
    ax.yaxis.label.set_color(color)

    # cumulative particles in deep sea
    ax2 = ax.twinx()
    ax2.set_ylabel('Cumulative particles\npassing shelf break (%)', color=color_cumulative)
    ax2.plot(age_arriving_ds, f_cumulative_per_age, color=color_cumulative)
    ax2.tick_params(axis='y', colors=color_cumulative)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_timeseries_in_deep_sea(particles:Particles, h_deep_sea:float, total_particles=None,
                                ax=None, show=True, output_path=None) -> plt.axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        fig.tight_layout()
        ax = plt.axes()
        ax.set_ylabel('# of particles')

    n_deep_sea = particles.get_n_deep_sea(h_deep_sea)
    n_particles_sim = particles.get_n_particles_in_simulation()

    ax.plot(particles.time, n_deep_sea, '-k', label='# in deep sea')
    ax.plot(particles.time, n_particles_sim, '--k', label='# in simulation')
    ax.legend(loc='upper left')

    color_ax2='g'
    ax2 = ax.twinx()
    ax2.set_ylabel('Fraction in deep sea', color=color_ax2)
    if total_particles is None:
        total_particles = particles.lon.shape[0]
    ax2.plot(particles.time, n_deep_sea/total_particles, '-', color=color_ax2)
    ax2.tick_params(axis='y', labelcolor=color_ax2)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_particle_locations(particles:Particles, location_info:LocationInfo,
                            time=None, t=None,
                            ax=None, show=True, color='k') -> plt.axes:
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    if t is None and time is None:
        log.info('''Plotting initial particles locations. Specify either requested
                 time or time index t if you want to plot a different time.''')
        ax.plot(particles.lon0, particles.lat0, '.', color=color)
    elif t == 0:
        ax.plot(particles.lon0, particles.lat0, '.', color=color)
    elif t is not None and t != 0:
        ax.plot(particles.lon[:, t], particles.lat[:, t], '.', color=color)
    elif t is None and time is not None:
        t = get_closest_time_index(particles.time, time)
        ax.plot(particles.lon[:, t], particles.lat[:, t], '.', color=color)

    if show is True:
        plt.show()
    else:
        return ax

def plot_initial_particle_density_entering_deep_sea(particles:Particles, location_info:LocationInfo,
                                                    h_deep_sea:float, dx=0.01,
                                                    ax=None, show=True, output_path=None,
                                                    cmap='plasma', vmin=None, vmax=None,
                                                    filter_kelp_prob=None):
    
    bathymetry = BathymetryData.read_from_netcdf('input/cwa_roms_grid.nc')

    l_deep_sea = particles.get_l_deep_sea(h_deep_sea)
    l_deep_sea_any_time = np.sum(l_deep_sea, axis=1).astype('bool')
    lon0_ds = particles.lon0[l_deep_sea_any_time]
    lat0_ds = particles.lat0[l_deep_sea_any_time]
    lon0_nds = particles.lon0[~l_deep_sea_any_time]
    lat0_nds = particles.lat0[~l_deep_sea_any_time]

    if filter_kelp_prob is not None:
        kelp_prob = KelpProbability.read_from_tiff('input/perth_kelp_probability.tif')
        kelp_prob_ds = kelp_prob.get_kelp_probability_at_point(lon0_ds, lat0_ds)
        kelp_prob_nds = kelp_prob.get_kelp_probability_at_point(lon0_nds, lat0_nds)
        l_kelp_ds = kelp_prob_ds>=filter_kelp_prob
        l_kelp_nds = kelp_prob_nds>=filter_kelp_prob
        lon0_ds = lon0_ds[l_kelp_ds]
        lat0_ds = lat0_ds[l_kelp_ds]
        lon0_nds = lon0_nds[l_kelp_nds]
        lat0_nds = lat0_nds[l_kelp_nds]

    grid = DensityGrid(location_info.lon_range, location_info.lat_range, dx)
    density_ds = get_particle_density(grid, lon0_ds, lat0_ds)
    density_nds = get_particle_density(grid, lon0_nds, lat0_nds)
    density = density_ds/(density_ds+density_nds)*100
    
    l_no_particles = np.logical_and(density_ds==0, density_nds==0)
    density[l_no_particles] = np.nan

    x, y = np.meshgrid(grid.lon, grid.lat)

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    c = ax.pcolormesh(x, y, density, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(c)
    cbar.set_label(f'Particles passing shelf break\n(% per {dx}$^o$ grid cell)')

    ax = plot_contours(bathymetry.lon, bathymetry.lat, bathymetry.h, location_info,
                       ax=ax, show=False, color='#808080', show_perth_canyon=False, highlight_contour=None)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def get_colormap_reds(n):
    colors = ['#fece6b', '#fd8e3c', '#f84627', '#d00d20', '#b50026', '#950026', '#830026']
    return colors[:n]

def get_colormap_reds_blues(n):
    n = n // 2
    reds = ['#ffae82', '#eb7352', '#c73b33', '#950026']
    blues = ['#0a2a6a', '#3050a1', '#5f7acd', '#93a8ed']
    colors = blues[:n]
    for i in range(n):
        colors.append(reds[i])
    return colors

def plot_particle_density(grid:DensityGrid, density:np.ndarray,
                          cmap='Reds', ranges=[10**x for x in range(0, 7)],
                          c_label_description='Particle density',
                          ax=None, show=True, output_path=None):
    
    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

        bathymetry = BathymetryData.read_from_netcdf('input/cwa_roms_grid.nc')
        ax = plot_contours(bathymetry.lon, bathymetry.lat, bathymetry.h, location_info, ax=ax, show=False, show_perth_canyon=False, color='#757575')

    x, y = np.meshgrid(grid.lon, grid.lat)
    density[density==0] = np.nan

    if cmap == 'RedBlue':
        colors = get_colormap_reds_blues(len(ranges))
    else:
        colors = get_colormap_reds(len(ranges))
    cm = LinearSegmentedColormap.from_list('cm_log_density', colors, N=len(ranges))
    norm = BoundaryNorm(ranges, ncolors=len(ranges))
    c = ax.pcolormesh(x, y, density, cmap=cm, norm=norm, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(c)
    cbar.set_label(f'{c_label_description}\n(#/{grid.dx}$^o$ grid cell)')

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax, cbar, c

if __name__ == '__main__':
    location_info = get_location_info('cwa_perth')
    h_deep_sea = 600 # m depth: max Leeuwin Undercurrent depth

    input_path = f'{get_dir_from_json("opendrift")}cwa-perth_2017-Mar-Aug.nc'
    particles = Particles.read_from_netcdf(input_path)

    plt.rcParams.update({'font.size' : 15})
    
    # animation_path = f'{get_dir_from_json("plots")}cwa-perth_animation_2017-Mar-Aug.gif'
    # animate_particles(particles, location_info, h_deep_sea, output_path=animation_path)

    # output_path_histogram = f'{get_dir_from_json("plots")}cwa-perth_histogram_arriving_2017-Mar-Aug.jpg'
    # plot_histogram_arriving_in_deep_sea(particles, h_deep_sea, output_path=output_path_histogram, show=False)
    
    # output_path_density = f'{get_dir_from_json("plots")}cwa-perth_initial_density_p80_2017-Mar-Aug.jpg'
    # plot_initial_particle_density_entering_deep_sea(particles, get_location_info('perth'), h_deep_sea,
    #                                                 output_path=output_path_density, show=False, filter_kelp_prob=0.8)

    # output_path_age = f'{get_dir_from_json("plots")}cwa-perth_age_2017-Mar-Aug.jpg'
    # plot_age_in_deep_sea(particles, h_deep_sea, output_path=output_path_age, show=False)

    output_path_sensitivity = f'{get_dir_from_json("plots")}cwa-perth_deep_sea_sensitivity_2017-Mar-Aug.jpg'
    plot_percentage_in_deep_sea_depending_on_depth(particles, output_path=output_path_sensitivity, show=False)

    output_path_sensitivity_age = f'{get_dir_from_json("plots")}cwa-perth_deep_sea_age_sensitivity_2017-Mar-Aug.jpg'
    plot_particle_age_in_deep_sea_depending_on_depth(particles, output_path=output_path_sensitivity_age, show=False)
    