from particles import Particles
from location_info import LocationInfo, get_location_info
from data.bathymetry_data import BathymetryData
from plot_tools.basic_maps import plot_basic_map
from plot_tools.plots_bathymetry import plot_contours
from tools.files import get_dir_from_json
from tools import log
from dataclasses import dataclass
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import random

# litter bag locations
@dataclass
class Litterbag:
    name: str
    depth: float
    lon_range: list
    lat_range: list

def get_litterbags():
    lon_range_deep = [115.41, 115.45]
    lat_range_deep = [-31.94, -31.91]
    lon_range_shallow = [115.67, 115.71]
    lat_range_shallow = [-31.86, -31.76]
    litterbags = [Litterbag('shallow', '10-20', lon_range_shallow, lat_range_shallow),
                  Litterbag('deep', 50, lon_range_deep, lat_range_deep)]
    return litterbags

def _get_age_and_percentage_particles_past_shelf(particles:Particles, h_deep_sea:float) -> tuple:

    _, age_arriving_ds, matrix_arriving_ds = particles.get_matrix_release_age_arriving_deep_sea(h_deep_sea)
    n_deep_sea_per_age = np.sum(matrix_arriving_ds, axis=0)
    total_particles = particles.lon.shape[0]
    f_deep_sea_per_age = n_deep_sea_per_age/total_particles*100 # divided by total # particles
    f_cumulative_per_age = np.cumsum(f_deep_sea_per_age)
    
    return age_arriving_ds, f_cumulative_per_age

def write_age_and_percentage_particles_past_shelf_to_csv(particles:Particles, litterbag:Litterbag,
                                                         h_deep_sea:float, output_path:str):

    particles_lb = particles.filter_based_on_release_lon_lat_range(litterbag.lon_range, litterbag.lat_range)

    age, percentage_particles = _get_age_and_percentage_particles_past_shelf(particles_lb, h_deep_sea)

    df = pd.DataFrame()
    df['Particle age (days)'] = age
    df[f'Percentage particles past shelf ({h_deep_sea} m)'] = percentage_particles

    log.info(f'Writing age and percentage of particles past shelf to: {output_path}')
    df.to_csv(output_path, index=False)

def plot_timeseries_particle_age_percentage_past_shelf(particles:Particles, h_deep_sea:float, litterbags:list,
                                                       show=True, output_path=None):
    
    fig = plt.figure(figsize=(3, 5))
    ax = plt.axes()

    linestyles = ['-', '-']
    colors = ['#172cd7', '#1d631b']

    for i, litterbag in enumerate(litterbags):
        particles_lb = particles.filter_based_on_release_lon_lat_range(litterbag.lon_range, litterbag.lat_range)
        age, percentage = _get_age_and_percentage_particles_past_shelf(particles_lb, h_deep_sea)
        
        ax.plot(age, percentage, linestyle=linestyles[i], color=colors[i], linewidth=3, label=f'{litterbag.depth} m')

    ax.set_xlim([0, 60])
    ax.set_xlabel('Particle age (days)')
    
    ax.set_ylim([0, 80])
    ax.set_ylabel('Particles past continental shelf edge (%)')

    ax.legend(loc='lower right', title='Release sites')

    ax.grid(True, linestyle='--', alpha=0.5)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_litterbag_locations_and_tracks_past_shelf(particles:Particles, h_deep_sea:float, litterbags:list,
                                                   location_info:LocationInfo, bathymetry:BathymetryData,
                                                   plot_n_tracks=10, show=True, output_path=None):
    
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plot_basic_map(ax, location_info)
    ax = plot_contours(bathymetry.lon, bathymetry.lat, bathymetry.h, location_info, ax=ax, show=False,
                       show_perth_canyon=False, highlight_contour=[h_deep_sea], color='#989898')

    def plot_box(lon_range, lat_range, color):
        ax.plot([lon_range[0], lon_range[1]], [lat_range[0], lat_range[0]], '-', color=color)
        ax.plot([lon_range[1], lon_range[1]], [lat_range[0], lat_range[1]], '-', color=color)
        ax.plot([lon_range[1], lon_range[0]], [lat_range[1], lat_range[1]], '-', color=color)
        ax.plot([lon_range[0], lon_range[0]], [lat_range[1], lat_range[0]], '-', color=color)

    legend_elements = []

    for litterbag in litterbags:

        particles_lb = particles.filter_based_on_release_lon_lat_range(litterbag.lon_range, litterbag.lat_range)
        l_ds = particles_lb.get_l_deep_sea(h_deep_sea)
        i_ds_lb = np.where(np.any(l_ds, axis=1))[0]
        i_plots = random.sample(list(i_ds_lb), plot_n_tracks)

        if litterbag.name == 'shallow':
            colors = iter(cm.Blues(np.linspace(0.3, 1, len(i_plots))))
            c_lb = '#307abc'
        elif litterbag.name == 'deep':
            colors = iter(cm.Greens(np.linspace(0.3, 1, len(i_plots))))
            c_lb = '#29924e'

        for i in i_plots:
            c = next(colors)

            t1 = np.where(l_ds[i, :])[0][0]
            t0 = np.where(~np.isnan(particles_lb.lon[i, :]))[0][0]

            ax.plot(particles_lb.lon[i, :t1+1], particles_lb.lat[i, :t1+1], '-', color=c)
            ax.plot(particles_lb.lon[i, t0], particles_lb.lat[i, t0], 'o', color=c_lb, markersize=8, markeredgecolor='k')
            ax.plot(particles_lb.lon[i, t1+1], particles_lb.lat[i, t1+1], 'x', color=c_lb)

        lb_label = f'{litterbag.depth} m'
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=c_lb, markersize=10, label=lb_label))

    ax.legend(handles=legend_elements, loc='upper left', title='Release sites')

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

if __name__ == '__main__':
    h_deep_sea = 200 # m (continental shelf edge)

    particles = Particles.read_from_netcdf(f'{get_dir_from_json("opendrift_output")}litterbags/litterbags_JanFeb2018.nc')

    litterbags = get_litterbags()

    for litterbag in litterbags:
        output_path = f'{get_dir_from_json("plots")}litterbags/litterbag_{litterbag.name}_particle_age_percentage_full_year.csv'
        write_age_and_percentage_particles_past_shelf_to_csv(particles, litterbag, h_deep_sea, output_path)

    location_info = get_location_info('cwa_perth_zoom2')
    bathymetry = BathymetryData.read_from_netcdf('input/cwa_roms_grid.nc')
    output_path = f'{get_dir_from_json("plots")}litterbags/example_particle_tracks_from_litterbags_to_shelf_edge.jpg'
    plot_litterbag_locations_and_tracks_past_shelf(particles, h_deep_sea, litterbags, location_info, bathymetry, show=False, output_path=output_path)

    output_path = f'{get_dir_from_json("plots")}litterbags/age_percentage_past_shelf_from_litterbags_full_year.jpg'
    plot_timeseries_particle_age_percentage_past_shelf(particles, h_deep_sea, litterbags, show=False, output_path=output_path)
