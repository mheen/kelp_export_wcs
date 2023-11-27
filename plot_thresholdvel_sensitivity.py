from location_info import get_location_info, LocationInfo
from tools.files import get_dir_from_json
from tools import log
from data.bathymetry_data import BathymetryData
from data.roms_data import RomsData, read_roms_grid_from_netcdf, get_vel_correction_factor_for_specific_height_above_sea_floor
from data.roms_data import get_daily_mean_roms_data, read_roms_data_from_multiple_netcdfs
from data.climate_data import read_dmi_data, read_mei_data
from data.kelp_data import KelpProbability
from data.wind_data import WindData, read_era5_wind_data_from_netcdf, get_daily_mean_wind_data, convert_u_v_to_meteo_vel_dir
from data.glider_data import GliderData
from plot_tools.general import add_subtitle
from plot_tools.basic_maps import plot_basic_map
from plot_tools.plots_bathymetry import plot_contours
from plot_tools.plots_roms import plot_exceedance_threshold_velocity, plot_roms_map_with_transect, plot_roms_map
from plot_tools.plots_roms import plot_roms_transect_specific_coordinates, plot_roms_transect_between_points
from plot_tools.plots_particles import plot_particle_density, _plot_age_in_deep_sea_cumulative_only
from plot_tools.plots_climate import plot_dmi_index, plot_mei_index
from particles import Particles, DensityGrid, get_particle_density
from dswc_detector import calculate_potential_energy_anomaly, exclude_roms_data_past_depth
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import numpy as np
import cmocean
import pandas as pd
import os

kelp_green = '#1b7931'

start_date = datetime(2017, 3, 1)
end_date = datetime(2017, 9, 30)

roms_dir = f'{get_dir_from_json("roms_data")}{start_date.year}/'
pts_dir = f'{get_dir_from_json("opendrift_output")}'
plots_dir = f'{get_dir_from_json("plots")}'

southern_boundary = -34.0
location_info = get_location_info('cwa_perth')
location_info.lat_range = [southern_boundary, location_info.lat_range[1]]
time_str = f'{start_date.strftime("%b")}{end_date.strftime("%b%Y")}'

roms_grid = read_roms_grid_from_netcdf('input/cwa_roms_grid.nc')
bathy = BathymetryData.read_from_netcdf(f'{get_dir_from_json("bathymetry")}',
                                        lon_str='lon', lat_str='lat', h_str='z',
                                        h_fac=-1)

# --- Particle density comparison ---
def plot_particle_density_comparison(pd1:np.ndarray, pd2:np.ndarray, pd_grid:DensityGrid,
                                     p1:Particles, p2:Particles, h_deep_sea:float,
                                     location_info:LocationInfo, title1:str, title2:str, 
                                     output_path:str):

    fig = plt.figure(figsize=(8, 11))
    plt.subplots_adjust(hspace=0.35)
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info)
    ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info, ax=ax1, show=False, show_perth_canyon=False, color='k', linewidths=0.5)
    ranges = [10**x for x in range(0, 7)]
    ticklabels = ['1', '10', '10$^2$', '10$^3$', '10$^4$', '10$^5$', '10$^6$']
    ax1, cbar1, c1 = plot_particle_density(pd_grid, pd1, location_info, ax=ax1, show=False)
    cbar1.remove()
    ax1 = add_subtitle(ax1, f'(a) {title1}')

    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2 = plot_basic_map(ax2, location_info, xmarkers='off')
    ax2 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info, ax=ax2, show=False, show_perth_canyon=False, color='k', linewidths=0.5)
    ax2, cbar2, c2 = plot_particle_density(pd_grid, pd2, location_info, ax=ax2, show=False)
    cbar2.remove()
    ax2 = add_subtitle(ax2, f'(b) {title2}')
    # colorbar
    l, b, w, h = ax1.get_position().bounds
    cbax = fig.add_axes([l, b-0.06, 2.2*w, 0.02])
    cbar = plt.colorbar(c1, ticks=ranges, orientation='horizontal', cax=cbax)
    cbar.ax.set_xticklabels(ticklabels)
    cbar.set_label(f'Particle density (#/{pd_grid.dx}$^o$ grid cell)')

    # timeseries age in deep sea
    # - non-decomposing
    ax3 = plt.subplot(2, 2, (3, 4))
    ax3, _, _ = _plot_age_in_deep_sea_cumulative_only(ax3, p1, h_deep_sea, label=title1)
    ax3, _, _ = _plot_age_in_deep_sea_cumulative_only(ax3, p2, h_deep_sea, linestyle='--', label=title2)
    ax3.set_xlim([0, 100])
    ax3.set_ylim([0, 80])
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_xlabel('Particle age (days)')
    ax3.set_ylabel(f'Particles past shelf edge at {h_deep_sea} m (%)')
    ax3.legend(loc='lower right')
    add_subtitle(ax3, '(c) Particle export comparison')
    # - decomposing
    k = -0.075
    total_particles = p1.lon.shape[0]
    
    _, age_arriving_ds1, matrix_arriving_ds1 = p1.get_matrix_release_age_arriving_deep_sea(200)
    n_deep_sea_per_age1 = np.sum(matrix_arriving_ds1, axis=0)
    n_deep_sea_decomposed1 = n_deep_sea_per_age1*np.exp(k*age_arriving_ds1)
    f_deep_sea_decomposed1 = n_deep_sea_decomposed1/total_particles*100
    f_decomposed1 = np.cumsum(f_deep_sea_decomposed1)
    
    _, age_arriving_ds2, matrix_arriving_ds2 = p2.get_matrix_release_age_arriving_deep_sea(200)
    n_deep_sea_per_age2 = np.sum(matrix_arriving_ds2, axis=0)
    n_deep_sea_decomposed2 = n_deep_sea_per_age2*np.exp(k*age_arriving_ds2)
    f_deep_sea_decomposed2 = n_deep_sea_decomposed2/total_particles*100
    f_decomposed2 = np.cumsum(f_deep_sea_decomposed2)
    
    ax4 = ax3.twinx()
    ax4.plot(age_arriving_ds1, f_decomposed1, color=kelp_green, linestyle='-')
    ax4.plot(age_arriving_ds2, f_decomposed2, color=kelp_green, linestyle='--')
    ax4.tick_params(axis='y', colors=kelp_green)
    ax4.yaxis.label.set_color(kelp_green)
    ax4.set_ylabel('Particles past shelf edge at 200 m (%)\n accounting for decomposition')
    ax4.set_ylim([0, 80])

    log.info(f'Saving figure to: {output_path}')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

def remove_southern_boundary(particles:Particles, s_lat=southern_boundary) -> Particles:
    i_s, j_s = np.where(particles.lat < s_lat)
    for k in range(len(i_s)):
        particles.lon[i_s[k], j_s[k]:] = np.nan
        particles.lat[i_s[k], j_s[k]:] = np.nan
        particles.z[i_s[k], j_s[k]:] = np.nan
        particles.age[i_s[k], j_s[k]:] = np.nan
    
    return particles


input_path_p_baseline = f'{pts_dir}cwa_perth_MarSep2017_baseline.nc'
p_baseline = Particles.read_from_netcdf(input_path_p_baseline)
p_baseline = remove_southern_boundary(p_baseline)
dx = 0.05
h_deep_sea = 200 # m

pd_grid = DensityGrid(location_info.lon_range, location_info.lat_range, dx)
density_baseline = get_particle_density(pd_grid, p_baseline.lon, p_baseline.lat)

input_path_p_threshold_high = f'{pts_dir}sensitivity/cwa_perth_MarSep2017_thresholdvel-012.nc'
p_threshold_high = Particles.read_from_netcdf(input_path_p_threshold_high)
p_threshold_high = remove_southern_boundary(p_threshold_high)
density_threshold_high = get_particle_density(pd_grid, p_threshold_high.lon, p_threshold_high.lat)


output_pd_thres_comparison = f'{plots_dir}threshold_velocity_0-12.pdf'
plot_particle_density_comparison(density_baseline, density_threshold_high, pd_grid,
                                p_baseline, p_threshold_high, h_deep_sea,
                                location_info, 'No threshold Mar-Sep 2017', 'Threshold 0.12 m/s Mar-Sep 2017',
                                output_pd_thres_comparison)