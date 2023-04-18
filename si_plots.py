from location_info import get_location_info, LocationInfo
from tools.files import get_dir_from_json
from tools import log
from data.bathymetry_data import BathymetryData
from data.roms_data import read_roms_grid_from_netcdf, get_vel_correction_factor_for_specific_height_above_sea_floor
from data.climate_data import read_dmi_data, read_mei_data
from plot_tools.general import add_subtitle
from plot_tools.basic_maps import plot_basic_map
from plot_tools.plots_bathymetry import plot_contours
from plot_tools.plots_roms import plot_exceedance_threshold_velocity, plot_roms_map
from plot_tools.plots_particles import plot_particle_density, _plot_age_in_deep_sea_cumulative_only
from plot_tools.plots_climate import plot_dmi_index, plot_mei_index
from particles import Particles, DensityGrid, get_particle_density
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import cartopy.crs as ccrs
import numpy as np

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)

start_date = datetime(2017, 3, 1)
end_date = datetime(2017, 8, 31)

roms_dir = f'{get_dir_from_json("roms_data")}cwa/{start_date.year}/'
pts_dir = f'{get_dir_from_json("opendrift_output")}'
plots_dir = f'{get_dir_from_json("plots")}si/'

location_info = get_location_info('cwa_perth')
time_str = f'{start_date.strftime("%b")}{end_date.strftime("%b%Y")}'

roms_grid = read_roms_grid_from_netcdf('input/cwa_roms_grid.nc')

# ---------------------------------------------------------------------------------
# Climate indices
# ---------------------------------------------------------------------------------
# time_mei, mei = read_mei_data()
# time_dmi, dmi = read_dmi_data()

# output_path = f'{plots_dir}figure_s1.jpg'
# xlim = [datetime(2000, 1, 1), datetime(2023, 1, 1)]

# fig = plt.figure(figsize=(10, 10))
# ax1 = plt.subplot(2, 1, 1)
# ax1 = plot_mei_index(time_mei, mei, ax=ax1, xlim=xlim, show=False)
# ax1.axvspan(datetime(2017, 3, 1), datetime(2017, 8, 31), alpha=0.5, color='#808080')
# ax1.xaxis.set_major_locator(mdates.YearLocator())
# ax1.set_xticklabels([])
# ax1.set_title('(a) El Nino Southern Oscillation indicator')

# ax2 = plt.subplot(2, 1, 2)
# ax2 = plot_dmi_index(time_dmi, dmi, ax=ax2, xlim=xlim, show=False)
# ax2.axvspan(datetime(2017, 3, 1), datetime(2017, 8, 31), alpha=0.5, color='#808080')
# ax2.xaxis.set_major_locator(mdates.YearLocator())
# for label in ax2.get_xticklabels(which='major'):
#     label.set(rotation=90, horizontalalignment='center')
# ax2.set_title('(b) Indian Ocean Dipole indicator')

# log.info(f'Saving figure to: {output_path}')
# plt.savefig(output_path, bbox_inches='tight', dpi=300)
# plt.close()

# ---------------------------------------------------------------------------------
# ROMS
# ---------------------------------------------------------------------------------
# # --- Exceedance threshold velocity plots ---
# thres_vel = 0.045#, 0.031]
# thres_sd = 0.016#, 0.015]
# thres_name = 'Ecklonia'#, 'Ecklonia (medium)']

# output_exceedance_his = f'{plots_dir}figure_s7a.jpg'
# output_exceedance_map = f'{plots_dir}figure_s7b.jpg'

# plot_exceedance_threshold_velocity(roms_dir, start_date, end_date, thres_vel, thres_sd, thres_name,
#                                    location_info, output_exceedance_his, output_exceedance_map)

# # --- Bottom layer depth ---
# output_bottom_layer_depth = f'{plots_dir}/figure_s4.jpg'
# layer_depth = roms_grid.z[1, :, :]-roms_grid.z[0, :, :]

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax = plot_basic_map(ax, location_info)
# ax = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info, ax=ax, show=False, show_perth_canyon=False, color='#757575')
# c = ax.pcolormesh(roms_grid.lon, roms_grid.lat, np.log10(layer_depth), cmap='viridis_r', vmin=0, vmax=3)
# cbar = plt.colorbar(c)
# ticks = [1, 2, 4, 6, 8, 10, 25, 50, 100, 250, 500, 1000]
# cbar.set_ticks(np.log10(ticks))
# cbar.set_ticklabels(ticks)
# cbar.set_label('Layer thickness (m)')
# ax.set_title('Thickness of CWA-ROMS bottom layer')
# log.info(f'Saving figure to: {output_bottom_layer_depth}')
# plt.savefig(output_bottom_layer_depth, bbox_inches='tight', dpi=300)

# # --- Logarithmic bottom layer ---
# # u(z) = u*/kappa*log(z/z0)
# # u* = sqrt(tau_b)
# # tau_b = kappa**2*u_sigma0**2/log**2(z_sigma0/z0)
# output_logarithmic_bottom_profiles = f'{plots_dir}figure_s5a.jpg'

# kappa = 0.41 # von Karman constant
# z0 = 1.65*10**(-5) # m bottom roughness
# u_sigma0 = 1 # m/s (using 1 so that it becomes a multiplication factor as a function of depth)
# z_sigma0 = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0]) # different layer depths

# ax = plt.axes()
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.set_xlabel('Velocity in logarithmic bottom layer (m/s)')
# ax.set_ylabel('Height above sea floor (m)')

# linestyles = ['-', '--', ':', '-', '--', ':']
# colors = ['k', 'k', 'k', '#808080', '#808080', '#808080']

# z = np.arange(z0, 20, 0.1)
# for i in range(len(z_sigma0)):
#     tau_b = kappa**2*u_sigma0**2/(np.log(z_sigma0[i]/z0))**2
#     u = np.sqrt(tau_b)/kappa*np.log(z/z0)
#     u[u>1.0] = 1.0
#     ax.plot(u, z, label=z_sigma0[i]*2, color=colors[i], linestyle=linestyles[i]) # label=z_sigma0*2 because velocities are in center of sigma layers

# ax.set_xlim([0.60, 1.0])
# ax.set_ylim([0.0, 20.0])
# ax.legend(loc='upper left', title='Bottom layer thickness (m):')
# log.info(f'Saving figure to: {output_logarithmic_bottom_profiles}')
# plt.savefig(output_logarithmic_bottom_profiles, bbox_inches='tight', dpi=300)
# plt.close()

# # --- Logarithmic bottom layer correction spatial variation ---
# z_drift = 0.5 # m -> assuming that seaweed would drift at 50 cm above seafloor
# output_logarithmic_correction = f'{plots_dir}figure_s5b.jpg'
# u_corr = get_vel_correction_factor_for_specific_height_above_sea_floor(z_drift)

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax = plot_basic_map(ax, location_info)
# ax = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info, ax=ax, show=False, show_perth_canyon=False, color='#757575')
# c = ax.pcolormesh(roms_grid.lon, roms_grid.lat, u_corr)
# cbar = plt.colorbar(c)
# cbar.set_label('Correction factor to current velocities')
# ax.set_title(f'Drift at {z_drift} m above sea floor')
# log.info(f'Saving figure to: {output_logarithmic_correction}')
# plt.savefig(output_logarithmic_correction, bbox_inches='tight', dpi=300)
# plt.close()

# ---------------------------------------------------------------------------------
# PARTICLES
# ---------------------------------------------------------------------------------
# input_path_p_baseline = f'{pts_dir}cwa_perth_MarAug2017_baseline.nc'
# input_path_p_threshold = f'{pts_dir}sensitivity/cwa_perth_MarAug2017_thresholdvel.nc'
# p_baseline = Particles.read_from_netcdf(input_path_p_baseline)
# p_threshold = Particles.read_from_netcdf(input_path_p_threshold)

# --- Particle density comparison ---
def plot_particle_density_comparison(pd1:np.ndarray, pd2:np.ndarray, pd_grid:DensityGrid,
                                     location_info:LocationInfo, title1:str, title2:str, 
                                     output_path:str, cmap='summer'):
    bathymetry = BathymetryData.read_from_netcdf('input/cwa_roms_grid.nc')

    fig = plt.figure(figsize=(11, 8))
    ax1 = plt.subplot(1, 3, 1, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info)
    ax1 = plot_contours(bathymetry.lon, bathymetry.lat, bathymetry.h, location_info, ax=ax1, show=False, show_perth_canyon=False, color='#757575')
    ranges = [10**x for x in range(0, 7)]
    ticklabels = ['1', '10', '10$^2$', '10$^3$', '10$^4$', '10$^5$', '10$^6$']
    ax1, cbar1, c1 = plot_particle_density(pd_grid, pd1, location_info, ax=ax1, show=False, cmap=cmap)
    cbar1.remove()
    ax1 = add_subtitle(ax1, title1)

    ax2 = plt.subplot(1, 3, 2, projection=ccrs.PlateCarree())
    ax2 = plot_basic_map(ax2, location_info, xmarkers='off')
    ax2 = plot_contours(bathymetry.lon, bathymetry.lat, bathymetry.h, location_info, ax=ax2, show=False, show_perth_canyon=False, color='#757575')
    ax2, cbar2, c2 = plot_particle_density(pd_grid, pd2, location_info, ax=ax2, show=False, cmap=cmap)
    cbar2.remove()
    ax2 = add_subtitle(ax2, title2)
    # colorbar
    l, b, w, h = ax1.get_position().bounds
    cbax = fig.add_axes([l, b-0.06, 2.2*w, 0.02])
    cbar = plt.colorbar(c1, ticks=ranges, orientation='horizontal', cax=cbax)
    cbar.ax.set_xticklabels(ticklabels)
    cbar.set_label(f'Particle density (#/{pd_grid.dx}$^o$ grid cell)')

    log.info(f'Saving figure to: {output_path}')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)

# dx = 0.05

# pd_grid = DensityGrid(location_info.lon_range, location_info.lat_range, dx)
# density_baseline = get_particle_density(pd_grid, p_baseline.lon, p_baseline.lat)
# density_threshold = get_particle_density(pd_grid, p_threshold.lon, p_threshold.lat)

# output_pd_thres_comparison = f'{plots_dir}figure_s8a.jpg'
# plot_particle_density_comparison(density_baseline, density_threshold, pd_grid, location_info,
#                                  '(a) Baseline Mar-Aug 2017', '(b) Threshold vel. Mar-Aug 2017',
#                                  output_pd_thres_comparison, cmap='summer')

# # -- Particle timeseries deep sea comparison ---
# h_deep_sea = 200 # m
# output_ptime_thres_comparison = f'{plots_dir}figure_s8b.jpg'

# fig = plt.figure(figsize=(11, 6))
# ax = plt.axes()
# ax = _plot_age_in_deep_sea_cumulative_only(ax, p_baseline, h_deep_sea, label='Baseline Mar-Aug 2017')
# ax = _plot_age_in_deep_sea_cumulative_only(ax, p_threshold, h_deep_sea, linestyle='--', label='Threshold vel. Mar-Aug 2017')
# ax.set_xlim([0, 100])
# ax.set_ylim([0, 80])
# ax.grid(True, linestyle='--', alpha=0.5)
# ax.set_xlabel('Particle age (days)')
# ax.set_ylabel(f'Cumulative particles passing shelf break at {h_deep_sea} m (%)')
# ax.legend(loc='lower right')
# log.info(f'Saving figure to {output_ptime_thres_comparison}')
# plt.savefig(output_ptime_thres_comparison, bbox_inches='tight', dpi=300)
