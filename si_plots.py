from location_info import get_location_info, LocationInfo
from tools.files import get_dir_from_json
from tools import log
from data.bathymetry_data import BathymetryData
from data.roms_data import read_roms_grid_from_netcdf, get_vel_correction_factor_for_specific_height_above_sea_floor
from data.climate_data import read_dmi_data, read_mei_data
from data.kelp_data import KelpProbability
from data.wind_data import WindData, read_era5_wind_data_from_netcdf, get_daily_mean_wind_data, convert_u_v_to_meteo_vel_dir
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
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import numpy as np
import cmocean
import pandas as pd
import os

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

# output_path = f'{plots_dir}figs1.jpg'
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
# # --- Horizontal resolution ---
# dx = np.sqrt(1/roms_grid.pm*1/roms_grid.pn) # square root of grid cell areas for approximate resolution (m)

# output_resolution = f'{plots_dir}figs3.jpg'

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax = plot_basic_map(ax, location_info)
# ax = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info, ax=ax, show=False, show_perth_canyon=False, color='k', linewidths=0.7)
# c = ax.pcolormesh(roms_grid.lon, roms_grid.lat, dx, vmin=1700, vmax=2100, cmap='viridis')
# cbar = plt.colorbar(c)
# cbar.set_label('Square root of grid cell area (m)')
# ax.set_title('Approximate horizontal resolution of CWA-ROMS grid')
# log.info(f'Saving figure to: {output_resolution}')
# plt.savefig(output_resolution, bbox_inches='tight', dpi=300)
# plt.close()

# # --- Exceedance threshold velocity plots ---
# thres_vel = 0.045#, 0.031]
# thres_sd = 0.016#, 0.015]
# thres_name = 'Ecklonia'#, 'Ecklonia (medium)']

# output_exceedance = f'{plots_dir}figs7.jpg'

# plot_exceedance_threshold_velocity(roms_dir, start_date, end_date, thres_vel, thres_sd, thres_name,
#                                    location_info, output_exceedance)

# # --- Bottom layer depth ---
# output_bottom_layer_depth = f'{plots_dir}/figs4.jpg'
# layer_depth = roms_grid.z[1, :, :]-roms_grid.z[0, :, :]

# ax = plt.axes(projection=ccrs.PlateCarree())
# ax = plot_basic_map(ax, location_info)
# ax = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info, ax=ax, show=False, show_perth_canyon=False, color='k', linewidths=0.7)
# c = ax.pcolormesh(roms_grid.lon, roms_grid.lat, np.log10(layer_depth), cmap=cmocean.cm.deep, vmin=0, vmax=3)
# cbar = plt.colorbar(c)
# ticks = [1, 2, 4, 6, 8, 10, 25, 50, 100, 250, 500, 1000]
# cbar.set_ticks(np.log10(ticks))
# cbar.set_ticklabels(ticks)
# cbar.set_label('Layer thickness (m)')
# ax.set_title('Thickness of CWA-ROMS bottom layer')
# log.info(f'Saving figure to: {output_bottom_layer_depth}')
# plt.savefig(output_bottom_layer_depth, bbox_inches='tight', dpi=300)
# plt.close()

# # --- Logarithmic bottom layer ---
# # u(z) = u*/kappa*log(z/z0)
# # u* = sqrt(tau_b)
# # tau_b = kappa**2*u_sigma0**2/log**2(z_sigma0/z0)
# output_logprofiles = f'{plots_dir}figs5.jpg'

# kappa = 0.41 # von Karman constant
# z0 = 1.65*10**(-5) # m bottom roughness
# u_sigma0 = 1 # m/s (using 1 so that it becomes a multiplication factor as a function of depth)
# z_sigma0 = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0]) # different layer depths

# fig = plt.figure(figsize=(8, 10))
# # Logarithmic profiles
# ax1 = plt.subplot(2, 1, 1)
# ax1.grid(True, linestyle='--', alpha=0.5)
# ax1.set_xlabel('Velocity in logarithmic bottom layer (m/s)')
# ax1.set_ylabel('Height above sea floor (m)')

# linestyles = ['-', '--', ':', '-', '--', ':']
# colors = ['k', 'k', 'k', '#808080', '#808080', '#808080']

# z = np.arange(z0, 20, 0.1)
# for i in range(len(z_sigma0)):
#     tau_b = kappa**2*u_sigma0**2/(np.log(z_sigma0[i]/z0))**2
#     u = np.sqrt(tau_b)/kappa*np.log(z/z0)
#     u[u>1.0] = 1.0
#     ax1.plot(u, z, label=z_sigma0[i]*2, color=colors[i], linestyle=linestyles[i]) # label=z_sigma0*2 because velocities are in center of sigma layers

# ax1.set_xlim([0.60, 1.0])
# ax1.set_ylim([0.0, 20.0])
# ax1.legend(loc='lower left', title='Bottom layer\nthickness (m):')
# add_subtitle(ax1, '(a) Logarithmic velocity profiles for different layer depths')

# # Spatial variation
# z_drift = 0.5 # m -> assuming that seaweed would drift at 50 cm above seafloor
# u_corr = get_vel_correction_factor_for_specific_height_above_sea_floor(z_drift)

# ax2 = plt.subplot(2, 1, 2, projection=ccrs.PlateCarree())
# ax2 = plot_basic_map(ax2, location_info)
# ax2 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info, ax=ax2, show=False, show_perth_canyon=False, color='k', linewidths=0.7)
# c = ax2.pcolormesh(roms_grid.lon, roms_grid.lat, u_corr, cmap=cmocean.cm.ice)
# cbar = plt.colorbar(c)
# cbar.set_label('Correction factor to current velocities')
# add_subtitle(ax2, f'(b) Correction factor for drift at\n     {z_drift} m above sea floor')
# l1, b1, w1, h1 = ax1.get_position().bounds
# l2, b2, w2, h2 = ax2.get_position().bounds
# ax2.set_position([l1+w2/2, b2, w2, h2])

# log.info(f'Saving figure to: {output_logprofiles}')
# plt.savefig(output_logprofiles, bbox_inches='tight', dpi=300)
# plt.close()

# ---------------------------------------------------------------------------------
# PARTICLES
# ---------------------------------------------------------------------------------
# input_path_p_baseline = f'{pts_dir}cwa_perth_MarSep2017_baseline.nc'
# input_path_p_threshold = f'{pts_dir}sensitivity/cwa_perth_MarSep2017_thresholdvel.nc'
# input_path_p_logarithmic = f'{pts_dir}sensitivity/cwa_perth_MarSep2017_logarithmicvel.nc'
# p_baseline = Particles.read_from_netcdf(input_path_p_baseline)
# p_threshold = Particles.read_from_netcdf(input_path_p_threshold)
# p_logarithmic = Particles.read_from_netcdf(input_path_p_logarithmic)

# --- Particle density comparison ---
def plot_particle_density_comparison(pd1:np.ndarray, pd2:np.ndarray, pd_grid:DensityGrid,
                                     p1:Particles, p2:Particles, h_deep_sea:float,
                                     location_info:LocationInfo, title1:str, title2:str, 
                                     output_path:str):
    bathymetry = BathymetryData.read_from_netcdf('input/cwa_roms_grid.nc')

    fig = plt.figure(figsize=(8, 11))
    plt.subplots_adjust(hspace=0.35)
    ax1 = plt.subplot(2, 2, 1, projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info)
    ax1 = plot_contours(bathymetry.lon, bathymetry.lat, bathymetry.h, location_info, ax=ax1, show=False, show_perth_canyon=False, color='k', linewidths=0.5)
    ranges = [10**x for x in range(0, 7)]
    ticklabels = ['1', '10', '10$^2$', '10$^3$', '10$^4$', '10$^5$', '10$^6$']
    ax1, cbar1, c1 = plot_particle_density(pd_grid, pd1, location_info, ax=ax1, show=False)
    cbar1.remove()
    ax1 = add_subtitle(ax1, f'(a) {title1}')

    ax2 = plt.subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2 = plot_basic_map(ax2, location_info, xmarkers='off')
    ax2 = plot_contours(bathymetry.lon, bathymetry.lat, bathymetry.h, location_info, ax=ax2, show=False, show_perth_canyon=False, color='k', linewidths=0.5)
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
    ax3 = plt.subplot(2, 2, (3, 4))
    ax3 = _plot_age_in_deep_sea_cumulative_only(ax3, p1, h_deep_sea, label=title1)
    ax3 = _plot_age_in_deep_sea_cumulative_only(ax3, p2, h_deep_sea, linestyle='--', label=title2)
    ax3.set_xlim([0, 100])
    ax3.set_ylim([0, 80])
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_xlabel('Particle age (days)')
    ax3.set_ylabel(f'Cumulative particles passing shelf break at {h_deep_sea} m (%)')
    ax3.legend(loc='lower right')
    add_subtitle(ax3, '(c) Particle age comparison passing shelf break')

    log.info(f'Saving figure to: {output_path}')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()

# dx = 0.05
# h_deep_sea = 200 # m

# pd_grid = DensityGrid(location_info.lon_range, location_info.lat_range, dx)
# density_baseline = get_particle_density(pd_grid, p_baseline.lon, p_baseline.lat)
# density_threshold = get_particle_density(pd_grid, p_threshold.lon, p_threshold.lat)
# density_logarithmic = get_particle_density(pd_grid, p_logarithmic.lon, p_logarithmic.lat)

# output_pd_log_comparison = f'{plots_dir}figs6.jpg'
# plot_particle_density_comparison(density_baseline, density_logarithmic, pd_grid,
#                                  p_baseline, p_logarithmic, h_deep_sea,
#                                  location_info, 'Baseline Mar-Aug 2017', 'Logarithmic vel. Mar-Aug 2017',
#                                  output_pd_log_comparison)

# output_pd_thres_comparison = f'{plots_dir}figs8.jpg'
# plot_particle_density_comparison(density_baseline, density_threshold, pd_grid,
#                                  p_baseline, p_threshold, h_deep_sea,
#                                  location_info, 'Baseline Mar-Aug 2017', 'Threshold vel. Mar-Aug 2017',
#                                  output_pd_thres_comparison)

# ---------------------------------------------------------------------------------
# DSWC
# ---------------------------------------------------------------------------------
output_dswc_conditions = f'{plots_dir}figs9.jpg'

start_date = datetime(2017, 3, 1)
end_date = datetime(2017, 9, 30)
location_info_perth = get_location_info('perth')
wind_data = read_era5_wind_data_from_netcdf(get_dir_from_json("era5_data"), start_date, end_date,
                                            lon_range=location_info_perth.lon_range,
                                            lat_range=location_info_perth.lat_range)
wind_data = get_daily_mean_wind_data(wind_data)
u_mean = np.nanmean(np.nanmean(wind_data.u, axis=1), axis=1)
v_mean = np.nanmean(np.nanmean(wind_data.v, axis=1), axis=1)
vel_mean, dir_mean = convert_u_v_to_meteo_vel_dir(u_mean, v_mean)

def read_dswc_components(csv_gw='temp_data/gravitational_wind_components_in_time.csv'):
    if not os.path.exists(csv_gw):
        raise ValueError(f'''Gravitational vs wind components file does not yet exist: {csv_gw}
                            Please create it first by running write_gravitation_wind_components_to_csv (in dswc_detector.py)''')
    df = pd.read_csv(csv_gw)
    time_gw = [datetime.strptime(t, '%Y-%m-%d') for t in df['time'].values][:-1]
    grav_c = df['grav_component'].values[:-1]
    wind_c = df['wind_component'].values[:-1]
    drhodx = df['drhodx'].values[:-1]
    phi = df['phi'].values[:-1]
    return time_gw, grav_c, wind_c, drhodx, phi

def determine_l_time_dwswc_conditions(dir_mean):
    time, g, w, drhodx, phi = read_dswc_components()
    l_drhodx = drhodx < 0
    l_phi = phi > 1
    l_prereq = np.logical_and(l_drhodx, l_phi)
    l_components = g > w
    l_onshore = np.logical_and(225 < dir_mean, dir_mean < 315)
    l_wind = np.logical_or(l_components, l_onshore)
    l_dswc = np.logical_and(l_prereq, l_components)
    return l_drhodx, l_phi, l_components, l_wind, l_onshore, l_dswc
    
time_gw, grav_c, wind_c, drhodx, phi = read_dswc_components()
l_drhodx, l_phi, l_components, l_wind, l_onshore, l_dswc = determine_l_time_dwswc_conditions(dir_mean)

ocean_blue = '#25419e'

fig = plt.figure(figsize=(8, 15))
plt.subplots_adjust(hspace=0.1)

# (a) timeseries density gradient
ax1 = plt.subplot(6, 2, (1, 2))
scale_rho = 10**5
scale_rho_str = '10$^{-5}$'
drhodx_units = 'kg m$^{-4}$'

ax1.plot(time_gw, drhodx*scale_rho, '-k')
ax1.plot([time_gw[0], time_gw[-1]], [0, 0], '-', color='#808080')
ax1.set_xlim([time_gw[0], time_gw[-1]])
ax1.set_ylim([-3.5, 3.0])
ax1.set_xticklabels([])
ax1.set_ylabel(f'Density gradient\n({scale_rho_str} {drhodx_units})')
add_subtitle(ax1, '(a) Horizontal density gradient')

ylim1 = ax1.get_ylim()
ax1.fill_between(time_gw, ylim1[0], ylim1[1], where=l_drhodx, color=ocean_blue, alpha=0.3)
ax1.set_ylim(ylim1)
ax1.grid(True, linestyle='--', axis='x')

# (b) timeseries PEA
ax2 = plt.subplot(6, 2, (3, 4))
ax2.plot(time_gw, phi, '-k')
ax2.set_xlim([time_gw[0], time_gw[-1]])
ax2.set_ylabel('PEA (J m$^{-3}$)')
ax2.set_ylim([0, 12.5])
ax2.set_xticklabels([])
add_subtitle(ax2, '(b) Potential energy anomaly')

ylim2 = ax2.get_ylim()
ax2.fill_between(time_gw, ylim2[0], ylim2[1], where=l_phi, color=ocean_blue, alpha=0.3)
ax2.set_ylim(ylim2)
ax2.grid(True, linestyle='--', axis='x')

# (c) timeseries gravitational vs wind components
gw_scale = 10**5
gw_scale_str = '10$^{-5}$'
gw_unit_str = 'J m$^{-3}$ s$^{-1}$'

ax4 = plt.subplot(6, 2, (5, 6))
ax4.plot(time_gw, grav_c*gw_scale, '-k', label='Grav.')
ax4.plot(time_gw, wind_c*gw_scale, '--', color=ocean_blue, label='Wind')
l5 = ax4.legend(loc='upper left', bbox_to_anchor=(0.0, 0.9))
ax4.set_xlim([time_gw[0], time_gw[-1]])
ax4.set_xticklabels([])
ax4.set_ylim([0, 2.0])
ax4.set_ylabel(f'Component strength\n({gw_scale_str} {gw_unit_str})')
add_subtitle(ax4, '(c) Gravitational stratification versus wind mixing components')

ylim4 = ax4.get_ylim()
ax4.fill_between(time_gw, ylim4[0], ylim4[1], where=l_components, color=ocean_blue, alpha=0.3)
ax4.set_ylim(ylim4)
ax4.grid(True, linestyle='--', axis='x')

# (d) timeseries wind
ax3 = plt.subplot(6, 2, (7, 8))

colors_w = ['#808080', ocean_blue]
edge_colors_w = ['k', '#434343']

ax3.plot(wind_data.time, vel_mean, '--', color='#282828')

for i in range(len(wind_data.time)):
    i_color = l_onshore[i].astype(int)
    rotation = 270-dir_mean[i]
    ax3.text(mdates.date2num(wind_data.time[i]), vel_mean[i], '  ', rotation=rotation,
                bbox=dict(boxstyle='rarrow', fc=colors_w[i_color], ec=edge_colors_w[i_color]),
                fontsize=4)

ax3.set_xlim([wind_data.time[0], wind_data.time[-1]])
ax3.set_xticklabels([])
ax3.set_ylabel('Wind speed (m/s)')
ax3.set_ylim([0, 17.5])
add_subtitle(ax3, '(d) Wind speed and direction')

ylim3 = ax3.get_ylim()
ax3.fill_between(time_gw, ylim3[0], ylim3[1], where=l_wind, color=ocean_blue, alpha=0.3)
ax3.set_ylim(ylim3)
ax3.grid(True, linestyle='--', axis='x')

# (e) combined dswc conditions
ax5 = plt.subplot(6, 2, (9, 10))
ax5.fill_between(time_gw, 0, 1, where=l_dswc, color=ocean_blue)
ax5.set_ylim([0, 1])
ax5.set_xlim([time_gw[0], time_gw[-1]])
ax5.set_yticks([])
ax5.set_yticklabels([])
ax5.grid(True, linestyle='--', axis='x')
add_subtitle(ax5, '(e) Conditions allowing for dense shelf water outflows')

legend_elements = [Patch(facecolor=ocean_blue, edgecolor='k', label='Suitable'),
                   Patch(facecolor='w', edgecolor='k', label='Unsuitable')]
ax5.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(-0.02, 1.0))

# (f) % time dswc conditions
month_dswc = []
p_dswc_c = []
for n in range(time_gw[0].month, time_gw[-1].month+1):
    l_time = [t.month == n for t in time_gw]
    month_dswc.append(datetime(time_gw[0].year, n, 1))
    p_dswc_c.append(np.sum(l_dswc[l_time])/np.sum(l_time))
    
month_dswc = np.array(month_dswc)
str_month_dswc = np.array([t.strftime('%b') for t in month_dswc])
p_dswc_c = np.array(p_dswc_c)

width = []
for n in range(time_gw[0].month, time_gw[-1].month+1):
    t0 = datetime(time_gw[0].year, n, 1)
    t1 = datetime(time_gw[0].year, n+1, 1)
    width.append(0.8*(t1-t0).days)

ax6 = plt.subplot(6, 2, 11)
ax6.bar(month_dswc, p_dswc_c*100, color=ocean_blue, tick_label=str_month_dswc, width=width)
ax6.set_ylabel('Occurrence of\nconditions\n(% of time)')
ax6.set_ylim([0, 100])
add_subtitle(ax6, '(f) Occurrence of suitable conditions')

l6, b6, w6, h6 = ax6.get_position().bounds
ax6.set_position([l6, b6-0.02, w6, h6])

# (g) % time dswc (figure 5d)
csv_dswc = 'temp_data/fraction_cells_dswc_in_time.csv'
if not os.path.exists(csv_dswc):
    raise ValueError(f'''DSWC occurrence file does not yet exist: {csv_dswc}
                        Please create it first by running write_fraction_cells_dswc_in_time_to_csv (in dswc_detector.py)''')
df = pd.read_csv(csv_dswc)
time_dswc = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in df['time'].values]
f_dswc = df['f_dswc'].values
l_dswc_n = f_dswc >= 0.1

p_dswc = []
for n in range(time_dswc[0].month, time_dswc[-1].month):
    l_time = [t.month == n for t in time_dswc]
    p_dswc.append(np.sum(l_dswc_n[l_time])/np.sum(l_time))

p_dswc = np.array(p_dswc)
    
ax7 = plt.subplot(6, 2, 12)
ax7.bar(month_dswc, p_dswc*100, color=ocean_blue, tick_label=str_month_dswc, width=width)
ax7.set_ylabel('Occurrence of\noutflows\n(% of time)')
ax7.yaxis.set_label_position("right")
ax7.yaxis.tick_right()
ax7.set_ylim([0, 100])
add_subtitle(ax7, '(g) Occurrence of outflows')

l7, b7, w7, h7 = ax7.get_position().bounds
ax7.set_position([l7, b7-0.02, w7, h7])

log.info(f'Saving figure to: {output_dswc_conditions}')
plt.savefig(output_dswc_conditions, bbox_inches='tight', dpi=300)
plt.close()

# ---------------------------------------------------------------------------------
# REEF CONTRIBUTION ANALYSIS (ACCOMPANIES FIGURE 6A)
# ---------------------------------------------------------------------------------
# h_deep_sea = 200
# dx = 0.01
# location_info_p = get_location_info('perth')

# grid = DensityGrid(location_info_p.lon_range, location_info_p.lat_range, dx)
# x, y = np.meshgrid(grid.lon, grid.lat)

# particle_path = f'{get_dir_from_json("opendrift_output")}cwa_perth_MarSep2017_baseline.nc'
# particles = Particles.read_from_netcdf(particle_path)

# --- Components that make up Figure 6A ---
# output_reefs = 'figs10.jpg'

# fig = plt.figure(figsize=(12, 6))
# plt.subplots_adjust(wspace=0.7)

# # (a) kelp probability map
# kelp_prob = KelpProbability.read_from_tiff('input/perth_kelp_probability.tif')
# ax1 = plt.subplot(1, 4, 1, projection=ccrs.PlateCarree())
# ax1 = plot_basic_map(ax1, location_info_p)
# ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_p, ax=ax1, show=False, show_perth_canyon=False, color='k', linewidths=0.7)
# ax1, cbar1, c1 = kelp_prob.plot(location_info_p, ax=ax1, show=False)
# cbar1.remove()
# l1, b1, w1, h1 = ax1.get_position().bounds
# cbax1 = fig.add_axes([l1+w1+0.01, b1, 0.02, h1])
# cbar1 = plt.colorbar(c1, cax=cbax1)
# cbar1.set_label('Probability of kelp')
# add_subtitle(ax1, '(a) Kelp probability')

# # (b) release # particles
# density0 = get_particle_density(grid, particles.lon0, particles.lat0)
# density0[density0==0] = np.nan

# ax2 = plt.subplot(1, 4, 2, projection=ccrs.PlateCarree())
# ax2 = plot_basic_map(ax2, location_info_p)
# ax2 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_p,
#                     ax=ax2, show=False, show_perth_canyon=False,
#                     color='k', linewidths=0.7)

# c2 = ax2.pcolormesh(x, y, density0, cmap='plasma', vmin=0, vmax=150)
# ax2.set_yticklabels([])
# l2, b2, w2, h2 = ax2.get_position().bounds
# cbax2 = fig.add_axes([l2+w2+0.01, b2, 0.02, h2])
# cbar2 = plt.colorbar(c2, cax=cbax2)
# cbar2.set_label('# particles released')
# add_subtitle(ax2, '(b) Particle release')

# # (c) # particles past shelf from initial location
# l_deep_sea = particles.get_l_deep_sea(h_deep_sea)
# l_deep_sea_anytime = np.any(l_deep_sea, axis=1)

# density_ds0 = get_particle_density(grid, particles.lon0[l_deep_sea_anytime],
#                                    particles.lat0[l_deep_sea_anytime])
# density_ds0[density_ds0==0.] = np.nan

# ax3 = plt.subplot(1, 4, 3, projection=ccrs.PlateCarree())
# ax3 = plot_basic_map(ax3, location_info_p)
# ax3 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_p,
#                     ax=ax3, show=False, show_perth_canyon=False,
#                     color='k', linewidths=0.7)

# c3 = ax3.pcolormesh(x, y, density_ds0, cmap='plasma', vmin=0, vmax=150)
# ax3.set_yticklabels([])
# l3, b3, w3, h3 = ax3.get_position().bounds
# cbax3 = fig.add_axes([l3+w3+0.01, b3, 0.02, h3])
# cbar3 = plt.colorbar(c3, cax=cbax3)
# cbar3.set_label('# particles passing shelf')
# add_subtitle(ax3, '(c) Passing shelf')

# # (d) mean time to get past shelf
# t_release = particles.get_release_time_index()
# p_ds, t_ds = particles.get_indices_arriving_in_deep_sea(h_deep_sea)
# dt_ds = np.array([(particles.time[t_ds[i]]-particles.time[t_release[p_ds[i]]]).total_seconds()/(24*60*60) for i in range(len(p_ds))])

# density_time_ds = get_particle_density(grid, particles.lon0[p_ds], particles.lat0[p_ds],
#                                        values=dt_ds)
# density_mean_dt = density_time_ds/density_ds0
# density_mean_dt[density_mean_dt==0.] = np.nan

# ax4 = plt.subplot(1, 4, 4, projection=ccrs.PlateCarree())
# ax4 = plot_basic_map(ax4, location_info_p)
# ax4 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_p,
#                     ax=ax4, show=False, show_perth_canyon=False,
#                     color='k', linewidths=0.7)

# c4 = ax4.pcolormesh(x, y, density_mean_dt, cmap=cmocean.cm.deep, vmin=0, vmax=30)
# ax4.set_yticklabels([])
# l4, b4, w4, h4 = ax4.get_position().bounds
# cbax4 = fig.add_axes([l4+w4+0.01, b4, 0.02, h4])
# cbar4 = plt.colorbar(c4, cax=cbax4)
# cbar4.set_label('Mean time for particles to pass shelf (days)')
# add_subtitle(ax4, '(c) Mean time')

# log.info(f'Saving figure to: {output_reefs}')
# plt.savefig(output_reefs, bbox_inches='tight', dpi=300)
# plt.close()

# # --- Map with % making it to deep sea ---
# output_initial_ds = 'figs11.jpg'

# density0 = get_particle_density(grid, particles.lon0, particles.lat0)

# l_deep_sea = particles.get_l_deep_sea(h_deep_sea)
# l_deep_sea_anytime = np.any(l_deep_sea, axis=1)

# density_ds0 = get_particle_density(grid, particles.lon0[l_deep_sea_anytime],
#                                    particles.lat0[l_deep_sea_anytime])

# density = density_ds0/density0*100
# density[density==0.] = np.nan

# fig = plt.figure(figsize=(6, 8))
# ax = plt.axes(projection=ccrs.PlateCarree())
# ax = plot_basic_map(ax, location_info_p)
# ax = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_p,
#                    ax=ax, show=False, show_perth_canyon=False,
#                    color='k', linewidths=0.7)

# c = ax.pcolormesh(x, y, density, cmap='plasma', vmin=0, vmax=100)
# l, b, w, h = ax.get_position().bounds
# cbax = fig.add_axes([l+w+0.03, b, 0.05, h])
# cbar = plt.colorbar(c, cax=cbax)
# cbar.set_label('Particles making it to deep sea (%)')
# ax.set_title('Export per initial location')

# log.info(f'Saving figure to: {output_initial_ds}')
# plt.savefig(output_initial_ds, bbox_inches='tight', dpi=300)
# plt.close()

# # --- Minimum and maximum example trajectories ---
# output_tracks = 'figs12.jpg'

# t_release = particles.get_release_time_index()
# p_ds, t_ds = particles.get_indices_arriving_in_deep_sea(h_deep_sea)
# dt_ds = np.array([(particles.time[t_ds[i]]-particles.time[t_release[p_ds[i]]]).total_seconds()/(24*60*60) for i in range(len(p_ds))])

# lon_examples = np.array([115.50, 115.31, 115.65, 115.64, 115.30, 115.56])
# lat_examples = np.array([-31.90, -31.73, -31.78, -32.43, -32.38, -32.17])

# i_ex, j_ex = grid.get_index(lon_examples, lat_examples)
# lon0 = particles.lon0[p_ds]
# lat0 = particles.lat0[p_ds]
# p_ex_min = []
# p_ex_max = []
# for i in range(len(lon_examples)):
#     l_lon = np.logical_and(lon0 >= grid.lon[i_ex[i].astype(int)], lon0 <= grid.lon[i_ex[i].astype(int)+1])
#     l_lat = np.logical_and(lat0 >= grid.lat[j_ex[i].astype(int)], lat0 <= grid.lat[j_ex[i].astype(int)+1])
#     ps_ex = np.where(np.logical_and(l_lon, l_lat))[0]
#     i_sort = np.argsort(dt_ds[ps_ex]) # sort time ascending
#     i_min = i_sort[0]
#     i_max = i_sort[-1]
#     p_ex_min.append(ps_ex[i_min])
#     p_ex_max.append(ps_ex[i_max])

# int_t = 8 # plots daily dots on tracks

# fig = plt.figure(figsize=(8, 6))
# plt.subplots_adjust(wspace=0.1)

# location_info_w = get_location_info('perth_wide_south')

# # (a) Minimum example tracks
# ax1 = plt.subplot(1, 2, 1, projection=ccrs.PlateCarree())
# ax1 = plot_basic_map(ax1, location_info_w)
# ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_w,
#                     ax=ax1, show=False, show_perth_canyon=False,
#                     color='k', linewidths=0.7)

# lon = particles.lon[p_ds, :]
# lat = particles.lat[p_ds, :]
# # cm = mpl.colormaps['summer']
# colors = ['#2e2d4d', '#5d8888', '#c88066', '#4f5478', '#ebc08b', '#c15251']
# for i in range(len(p_ex_min)):
#     # color = cm(i/(len(p_ex)-1))
#     color = colors[i]
#     ax1.plot(lon[p_ex_min[i], :t_ds[p_ex_min[i]]], lat[p_ex_min[i], :t_ds[p_ex_min[i]]], '-', color=color)
#     ax1.plot(lon[p_ex_min[i], :t_ds[p_ex_min[i]]:int_t], lat[p_ex_min[i], :t_ds[p_ex_min[i]]:int_t], '.', color=color)
#     ax1.plot(lon0[p_ex_min[i]], lat0[p_ex_min[i]], 'xk')
#     ax1.plot(lon[p_ex_min[i], t_ds[p_ex_min[i]]], lat[p_ex_min[i], t_ds[p_ex_min[i]]], 'ok')
    
# add_subtitle(ax1, '(a) Shortest particle trajectories')

# # (b) Maximum example tracks
# ax2 = plt.subplot(1, 2, 2, projection=ccrs.PlateCarree())
# ax2 = plot_basic_map(ax2, location_info_w)
# ax2 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_w,
#                     ax=ax2, show=False, show_perth_canyon=False,
#                     color='k', linewidths=0.7)
# ax2.set_yticklabels([])

# lon = particles.lon[p_ds, :]
# lat = particles.lat[p_ds, :]
# # cm = mpl.colormaps['summer']
# colors = ['#2e2d4d', '#5d8888', '#c88066', '#4f5478', '#ebc08b', '#c15251']
# for i in range(len(p_ex_max)):
#     # color = cm(i/(len(p_ex)-1))
#     color = colors[i]
#     ax2.plot(lon[p_ex_max[i], :t_ds[p_ex_max[i]]], lat[p_ex_max[i], :t_ds[p_ex_max[i]]], '-', color=color)
#     ax2.plot(lon[p_ex_max[i], :t_ds[p_ex_max[i]]:int_t], lat[p_ex_max[i], :t_ds[p_ex_max[i]]:int_t], '.', color=color)
#     ax2.plot(lon0[p_ex_max[i]], lat0[p_ex_max[i]], 'xk')
#     ax2.plot(lon[p_ex_max[i], t_ds[p_ex_max[i]]], lat[p_ex_max[i], t_ds[p_ex_max[i]]], 'ok')
    
# add_subtitle(ax2, '(b) Longest particle trajectories')

# log.info(f'Saving figure to: {output_tracks}')
# plt.savefig(output_tracks, bbox_inches='tight', dpi=300)
# plt.close()