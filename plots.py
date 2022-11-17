from basic_maps import perth_map
from bathymetry_data import BathymetryData
from kelp_map import KelpProbability
from glider_data import GliderData
from roms_data import RomsData
from particles import Particles
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.append('..')
from py_tools.files import get_dir_from_json
from py_tools import log

def location_overview(show=True, output_path=None):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = perth_map(ax)

    kelp_prob = KelpProbability.read_from_tiff()
    ax = kelp_prob.plot(ax=ax, show=False)

    bathymetry = BathymetryData.read_from_netcdf()
    ax = bathymetry.plot_contours(ax=ax, show=False)

    if show is True:
        plt.show()
    
    if output_path:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

def event_map(parameter='temp', clabel='Sea Surface Temperature ($^o$C)', s=-1,
              vmin=18, vmax=22, cmap='RdBu_r', show=True, output_path=None):
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = perth_map(ax)

    roms = RomsData.read_from_netcdf(f'{get_dir_from_json("input/dirs.json", "roms_data")}2022/perth_his_20220701.nc')
    ax = roms.plot_map(parameter, 0, s, clabel=clabel, vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, show=False)

    bathymetry = BathymetryData.read_from_netcdf()
    ax = bathymetry.plot_contours(ax=ax, show=False)

    glider = GliderData.read_from_netcdf(f'{get_dir_from_json("input/dirs.json", "glider_data")}IMOS_ANFOG_BCEOPSTUV_20220628T064224Z_SL286_FV01_timeseries_END-20220712T082641Z.nc')
    glider_subset = glider.get_data_in_time_frame(datetime(2022, 6, 30, 22, 30), datetime(2022, 7, 2, 15))
    ax = glider_subset.plot_track(ax=ax, show=False, show_labels=False, color='#808080')

    ax.set_title(roms.time[0].strftime('%d-%m-%Y'))

    if show is True:
        plt.show()

    if output_path:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

def particles_in_deep_sea_per_depth(p:Particles, h_deep_sea:int,
                                    depths=[5, 10, 20, 30, 40, 50],
                                    linestyles=['.', ':', '--', '-.', '-'],
                                    colors=['k', 'k', 'k', 'k', 'k'],
                                    labels=['10 m', '20 m', '30 m', '40 m', '50 m'],
                                    use_overal_total=True,
                                    show=True, output_path=None):
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes()

    if use_overal_total is True:
        total_particles = p.lon.shape[0]
    else:
        total_particles = None
    ax = p.plot_age_in_deep_sea(h_deep_sea, linestyle='-', color='g', label='Total', ax=ax, show=False)
    for i in range(len(depths)-1):
        p_depth = p.filter_based_on_release_depth(depths[i], depths[i+1])
        ax = p_depth.plot_age_in_deep_sea(h_deep_sea, total_particles=total_particles,
                                          linestyle=linestyles[i], color=colors[i], label=labels[i],
                                          ax=ax, show=False)
    
    ax.legend(loc='upper left')
    ax.set_xlabel('Particle age (days)')
    ax.set_ylabel('Fraction in deep sea')

    if show is True:
        plt.show()

    if output_path:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

if __name__ == '__main__':
    # location_overview(show=False, output_path=f'{get_dir_from_json("input/dirs.json", "plots")}perth_location_overview.jpg')

    # event_map(show=False, output_path=f'{get_dir_from_json("input/dirs.json", "plots")}event-072022_overview_sst.jpg')
    # event_map(parameter='velocity', clabel='Bottom velocity (m/s)', s=0, vmin=0, vmax=0.8, cmap='viridis',
    #           show=False, output_path=f'{get_dir_from_json("input/dirs.json", "plots")}event-072022_overview_bottomvel.jpg')

    # glider = GliderData.read_from_netcdf(f'{get_dir_from_json("input/dirs.json", "glider_data")}IMOS_ANFOG_BCEOPSTUV_20220628T064224Z_SL286_FV01_timeseries_END-20220712T082641Z.nc')
    # glider_subset = glider.get_data_in_time_frame(datetime(2022, 6, 30, 22, 30), datetime(2022, 7, 2, 15))
    # ax = glider_subset.plot_transect(show=False)
    # ax.set_title(f'Glider transect {glider_subset.time[0].strftime("%d-%b %H:%M")} to {glider_subset.time[-1].strftime("%-d-%b %H:%M")}')
    # plt.savefig(f'{get_dir_from_json("input/dirs.json", "plots")}event-072022_glider-transect.jpg', bbox_inches='tight', dpi=300)

    file_name = 'perth_event-2022-06-28-2022-07-05'
    h_deep_sea = 150
    p = Particles.read_from_netcdf(f'{get_dir_from_json("input/dirs.json", "opendrift")}{file_name}.nc')
    particles_in_deep_sea_per_depth(p, h_deep_sea, use_overal_total=True,
                                    show=False,
                                    output_path=f'{get_dir_from_json("input/dirs.json", "plots")}{file_name}_age_deep_sea_depth.jpg')
    