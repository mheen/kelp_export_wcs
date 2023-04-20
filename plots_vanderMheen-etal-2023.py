from tools import log
from data.kelp_data import KelpProbability
from data.roms_data import read_roms_grid_from_netcdf
from plot_tools.basic_maps import plot_basic_map
from plot_tools.general import add_subtitle
from plot_tools.plots_bathymetry import plot_contours
from location_info import LocationInfo, get_location_info
from datetime import datetime, date
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import cartopy.crs as ccrs
import numpy as np
import cmocean

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)

kelp_green = '#1b7931'
ocean_blue = '#25419e'

roms_grid = read_roms_grid_from_netcdf('input/cwa_roms_grid.nc')

def figure1(show=True, output_path=None):

    fig = plt.figure(figsize=(12, 8))
    plt.subplots_adjust(wspace=0.35)

    location_info_p = get_location_info('perth')
    kelp_prob = KelpProbability.read_from_tiff('input/perth_kelp_probability.tif')
    ax1 = plt.subplot(3, 3, (1, 4), projection=ccrs.PlateCarree())
    ax1 = plot_basic_map(ax1, location_info_p)
    ax1 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_p, ax=ax1, show=False, show_perth_canyon=False, color='k', linewidths=0.7)
    ax1, cbar1, c1 = kelp_prob.plot(location_info_p, ax=ax1, show=False)
    cbar1.remove()
    l1, b1, w1, h1 = ax1.get_position().bounds
    cbax1 = fig.add_axes([l1+w1+0.01, b1, 0.02, h1])
    cbar1 = plt.colorbar(c1, cax=cbax1)
    cbar1.set_label('Probability of kelp')
    add_subtitle(ax1, '(a) Perth kelp reefs')

    location_info_pw = get_location_info('perth_wide')
    ax3 = plt.subplot(3, 3, (2, 6), projection=ccrs.PlateCarree())
    ax3 = plot_basic_map(ax3, location_info_pw, ymarkers='off')
    ax3 = plot_contours(roms_grid.lon, roms_grid.lat, roms_grid.h, location_info_pw, ax=ax3, show=False, show_perth_canyon=True, color='k', linewidths=0.7)
    c3 = ax3.pcolormesh(roms_grid.lon, roms_grid.lat, roms_grid.h, vmin=0, vmax=4000, cmap=cmocean.cm.deep)
    l3, b3, w3, h3 = ax3.get_position().bounds
    cbax3 = fig.add_axes([l3+w3+0.01, b3, 0.02, h3])
    cbar3 = plt.colorbar(c3, cax=cbax3)
    cbar3.set_label('Bathymetry (m)')
    add_subtitle(ax3, '(c) Main oceanographic features')
    # ax3.set_position([l3+0.02, b3+0.05, w3, h3])

    # detritus production from de Bettignies et al. (2013)
    time_detritus = ['Mar-May', 'Jun-Jul', 'Sep-Oct', 'Dec-Feb']
    detritus = [4.8, 2.12, 0.70, 0.97]
    detritus_sd = [1.69, 0.84, 0.45, 0.81]

    ax2 = plt.subplot(3, 3, 7)
    ax2.bar(np.arange(len(time_detritus)), detritus, color=kelp_green, tick_label=time_detritus, yerr=detritus_sd)
    ax2.set_ylim([0, 7.5])
    ax2.set_ylabel('Detritus production\n(g/kelp/day)')
    add_subtitle(ax2, '(b) Seasonal detritus production')
    l2, b2, w2, h2 = ax2.get_position().bounds
    ax2.set_position([l1, b2, w1+0.03, h2])



    if show is True:
        plt.show()

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    plt.close()

if __name__ == '__main__':
    figure1()