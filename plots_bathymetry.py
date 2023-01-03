from tools.files import get_dir_from_json
from tools import log
from location_info import LocationInfo, get_location_info
from basic_maps import plot_basic_map
from bathymetry_data import BathymetryData
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

def plot_contours(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                  location_info:LocationInfo, ax=None, show=True, fontsize=10,
                  color='k', highlight_contour=[600], show_perth_canyon=True) -> plt.axes:

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)
        
    def _fmt(x):
        s = f'{x:.0f}'
        return s

    # get masked arrays so contour labels fall within plot if range is limited
    mask = np.ones(h.shape).astype(bool)
    mask = mask & (lon>=location_info.lon_range[0]) & (lon<=location_info.lon_range[1])
    mask = mask & (lat>=location_info.lat_range[0]) & (lat<=location_info.lat_range[1])
    xm = np.ma.masked_where(~mask, lon)
    ym = np.ma.masked_where(~mask, lat)
    zm = np.ma.masked_where(~mask, h)

    # remove depth to highlight from contour levels
    h_pc = [1000]
    if show_perth_canyon is True and highlight_contour is not None:
        cs_remove = highlight_contour+h_pc
    elif show_perth_canyon is False and highlight_contour is not None:
        cs_remove = highlight_contour
    elif show_perth_canyon is True and highlight_contour is None:
        cs_remove = h_pc
    else:
        cs_remove = []
    contour_levels = [cl for cl in location_info.contour_levels if cl not in cs_remove]

    cs = ax.contour(xm, ym, zm, levels=contour_levels,
                    colors=color, linewidths=1, transform=ccrs.PlateCarree())
    ax.clabel(cs, cs.levels, fontsize=fontsize, inline=True, fmt=_fmt)

    if highlight_contour is not None:
        cs_hl = ax.contour(xm, ym, zm, highlight_contour,
                        colors=color, linewidths=2.0, transform=ccrs.PlateCarree())
        ax.clabel(cs_hl, cs_hl.levels, fontsize=fontsize, inline=True, fmt=_fmt)

    if show_perth_canyon is True:
        cs_pc = ax.contour(xm, ym, zm, [1000],
                        colors=color, linewidths=1.5, linestyles='--', transform=ccrs.PlateCarree())
        ax.clabel(cs_pc, cs_pc.levels, fontsize=fontsize, inline=True, fmt=_fmt)
        t = ax.text(114.93, -32.10, 'Perth canyon', va='center', ha='right', transform=ccrs.PlateCarree())
        t.set_bbox(dict(facecolor='w', alpha=0.8, edgecolor='w'))

    if show is True:
        plt.show()
    else:
        return ax

def plot_bathymetry(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                    location_info:LocationInfo, ax=None, show=True,
                    cmap='BrBG', vmin=None, vmax=None) -> plt.axes:

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)
    
    c = ax.pcolormesh(lon, lat, h, cmap=cmap, vmin=vmin, vmax=vmax)

    l,b,w,h = ax.get_position().bounds
    fig = plt.gcf()
    cbax = fig.add_axes([l+w+0.02, b, 0.05*w, h])
    cbar = plt.colorbar(c, cax=cbax)
    cbar.set_label('Bathymetry (m)')

    if show is True:
        plt.show()
    else:
        return ax

def plot_bathymetry_and_contours(lon:np.ndarray, lat:np.ndarray, h:np.ndarray,
                                 location_info:LocationInfo, ax=None,
                                 show=True, output_path=None, contour_fontsize=12,
                                 color='k', highlight_contour=[600], show_perth_canyon=True,
                                 cmap='BrBG', vmin=None, vmax=None):

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax = plot_basic_map(ax, location_info)

    ax = plot_bathymetry(lon, lat, h, location_info, ax=ax, show=False,
                         cmap=cmap, vmin=vmin, vmax=vmax)
    ax = plot_contours(lon, lat, h, location_info, ax=ax, show=False, color=color, fontsize=contour_fontsize,
                       highlight_contour=highlight_contour, show_perth_canyon=show_perth_canyon)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

if __name__ == '__main__':
    location_info = get_location_info('cwa_perth_zoom')
    bathymetry = BathymetryData.read_from_netcdf('input/cwa_roms_grid.nc')
    
    plt.rcParams.update({'font.size' : 20})
    output_path = f'{get_dir_from_json("plots")}cwa-perth-zoom_bathymetry.jpg'
    _ = plot_bathymetry_and_contours(bathymetry.lon, bathymetry.lat, bathymetry.h, location_info,
                                     show=False, output_path=output_path, vmin=0, vmax=4000,
                                     contour_fontsize=17, cmap='viridis_r')