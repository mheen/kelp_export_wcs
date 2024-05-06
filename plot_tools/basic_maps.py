import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from location_info import LocationInfo, get_location_info
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.io import shapereader
import matplotlib.pyplot as plt

def add_grid(ax:plt.axes, meridians:list, parallels:list,
              xmarkers:str, ymarkers:str, draw_grid:bool) -> plt.axes:

    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.set_xticks(meridians, crs=ccrs.PlateCarree())
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.set_yticks(parallels, crs=ccrs.PlateCarree())

    if xmarkers == 'top':
        ax.xaxis.tick_top()
    if xmarkers == 'off':
        ax.set_xticklabels([])
    if ymarkers == 'right':
        ax.yaxis.tick_right()
    if ymarkers == 'off':
        ax.set_yticklabels([])

    if draw_grid is True:
        ax.grid(b=True, linewidth=0.5, color='k', linestyle=':', zorder=10)

    return ax

def plot_basic_map(ax:plt.axes, location_info:LocationInfo,
                   xmarkers='bottom', ymarkers='left',
                   draw_grid=False, zorder_c=5, facecolor='#989898') -> plt.axes:
    shp = shapereader.Reader('input/GSHHS_coastline_GSR.shp')
    for _, geometry in zip(shp.records(), shp.geometries()):
        ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor=facecolor,
                           edgecolor='black', zorder=zorder_c)
        
    shp2 = shapereader.Reader('input/GI.shp')
    for _, geometry in zip(shp2.records(), shp2.geometries()):
        ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor=facecolor,
                          edgecolor='black', zorder=100)
    
    ax = add_grid(ax, location_info.meridians, location_info.parallels, xmarkers, ymarkers, draw_grid)

    ax.set_extent([location_info.lon_range[0], location_info.lon_range[1],
                   location_info.lat_range[0], location_info.lat_range[1]],
                   ccrs.PlateCarree())
    
    return ax
