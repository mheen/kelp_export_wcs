from roms_data import RomsGrid
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.io import shapereader
import matplotlib.pyplot as plt

perth_grid = RomsGrid.read_from_netcdf()
lon_range_perth, lat_range_perth = perth_grid.get_lon_lat_range()
perth_meridians = [115.3, 115.7]
perth_parallels = [-32.4, -32.0, -31.6]

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
        ax.set_yticklabels([])
    if ymarkers == 'right':
        ax.yaxis.tick_right()
    if ymarkers == 'off':
        ax.set_yticklabels([])

    if draw_grid is True:
        ax.grid(b=True, linewidth=0.5, color='k', linestyle=':', zorder=10)

    return ax

def perth_map(ax:plt.axes, xmarkers='bottom', ymarkers='left', draw_grid=False) -> plt.axes:
    shp = shapereader.Reader('input/GSHHS_coastline_GSR.shp')
    for _, geometry in zip(shp.records(), shp.geometries()):
        ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray',
                          edgecolor='black', zorder=5)

    ax = add_grid(ax, perth_meridians, perth_parallels, xmarkers, ymarkers, draw_grid)

    ax.set_extent([lon_range_perth[0], lon_range_perth[1],
                   lat_range_perth[0], lat_range_perth[1]],
                   ccrs.PlateCarree())

    return ax