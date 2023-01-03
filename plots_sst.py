from satellite_data import SatelliteSST, read_satellite_sst_from_netcdf, get_monthly_mean_sst
from tools.files import get_dir_from_json
from tools import log
from location_info import LocationInfo, get_location_info
from basic_maps import plot_basic_map
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

def plot_sst(sst_data:SatelliteSST, location_info:LocationInfo, t=0,
             ax=None, show=True, output_path=None, title='',
             cmap='viridis', clabel='Sea surface temperature ($^o$C)',
             vmin=None, vmax=None) -> plt.axes:

    if ax is None:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax = plot_basic_map(ax, location_info)

    x, y = np.meshgrid(sst_data.lon, sst_data.lat)
    if len(sst_data.sst.shape) == 3:
        sst = sst_data.sst[t, :, :]
    elif len(sst_data.sst.shape) == 2:
        sst = sst_data.sst
    else:
        raise ValueError('Unknown SST data dimensions')
    c = ax.pcolormesh(x, y, sst, cmap=cmap, vmin=vmin, vmax=vmax)
    
    ax.set_title(title)

    # colorbar
    l,b,w,h = ax.get_position().bounds
    fig = plt.gcf()
    cbax = fig.add_axes([l+w+0.02, b, 0.05*w, h])
    cbar = plt.colorbar(c, cax=cbax)
    cbar.set_label(clabel)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

if __name__ == '__main__':
    location_info = get_location_info('gsr')

    input_path = f'{get_dir_from_json("satellite_sst")}gsr_monthly_mean_June.nc'
    sst_data = read_satellite_sst_from_netcdf(input_path)

    plot_sst(sst_data, location_info, title='June monthly mean SST', vmin=18, vmax=25)
    