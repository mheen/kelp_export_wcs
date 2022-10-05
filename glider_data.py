from netCDF4 import Dataset
import numpy as np
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
from cartopy.io import shapereader
import cartopy.feature as cftr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import sys
sys.path.append('..')
from py_tools.timeseries import convert_time_to_datetime, get_l_time_range
from py_tools.files import get_dir_from_json

class GliderData:
    def __init__(self, time:np.ndarray,
                 lon:np.ndarray,
                 lat:np.ndarray,
                 depth:np.ndarray,
                 temp:np.ndarray,
                 salt:np.ndarray,
                 ox2:np.ndarray,
                 cphl:np.ndarray,
                 u:np.ndarray,
                 v:np.ndarray):

        self.time = time
        self.lon = lon
        self.lat = lat
        self.depth = depth
        self.temp = temp
        self.salt = salt
        self.ox2 = ox2
        self.cphl = cphl
        self.u = u
        self.v = v

    def plot_glider_track(self, ax=None, show=True, show_labels=True):
        if ax is None:
            ax = plt.axes(projection=ccrs.PlateCarree())
            shp = shapereader.Reader('input/GSHHS_coastline_GSR.shp')
            for record, geometry in zip(shp.records(), shp.geometries()):
                ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor='lightgray',
                                edgecolor='black')
            ax.set_extent([np.floor(np.nanmin(self.lon)), np.ceil(np.nanmax(self.lon)),
                           np.floor(np.nanmin(self.lat)), np.ceil(np.nanmax(self.lat))],
                           ccrs.PlateCarree())
        
        l_nonans_position = np.logical_and(~np.isnan(self.lon), ~np.isnan(self.lat))
        ax.plot(self.lon[l_nonans_position][0], self.lat[l_nonans_position][0],
                'or', markersize=5, label='Initial location',
                transform=ccrs.PlateCarree(), zorder=5)
        ax.plot(self.lon[l_nonans_position][-1], self.lat[l_nonans_position][-1],
                'xr', markersize=5, label='Final location',
                transform=ccrs.PlateCarree(), zorder=6)
        ax.plot(self.lon, self.lat, '.k', transform=ccrs.PlateCarree())
        
        if show_labels is True:
            # label locations with date
            time_dates = np.array([t.date() for t in self.time])
            n_days = (time_dates[-1]-time_dates[0]).days
            i_label = []
            for i in range(n_days+1):
                i_date = np.where(np.logical_and(time_dates==time_dates[0]+timedelta(days=i), l_nonans_position))[0]
                if any(i_date):
                    i_label.append(i_date[0])
            time_labels = [t.strftime('%d-%m-%Y %H:%M') for t in self.time[i_label]]
            for i in range(len(i_label)):
                ax.text(self.lon[i_label[i]], self.lat[i_label[i]], time_labels[i], transform=ccrs.PlateCarree())

        if show is True:
            ax.legend(loc='upper right')
            plt.show()
        else:
            return ax

    def get_data_in_time_frame(self, start_time:datetime, end_time:datetime):
        l_time = get_l_time_range(self.time, start_time, end_time)
        return GliderData(self.time[l_time], self.lon[l_time], self.lat[l_time],
                          self.depth[l_time], self.temp[l_time], self.salt[l_time],
                          self.ox2[l_time], self.cphl[l_time], self.u[l_time], self.v[l_time])


    @staticmethod
    def read_from_netcdf(input_path:str, use_qc_flags=[1, 2]) -> tuple:
        # IMOS standard quality control flags:
        # 0: no qc performed
        # 1: good data
        # 2: probably good data
        # 3: bad data that are potentially correctible
        # 4: bad data
        # 5: value changed
        # 6: not used
        # 7: interpolated values
        # 8: missing values
        
        nc = Dataset(input_path)

        # time
        time = nc['TIME'][:].filled(fill_value=np.nan)
        time_qc = nc['TIME_quality_control'][:].filled()
        l_time = sum([time_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        time[~l_time] = np.nan
        time_datetime = convert_time_to_datetime(time, nc['TIME'].units)

        # coordinates
        lon = nc['LONGITUDE'][:].filled(fill_value=np.nan)
        lon_qc = nc['LONGITUDE_quality_control'][:].filled()
        l_lon = sum([lon_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        lon[~l_lon] = np.nan

        lat = nc['LATITUDE'][:].filled(fill_value=np.nan)
        lat_qc = nc['LATITUDE_quality_control'][:].filled()
        l_lat = sum([lat_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        lat[~l_lat] = np.nan

        depth = nc['DEPTH'][:].filled(fill_value=np.nan)
        depth_qc = nc['DEPTH_quality_control'][:].filled()
        l_depth = sum([depth_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        depth[~l_depth] = np.nan

        # temperature
        temp = nc['TEMP'][:].filled(fill_value=np.nan)
        temp_qc = nc['TEMP_quality_control'][:].filled()
        l_temp = sum([temp_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        temp[~l_temp] = np.nan

        # salinity
        salt = nc['PSAL'][:].filled(fill_value=np.nan)
        salt_qc = nc['PSAL_quality_control'][:].filled()
        l_salt = sum([salt_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        salt[~l_salt] = np.nan

        # oxygen
        ox2 = nc['DOX2'][:].filled(fill_value=np.nan)
        ox2_qc = nc['DOX2_quality_control'][:].filled()
        l_ox2 = sum([ox2_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        ox2[~l_ox2] = np.nan

        # chlorophyll
        cphl = nc['CPHL'][:].filled(fill_value=np.nan)
        cphl_qc = nc['CPHL_quality_control'][:].filled()
        l_cphl = sum([cphl_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        cphl[~l_cphl] = np.nan

        # current velocities
        u = nc['UCUR'][:].filled(fill_value=np.nan)
        u_qc = nc['UCUR_quality_control'][:].filled()
        l_u = sum([u_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        u[~l_u] = np.nan

        v = nc['VCUR'][:].filled(fill_value=np.nan)
        v_qc = nc['VCUR_quality_control'][:].filled()
        l_v = sum([v_qc==use_qc_flag for use_qc_flag in use_qc_flags]).astype(bool)
        v[~l_v] = np.nan

        nc.close()

        return GliderData(time_datetime, lon, lat, depth, temp, salt, ox2, cphl, u, v)

if __name__ == '__main__':
    glider_data = GliderData.read_from_netcdf(f'{get_dir_from_json("input/dirs.json", "glider_data")}IMOS_ANFOG_BCEOPSTUV_20220628T064224Z_SL286_FV01_timeseries_END-20220712T082641Z.nc')
    glider_data_subset = glider_data.get_data_in_time_frame(datetime(2022, 6, 30, 22, 30), datetime(2022, 7, 2, 15))
    glider_data_subset.plot_glider_track()
