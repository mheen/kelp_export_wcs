import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json
from tools.timeseries import convert_time_to_datetime
import pandas as pd
from netCDF4 import Dataset
from datetime import datetime, timedelta
import numpy as np

# MEI: Multivariate El Nino Southern Oscillation Index (from NOAA https://www.psl.noaa.gov/enso/mei)
def read_mei_data(input_path='input/MEI-v2_2023.csv', skiprows=3) -> tuple:
    df = pd.read_csv(input_path, skiprows=skiprows)
    month_strs = ['DJ', 'JF', 'FM', 'MA', 'AM', 'MJ', 'JJ', 'JA', 'AS', 'SO', 'ON', 'ND']

    time = []
    mei = []

    for year in df['YEAR'].values:
        df_year = df.loc[df['YEAR']==year]
        for m in range(1, 13):
            time.append(datetime(year, m, 1))
            mei.append(df_year[month_strs[m-1]].values[0])

    return np.array(time), np.array(mei)

# DMI: Dipole Mode Index, indicator for the Indian Ocean Dipole (from NOAA/ESRL: https://stateoftheocean.osmc.noaa.gov/sur/ind/dmi.php)
def read_dmi_data(input_path='input/dmi_2023.nc') -> tuple:
    nc = Dataset(input_path)

    time_org = nc['WEDCEN2'][:].filled(fill_value=np.nan)
    time_units = nc['WEDCEN2'].units
    time = convert_time_to_datetime(time_org, time_units)

    dmi = nc['DMI'][:].filled(fill_value=np.nan)

    nc.close()

    return time, dmi
