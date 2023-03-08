from data.roms_data import get_roms_data_for_transect, get_depth_integrated_gradient_along_transect, write_depth_integrated_gradient_data_to_netcdf
from tools.coordinates import get_transect_lons_lats_ds_from_json
from tools.files import get_dir_from_json
from datetime import datetime

# --- write ROMS density gradient data to netcdf ---
roms_input_dir = f'{get_dir_from_json("roms_data")}2017/'
start_date = datetime(2017, 3, 1)
end_date = datetime(2017, 8, 1)
lon1, lat1, lon2, lat2, ds = get_transect_lons_lats_ds_from_json('two_rocks_glider')
roms_data = get_roms_data_for_transect(roms_input_dir, start_date, end_date, lon1, lat1, lon2, lat2)
density_gradient, density, distance, z, time = get_depth_integrated_gradient_along_transect(roms_data,
                                                                                            'density',
                                                                                            lon1, lat1,
                                                                                            lon2, lat2,
                                                                                            ds)
time_str = f'{start_date.strftime("%b")}-{end_date.strftime("%b%Y")}'
output_path = f'{get_dir_from_json("processed_data")}roms_transects/TR_density_{time_str}.nc'
write_depth_integrated_gradient_data_to_netcdf(output_path, time, density_gradient, density, distance, z)