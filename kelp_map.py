import rasterio
import numpy as np

def get_kelp_coordinates(probability_threshold=0.8,
                         input_path='input/perth_kelp_probability.tif') -> tuple:
    dataset = rasterio.open(input_path)

    prob = dataset.read()[0] # read all data from first band
    i_kelp = np.where(prob>=0.8)

    lon, lat = rasterio.transform.xy(dataset.transform, i_kelp[0], i_kelp[1], offset='center')

    return np.array(lon), np.array(lat)
    