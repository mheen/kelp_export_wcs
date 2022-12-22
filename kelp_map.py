from basic_maps import perth_map
import rasterio
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from py_tools import log

class KelpProbability:
    def __init__(self, lon:np.ndarray,
                 lat:np.ndarray,
                 probability:np.ndarray):
        self.lon = lon
        self.lat = lat
        self.prob = probability

    def plot(self, ax=None, show=True, output_path=None, probability_threshold=0.) -> plt.axes:
        if ax is None:
            ax = plt.axes(projection=ccrs.PlateCarree())
            ax = perth_map(ax)

        l_prob = self.prob >= probability_threshold
        prob = np.copy(self.prob)
        prob[~l_prob] = np.nan

        c = ax.pcolormesh(self.lon, self.lat, prob, cmap='Greens')
        cbar = plt.colorbar(c)
        cbar.set_label('Probability of kelp')

        if output_path:
            log.info(f'Saving figure to: {output_path}')
            plt.savefig(output_path, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()
        else:
            return ax

    @staticmethod
    def read_from_tiff(input_path='input/perth_kelp_probability.tif') -> tuple:
        log.info(f'Reading kelp probability map from: {input_path}')

        dataset = rasterio.open(input_path)

        prob = dataset.read()[0]
        prob[prob<=0.] = np.nan

        height = prob.shape[0]
        width = prob.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(dataset.transform, rows, cols)
        lon = np.array(xs)
        lat = np.array(ys)

        return KelpProbability(lon, lat, prob)

def generate_random_releases_based_on_probability(rng: np.random.default_rng, n_thin=25, input_path='input/perth_kelp_probability.tif') -> tuple:
    log.info(f'''Getting kelp release locations based on probability
             from probability map: {input_path}''')

    kelp_prob = KelpProbability.read_from_tiff(input_path)
    # random release depending on kelp probability
    random_ints_prob = rng.integers(1, high=100, size=kelp_prob.prob.shape)
    l_prob = random_ints_prob >= kelp_prob.prob
    # randomly thin releases based on n_thin (to keep number of particles manageable)
    random_ints = rng.integers(1, high=n_thin+1, size=kelp_prob.prob.shape)
    l_thin = random_ints == n_thin
    l_release = np.logical_and(l_prob, l_thin)

    return kelp_prob.lon[l_release], kelp_prob.lat[l_release]

def get_kelp_coordinates(probability_threshold=0.8,
                         input_path='input/perth_kelp_probability.tif') -> tuple:
    log.info(f'''Getting kelp coordinates using probability threshold {probability_threshold},
             from probability map: {input_path}''')

    dataset = rasterio.open(input_path)

    prob = dataset.read()[0] # read all data from first band
    i_kelp = np.where(prob>=probability_threshold)

    lon, lat = rasterio.transform.xy(dataset.transform, i_kelp[0], i_kelp[1], offset='center')

    return np.array(lon), np.array(lat)

if __name__ == '__main__':
    kelp_prob = KelpProbability.read_from_tiff()