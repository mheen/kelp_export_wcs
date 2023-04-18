import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools import log
from tools.timeseries import add_month_to_time
from tools.files import get_dir_from_json
from data.climate_data import read_dmi_data, read_mei_data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.units as munits
import matplotlib.ticker as tck
import numpy as np
from datetime import date, datetime

converter = mdates.ConciseDateConverter()
munits.registry[np.datetime64] = converter
munits.registry[date] = converter
munits.registry[datetime] = converter

locator = mdates.AutoDateLocator(minticks=5, maxticks=15)
formatter = mdates.ConciseDateFormatter(locator)

def plot_mei_index(time:np.ndarray, mei:np.ndarray, ax=None,
                   color_en='#900C3F', color_ln='#1e1677', xlim=None,
                   show=True, output_path=None) -> plt.axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()

    mask_en = mei>0
    mask_ln = mei<0

    if xlim is None:
        xlim = [time[0], time[-1]]

    days_in_month = np.array([(add_month_to_time(t, 1)-t).days for t in time])

    ax.bar(time[mask_en], mei[mask_en], color=color_en, width=days_in_month[mask_en])
    ax.bar(time[mask_ln], mei[mask_ln], color=color_ln, width=days_in_month[mask_ln])

    ax.plot([time[0], time[-1]], [0, 0], '-k')
    ax.fill_between([time[0], time[-1]], -0.5, 0.5, color='w', alpha=0.7)

    ax.text(add_month_to_time(xlim[0], 3), 2.0, 'EL NINO', rotation='vertical', va='center', ha='left')
    ax.text(add_month_to_time(xlim[0], 3), -2.0, 'LA NINA', rotation='vertical', va='center', ha='left')

    ax.set_ylim([-3.0, 3.0])
    ax.set_yticks([-3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0])
    ax.set_yticklabels([-3.0, -2.0, '', -1.0, '', 0, '', 1.0, '', 2.0, 3.0])
    ax.set_ylabel('Multivariate ENSO Index v2')
    ax.set_xlim(xlim)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax.twinx()
    ax2.set_ylim([-3.0, 3.0])
    ax2.set_yticks([-2.5, -1.75, -1.25, -0.75, 0.75, 1.25, 1.75, 2.5])
    ax2.set_yticklabels(['Very strong', 'Strong', 'Moderate', 'Weak', 'Weak', 'Moderate', 'Strong', 'Very strong'])
    ax2.yaxis.set_tick_params(length=0)

    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

def plot_dmi_index(time:np.ndarray, dmi:np.ndarray, ax=None,
                   color_pd='#900C3F', color_nd='#1e1677', xlim=None,
                   show=True, output_path=None) -> plt.axes:
    if ax is None:
        fig = plt.figure(figsize=(10, 5))
        ax = plt.axes()

    mask_pd = dmi>0
    mask_nd = dmi<0

    if xlim is None:
        xlim = [time[0], time[-1]]

    days_in_month = np.array([(add_month_to_time(t, 1)-t).days for t in time])

    ax.bar(time[mask_pd], dmi[mask_pd], color=color_pd, width=days_in_month[mask_pd])
    ax.bar(time[mask_nd], dmi[mask_nd], color=color_nd, width=days_in_month[mask_nd])

    ax.plot([time[0], time[-1]], [0, 0], '-k')
    ax.fill_between([time[0], time[-1]], -0.5, 0.5, color='w', alpha=0.7)

    ax.text(add_month_to_time(xlim[0], 3), 2.0, 'Positive IOD', rotation='vertical', va='center', ha='left')
    ax.text(add_month_to_time(xlim[0], 3), -2.0, 'Negative IOD', rotation='vertical', va='center', ha='left')

    ax.set_ylim([-3.0, 3.0])
    ax.set_yticks([-3.0, -2.0, -1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 3.0])
    ax.set_yticklabels([-3.0, -2.0, '', -1.0, '', 0, '', 1.0, '', 2.0, 3.0])
    ax.set_ylabel('Dipole Mode Index ($^o$C)')
    ax.set_xlim(xlim)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    if output_path is not None:
        log.info(f'Saving figure to: {output_path}')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        return ax

