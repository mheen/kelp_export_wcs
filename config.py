import os, sys
parent = os.path.abspath('.')
sys.path.insert(1, parent)

from tools.files import get_dir_from_json
from dataclasses import dataclass
import configparser
from datetime import datetime, timedelta

defaults_path = 'configs/defaults.ini'
pts_section_header = 'pts'

@dataclass
class PtsConfig:
    start_date: datetime
    end_date_releases: datetime
    end_date_run: datetime
    release_region: str
    run_region: str
    region_name: str
    input_dir: str
    output_dir: str
    n_thin_initial: int
    dt: timedelta
    dt_out: timedelta
    export_variables: list
    elements: str
    reader: str
    advection_scheme: str
    vertical_advection: bool
    vertical_mixing: bool
    horizontal_diffusivity: float
    use_auto_landmask: bool
    coastline_action: str
    grid_file: bool
    extra_description: str
    sub_output_dir: str
    input_dir_description: str

def get_pts_config(input_path:str) -> PtsConfig:
    config = configparser.ConfigParser()
    config.read(defaults_path)
    config.read(input_path)

    start_date = datetime.strptime(config[pts_section_header]['start_date'], '%d-%m-%Y')
    end_date_releases = datetime.strptime(config[pts_section_header]['end_date_releases'], '%d-%m-%Y')
    end_date_run = datetime.strptime(config[pts_section_header]['end_date_run'], '%d-%m-%Y')
    release_region = config[pts_section_header]['release_region']
    run_region = config[pts_section_header]['run_region']
    region_name = config[pts_section_header]['region_name']
    input_dir = get_dir_from_json(config[pts_section_header]['input_dir'])
    output_dir = get_dir_from_json(config[pts_section_header]['output_dir'])
    n_thin_inital = int(config[pts_section_header]['n_thin_initial'])
    dt = timedelta(seconds=int(config[pts_section_header]['dt']))
    dt_out = timedelta(seconds=int(config[pts_section_header]['dt_out']))
    export_variables = [i.strip() for i in config[pts_section_header]['export_variables'].split(',')]
    elements = config[pts_section_header]['elements']
    reader = config[pts_section_header]['reader']
    advection_scheme = config[pts_section_header]['advection_scheme']
    vertical_advection = bool(0 if config[pts_section_header]['vertical_advection'].lower == 'false' else 1)
    vertical_mixing = bool(0 if config[pts_section_header]['vertical_mixing'].lower == 'false' else 1)
    horizontal_diffusivity = float(config[pts_section_header]['horizontal_diffusivity'])
    use_auto_landmask = bool(0 if config[pts_section_header]['use_auto_landmask'].lower() == 'false' else 1)
    coastline_action = config[pts_section_header]['coastline_action']
    grid_file = bool(0 if config[pts_section_header]['grid_file'].lower == 'false' else 1)
    extra_description = config[pts_section_header]['extra_description']
    sub_output_dir = config[pts_section_header]['sub_output_dir']
    input_dir_description = config[pts_section_header]['input_dir_description']

    pts_config = PtsConfig(start_date, end_date_releases, end_date_run, release_region, run_region, region_name,
                           input_dir, output_dir, n_thin_inital, dt, dt_out, export_variables, elements,
                           reader, advection_scheme, vertical_advection, vertical_mixing, horizontal_diffusivity,
                           use_auto_landmask, coastline_action, grid_file, extra_description, sub_output_dir,
                           input_dir_description)

    return pts_config

if __name__ == '__main__':
    config = get_pts_config('configs/cwa_perth.ini')
