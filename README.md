# kelp_export_wcs
Export of kelp detritus from the Wadjemup Continental Shelf

## Set-up
Use the kelp_export_wcs.yml file to create the conda environment needed to use this code.

The `get_dir_from_json` function (in `tools/files.py`) by default expects a JSON file with input directories listed in `input/dirs.json`. This file needs to be made and relevant directories added to it (or the `get_dir_from_json` commands need to be replaced with full paths).

## Particle tracking simulations
Main script to run particle tracking simulations: pts_kelp_opendrift.py

## Manuscript plots
Main manuscript plots are made in: plots_vanderMheen-etal-2023.py

Supplemental plots are made in: plots_si_vanderMheen-etal-2023.py
