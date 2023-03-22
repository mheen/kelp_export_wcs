# This file is part of OpenDrift.
#
# OpenDrift is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2
#
# OpenDrift is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with OpenDrift.  If not, see <https://www.gnu.org/licenses/>.
#
# Copyright 2015, Knut-Frode Dagestad, MET Norway

import numpy as np
from opendrift.models.oceandrift import Lagrangian3DArray, OceanDrift
import logging

logger = logging.getLogger(__name__)

class BottomDrifters(OceanDrift):
    """
    Buoyant particle trajectory model with particles subject vertical
    turbulent mixing with diurnal vertical migration to represent
    the behaviour of prawn larvae, based on the LarvalFish model.
    """

    ElementType = Lagrangian3DArray

    required_variables = {
        'x_sea_water_velocity': {'fallback': 0},
        'y_sea_water_velocity': {'fallback': 0},
        'land_binary_mask': {'fallback': None},
        'sea_floor_depth_below_sea_level': {'fallback': 100},
        'sea_water_temperature': {'fallback': 10, 'profiles': True},
        'sea_water_salinity': {'fallback': 34, 'profiles': True}
    }

    def __init__(self, *args, **kwargs):
        
        # Calling general constructor of parent class
        super(BottomDrifters, self).__init__(*args, **kwargs)

    def update_depth(self):
        self.elements.z = -self.environment.sea_floor_depth_below_sea_level

    def update(self):
        self.update_depth()
        self.advect_ocean_current()
