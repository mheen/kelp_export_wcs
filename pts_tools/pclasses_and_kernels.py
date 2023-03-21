from parcels import FieldSet, ParticleSet, JITParticle, Variable
import numpy as np

class SamplingParticle(JITParticle):
    drift_age = Variable('drift_age', dtype=np.float32, initial=0)
    temperature = Variable('temperature', dtype=np.float32, initial=0)
    salinity = Variable('salinity', dtype=np.float32, initial=0)
    bathymetry = Variable('bathymetry', dtype=np.float32, initial=0)

def create_samplingparticle(fieldset:FieldSet):
    class SamplingParticle(JITParticle):
        drift_age = Variable('drift_age', dtype=np.float32, initial=0)
        temperature = Variable('temperature', dtype=np.float32, initial=fieldset.temperature)
        salinity = Variable('salinity', dtype=np.float32, initial=fieldset.salinity)
        bathymetry = Variable('bathymetry', dtype=np.float32, initial=fieldset.bathymetry)
    return SamplingParticle

# --- Kernels ---
def delete_particle(particle, fieldset, time):
    particle.delete()

def AdvectionRK4DepthCorrector(particle, fieldset:FieldSet, time):
    """Advection of particles using fourth-order Runge-Kutta integration
    as defined in Parcels but with added depth correction between integrations
    to prevent particle going through the ocean floor.

    Function needs to be converted to Kernel object before execution"""
    (u1, v1) = fieldset.UV[particle]
    lon1, lat1 = (particle.lon + u1*.5*particle.dt, particle.lat + v1*.5*particle.dt)
    # depth1 = fieldset.maxdepth[0, 0, lat1, lon1]
    xsi, eta, _, xi, yi, _ = fieldset.U.search_indices(lon1, lat1, particle.depth)
    depth_vector = (1-xsi)*(1-eta) * fieldset.U.grid.depth[:, yi, xi] + \
                    xsi*(1-eta) * fieldset.U.grid.depth[:, yi, xi+1] + \
                    xsi*eta * fieldset.U.grid.depth[:, yi+1, xi+1] + \
                    (1-xsi)*eta * fieldset.U.grid.depth[:, yi+1, xi]
    depth1 = depth_vector[0]

    (u2, v2) = fieldset.UV[time + .5 * particle.dt, depth1, lat1, lon1]
    lon2, lat2 = (particle.lon + u2*.5*particle.dt, particle.lat + v2*.5*particle.dt)
    # depth2 = fieldset.maxdepth[0, 0, lat2, lon2]
    xsi, eta, _, xi, yi, _ = fieldset.U.search_indices(lon2, lat2, depth1)
    depth_vector = (1-xsi)*(1-eta) * fieldset.U.grid.depth[:, yi, xi] + \
                    xsi*(1-eta) * fieldset.U.grid.depth[:, yi, xi+1] + \
                    xsi*eta * fieldset.U.grid.depth[:, yi+1, xi+1] + \
                    (1-xsi)*eta * fieldset.U.grid.depth[:, yi+1, xi]
    depth2 = depth_vector[0]
    
    (u3, v3) = fieldset.UV[time + .5 * particle.dt, depth2, lat2, lon2]
    lon3, lat3 = (particle.lon + u3*particle.dt, particle.lat + v3*particle.dt)
    # depth3 = fieldset.maxdepth[0, 0, lat3, lon3]
    xsi, eta, _, xi, yi, _ = fieldset.U.search_indices(lon3, lat3, depth2)
    depth_vector = (1-xsi)*(1-eta) * fieldset.U.grid.depth[:, yi, xi] + \
                    xsi*(1-eta) * fieldset.U.grid.depth[:, yi, xi+1] + \
                    xsi*eta * fieldset.U.grid.depth[:, yi+1, xi+1] + \
                    (1-xsi)*eta * fieldset.U.grid.depth[:, yi+1, xi]
    depth3 = depth_vector[0]
    
    (u4, v4) = fieldset.UV[time + particle.dt, depth3, lat3, lon3]

    particle.lon += (u1 + 2*u2 + 2*u3 + u4) / 6. * particle.dt
    particle.lat += (v1 + 2*v2 + 2*v3 + v4) / 6. * particle.dt
    # particle.depth = fieldset.maxdepth[0, 0, particle.lat, particle.lon]
    xsi, eta, _, xi, yi, _ = fieldset.U.search_indices(particle.lon, particle.lat, depth3)
    depth_vector = (1-xsi)*(1-eta) * fieldset.U.grid.depth[:, yi, xi] + \
                    xsi*(1-eta) * fieldset.U.grid.depth[:, yi, xi+1] + \
                    xsi*eta * fieldset.U.grid.depth[:, yi+1, xi+1] + \
                    (1-xsi)*eta * fieldset.U.grid.depth[:, yi+1, xi]
    particle.depth = depth_vector[0]

def SampleAge(particle, fieldset:FieldSet, time):
    particle.drift_age += particle.dt

def SampleTemperature(particle, fieldset:FieldSet, time):
    particle.temperature = fieldset.temperature[time, particle.depth, particle.lat, particle.lon]

def SampleSalinity(particle, fieldset:FieldSet, time):
    particle.salinity = fieldset.salinity[time, particle.depth, particle.lat, particle.lon]

def SampleBathymetry(particle, fieldset:FieldSet, time):
    particle.bathymetry = fieldset.bathymetry[0, 0, particle.lat, particle.lon]

def BottomDrift(particle, fieldset:FieldSet, time):
    particle.depth = fieldset.maxdepth[0, 0, particle.lat, particle.lon]

KERNELS = {
    "sample_age": SampleAge,
    "sample_temp": SampleTemperature,
    "sample_sal": SampleSalinity,
    "sample_bathy": SampleBathymetry,
    "bottom_drift": BottomDrift
}