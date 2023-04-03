import numpy as np

def get_angle_between_mask_and_velocity(mask:np.ndarray[bool],
                                        u:np.ndarray[float],
                                        v:np.ndarray[float]) -> np.ndarray[float]:
    '''Calculates the angle between a given mask (this can be a
    land mask or a mask indicating the continental shelf edge)
    and given velocities (current or wind). This is done by:
    
    The angle of the mask (land/shelf) can be determined by:
    alpha_mask = arctan2(dL/dx, dL/dy)
    since the gradient of the mask will indicate the direction.
    
    The dot product between two vectors is the cosine of the
    angle between vectors:
    cos(theta) = x.y/|x||y|
    
    So the angle between a mask and velocity can be calculated:
    theta = arccos((u dL/dx + v dL/dy)/(sqrt(u**2+v**2)sqrt(dL/dx**2+dL/dy**2)))'''

    dLdy, dLdx = np.gradient(mask)

    numerator = u*dLdx+v*dLdy
    denominator = np.sqrt(u**2+v**2)*np.sqrt(dLdx**2+dLdy**2)
    theta = np.arccos(numerator/denominator)

    return theta

def get_cross_shelf_velocity(h:np.ndarray[float],
                             u:np.ndarray[float],
                             v:np.ndarray[float]) -> np.ndarray[float]:
    
    theta = get_angle_between_mask_and_velocity(h, u, v)
    u_cross_shelf = np.sqrt(u**2+v**2)*np.cos(theta)

    return u_cross_shelf