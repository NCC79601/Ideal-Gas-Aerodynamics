# Things about collision detection
import numpy as np


def is_collide_ptcl_ptcl(pos1, v1, pos2, v2, r) -> tuple:
    '''
    Judge whether two particles collide or not, and return the collision time if they collide

    Parameters:
    - pos1: position of the first particle
    - v1: velocity of the first particle
    - pos2: position of the second particle
    - v2: velocity of the second particle
    - r: radius of the particles

    Returns:
    - (is_collide, t): whether the particles collide or not, and the collision time
    '''
    x1, y1 = pos1[0], pos1[1]
    x2, y2 = pos2[0], pos2[1]
    v1x, v1y = v1[0], v1[1]
    v2x, v2y = v2[0], v2[1]
    # compute quadratic equation coefficients
    a = (v1x - v2x) ** 2 + (v1y - v2y) ** 2
    b = 2 * ((x1 - x2) * (v1x - v2x) + (y1 - y2) * (v1y - v2y))
    c = (x1 - x2) ** 2 + (y1 - y2) ** 2 - (2 * r) ** 2
    # check if particles collide
    if b ** 2 - 4 * a * c >= 0:
        if a == 0:
            return True, np.inf
        t = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
        return True, t
    else:
        return False, None


def is_collide_ptcl_wall(pos, v, point1, point2, r) -> tuple:
    '''
    Judge whether a particle collide with a wall or not, and return the collision time if they collide

    Parameters:
    - pos: position of the particle
    - v: velocity of the particle
    - point1: the first point of the wall
    - point2: the second point of the wall
    - r: radius of the particle

    Returns:
    - (is_collide, t): whether the particle collide with the wall or not, and the collision time
    '''
    x, y   = pos[0], pos[1]
    vx, vy = v[0], v[1]
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    v_dir  = np.array([x2 - x1, y2 - y1])

    # judge whether the particle is moving towards the wall
    if np.cross(v_dir, np.array([x - x1, y - y1])) * np.cross(v_dir, np.array([vx, vy])) > 0:
        return False, None
    
    # compute line equation coefficients (A * x + B * y + C = 0)
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    # compute collision time
    t = (-r * np.sqrt(A**2 + B**2) - (A * x + B * y + C)) / (A * vx + B * vy)
    return True, t


def collide_particle_particle(p1, v1, p2, v2) -> None:
    '''
    Perform collision between particles

    Parameters:
    - p1: position of the first particle
    - v1: velocity of the first particle
    - p2: position of the second particle
    - v2: velocity of the second particle

    Returns:
    - (v1_new, v2_new): new velocities of the particles after collision
    '''
    # reference: https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
    v1_new = v1 - (np.dot(v1 - v2, p1 - p2) / np.linalg.norm(p1 - p2) ** 2) * (p1 - p2)
    v2_new = v2 - (np.dot(v2 - v1, p2 - p1) / np.linalg.norm(p2 - p1) ** 2) * (p2 - p1)
    
    return v1_new, v2_new


def collide_particle_wall(v, point1, point2) -> None:
    '''
    Perform collision between particle `i` and wall `wall_id`

    Parameters:
    - v: velocity of the particle
    - point1: the first point of the wall
    - point2: the second point of the wall

    Returns:
    - new velocity of the particle after collision
    '''
    # compute normal vector of the wall
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    n = np.array([y1 - y2, x2 - x1])
    n /= np.linalg.norm(n)

    # compute new velocity
    new_v = v - 2 * np.dot(v, n) * n

    return new_v