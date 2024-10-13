# -*- coding: utf-8 -*-
"""
This is a module for simulating galactic collisions
inspired by Schroder & Comins, Astronomy 1988/12, 90-96
"""

import numpy as np

import astropy.units as unit
from astropy.constants import G,kpc

from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle


# SETUP
# =====

def parameters(*args):
    '''
    creates dictionary of galaxy parameters

    args: mass in solar masses, radius in kpc,
          position of center in kpc, velocity of center in km/s,
          normal vector of galactic plane,
          number of rings, number of stars,
          softening factor to limit potential at center

    returns: dictionary of parameters with astropy units
    '''
    if len(args) != 8:
        print("Error: 8 arguments required")
    else:
        try:
            if len(args[2]) == 3 and len(args[3]) == 3 and len(args[4]) == 3:
                return { "mass"       : args[0]*unit.M_sun,
                         "radius"     : args[1]*unit.kpc,
                         "center_pos" : args[2]*unit.kpc,
                         "center_vel" : args[3]*unit.km/unit.s,
                         "normal"     : args[4],
                         "N_rings"    : args[5],
                         "N_stars"    : args[6],
                         "softening"  : args[7] }
            else:
                print("Error: 3., 4., and 5. argument must be a 3-tuple")
        except TypeError:
            print("Error: invalid argument")


def init_disk(galaxy, time_step=0.1*unit.Myr):
    '''
    initializes galaxy by setting stars in random positions
    and Keplerian velocities half a time step in advance
    (Leapfrog scheme)

    args: dictionary of galaxy parameters,
          numerical time step
    '''
    dr = (1 - galaxy['softening'])*galaxy['radius']/galaxy['N_rings'] # width of a ring
    N_stars_per_ring = int(galaxy['N_stars']/galaxy['N_rings'])

    # rotation angle and axis
    norm = np.sqrt(galaxy['normal'][0]**2 + galaxy['normal'][1]**2 + galaxy['normal'][2]**2)
    cos_theta = galaxy['normal'][2]/norm
    sin_theta = np.sqrt(1-cos_theta**2)
    u = np.cross([0,0,1], galaxy['normal']/norm)
    norm = np.sqrt(u[0]**2 + u[1]**2 + u[2]**2)

    if norm > 0:
        u /= norm # unit vector

        # rotation matrix for coordinate transformation
        # from galactic plane to observer's frame
        rotation = [[u[0]*u[0]*(1-cos_theta) + cos_theta,
                     u[0]*u[1]*(1-cos_theta) - u[2]*sin_theta,
                     u[0]*u[2]*(1-cos_theta) + u[1]*sin_theta],
                     [u[1]*u[0]*(1-cos_theta) + u[2]*sin_theta,
                      u[1]*u[1]*(1-cos_theta) + cos_theta,
                      u[1]*u[2]*(1-cos_theta) - u[0]*sin_theta],
                     [u[2]*u[0]*(1-cos_theta) - u[1]*sin_theta,
                      u[2]*u[1]*(1-cos_theta) + u[0]*sin_theta,
                      u[2]*u[2]*(1-cos_theta) + cos_theta]]

        # print angels defining orientation of galaxy
        phi = np.arctan2(galaxy['normal'][1], galaxy['normal'][0])
        theta = np.arccos(cos_theta)
        print("Plane normal: phi = {:.1f}°, theta = {:.1f}°".\
              format(np.degrees(phi), np.degrees(theta)))

    else:
        rotation = np.identity(3)

    galaxy['stars_pos'] = np.array([])
    galaxy['stars_vel'] = np.array([])

    # begin with innermost radius given by softening factor
    R = galaxy['softening']*galaxy['radius']
    for n in range(galaxy['N_rings']):

        # radial and angular coordinates in center-of-mass frame
        r_star = R + dr * np.random.random_sample(size=N_stars_per_ring)
        phi_star = 2*np.pi * np.random.random_sample(size=N_stars_per_ring)

        # Cartesian coordinates in observer's frame
        vec_r = np.dot(rotation,
                       r_star*[np.cos(phi_star),
                               np.sin(phi_star),
                               np.zeros(N_stars_per_ring)])
        x = galaxy['center_pos'][0] + vec_r[0]
        y = galaxy['center_pos'][1] + vec_r[1]
        z = galaxy['center_pos'][2] + vec_r[2]

        # orbital periods and angular displacements over one timestep
        T_star = 2*np.pi * ((G*galaxy['mass'])**(-1/2) * r_star**(3/2)).to(unit.s)
        delta_phi = 2*np.pi * time_step.to(unit.s).value / T_star.value

        # velocity components in observer's frame half a step in advance
        # (Leapfrog scheme)
        vec_v = np.dot(rotation,
                       (r_star.to(unit.km)/time_step.to(unit.s)) *
                       [(np.cos(phi_star) - np.cos(phi_star - delta_phi)),
                        (np.sin(phi_star) - np.sin(phi_star - delta_phi)),
                        np.zeros(N_stars_per_ring)])
        v_x = galaxy['center_vel'][0] + vec_v[0]
        v_y = galaxy['center_vel'][1] + vec_v[1]
        v_z = galaxy['center_vel'][2] + vec_v[2]

        if galaxy['stars_pos'].size == 0:
            galaxy['stars_pos'] = np.array([x,y,z])
            galaxy['stars_vel'] = np.array([v_x,v_y,v_z])
        else:
            galaxy['stars_pos'] = np.append(galaxy['stars_pos'], np.array([x,y,z]), axis=1)
            galaxy['stars_vel'] = np.append(galaxy['stars_vel'], np.array([v_x,v_y,v_z]), axis=1)

        R += dr

    # units get lost through np.array
    galaxy['stars_pos'] *= unit.kpc
    galaxy['stars_vel'] *= unit.km/unit.s

    # typical velocity scale defined by Kepler velocity at half the disk radius
    galaxy['vel_scale'] = np.sqrt(G*galaxy['mass']/(0.5*R)).to(unit.km/unit.s)


# NUMERICAL SOLVER
# ================

def evolve_disk(galaxy, time_step=0.1*unit.Myr, N_steps=1000, N_snapshots=100):
    '''
    evolves isolated disk using Leapfrog integration

    args: dictionary of galaxy,
          numerical timestep, number of timesteps,
          number of snapshots

    returns: array of snapshot times,
             array of snapshots (spatial coordinates of centers and stars)
    '''
    dt = time_step.to(unit.s).value
    r_min = galaxy['softening']*galaxy['radius'].to(unit.m).value
    N_stars = galaxy['N_stars']

    # mass, position and velocity of galactic center
    M = galaxy['mass'].to(unit.kg).value
    X, Y, Z = galaxy['center_pos'].to(unit.m).value
    V_x, V_y, V_z = galaxy['center_vel'].to(unit.m/unit.s).value

    # initialize stellar coordintes
    x = galaxy['stars_pos'][0].to(unit.m).value
    y = galaxy['stars_pos'][1].to(unit.m).value
    z = galaxy['stars_pos'][2].to(unit.m).value

    # intialize stellar velocities
    v_x = galaxy['stars_vel'][0].to(unit.m/unit.s).value
    v_y = galaxy['stars_vel'][1].to(unit.m/unit.s).value
    v_z = galaxy['stars_vel'][2].to(unit.m/unit.s).value

    # array to store snapshots of all positions (centers and stars)
    snapshots = np.zeros(shape=(N_snapshots+1,3,N_stars+1))
    snapshots[0] = [np.append([X], x), np.append([Y], y), np.append([Z], z)]
    #print(snapshots.shape)

    # number of steps per snapshot
    div = max(int(N_steps/N_snapshots), 1)

    print("Solving equations of motion for single galaxy (Leapfrog integration)")

    for n in range(1,N_steps+1):

        # radial distances from center
        r = np.maximum(np.sqrt((X - x)**2 + (Y - y)**2 + (Z - z)**2), r_min)

        # update velocities of stars (acceleration due to gravity of center)
        v_x += G.value*M * ((X - x)/r**3) * dt
        v_y += G.value*M * ((Y - y)/r**3) * dt
        v_z += G.value*M * ((Z - z)/r**3) * dt

        # update positions of stars
        x += v_x*dt
        y += v_y*dt
        z += v_z*dt

        # update position of center
        X += V_x*dt
        Y += V_y*dt
        Z += V_z*dt

        if n % div == 0:
            i = int(n/div)
            snapshots[i] = [np.append([X], x), np.append([Y], y), np.append([Z], z)]

        # fraction of computation done
        print("\r{:3d} %".format(int(100*n/N_steps)), end="")

    time = np.linspace(0*time_step, N_steps*time_step, N_snapshots+1, endpoint=True)
    print(" (stopped at t = {:.1f})".format(time[-1]))

    snapshots *= unit.m

    return time, snapshots.to(unit.kpc)


def evolve_two_disks(primary, secondary, time_step=0.1*unit.Myr, N_steps=1000, N_snapshots=100):
    '''
    evolves primary and secondary disk using Leapfrog integration

    args: dictionaries of primary and secondary galaxy,
          numerical timestep, number of timesteps,
          number of snapshots

    returns: array of snapshot times,
             array of snapshots (spatial coordinates of centers and stars)
    '''
    dt = time_step.to(unit.s).value

    r_min1 = primary['softening']*primary['radius'].to(unit.m).value
    r_min2 = secondary['softening']*secondary['radius'].to(unit.m).value

    N1, N2 = primary['N_stars'], secondary['N_stars']

    # mass, position and velocity of primary galactic center
    M1 = primary['mass'].to(unit.kg).value
    X1, Y1, Z1 = primary['center_pos'].to(unit.m).value
    V1_x, V1_y, V1_z = primary['center_vel'].to(unit.m/unit.s).value

    # mass, position and velocity of secondary galactic center
    M2 = secondary['mass'].to(unit.kg).value
    X2, Y2, Z2 = secondary['center_pos'].to(unit.m).value
    V2_x, V2_y, V2_z = secondary['center_vel'].to(unit.m/unit.s).value

    # stellar coordintes of primary
    x = primary['stars_pos'][0].to(unit.m).value
    y = primary['stars_pos'][1].to(unit.m).value
    z = primary['stars_pos'][2].to(unit.m).value

    # stellar coordintes of secondary
    x = np.append(x, secondary['stars_pos'][0].to(unit.m).value)
    y = np.append(y, secondary['stars_pos'][1].to(unit.m).value)
    z = np.append(z, secondary['stars_pos'][2].to(unit.m).value)

    # stellar velocities of primary
    v_x = primary['stars_vel'][0].to(unit.m/unit.s).value
    v_y = primary['stars_vel'][1].to(unit.m/unit.s).value
    v_z = primary['stars_vel'][2].to(unit.m/unit.s).value

    # stellar velocities of secondary
    v_x = np.append(v_x, secondary['stars_vel'][0].to(unit.m/unit.s).value)
    v_y = np.append(v_y, secondary['stars_vel'][1].to(unit.m/unit.s).value)
    v_z = np.append(v_z, secondary['stars_vel'][2].to(unit.m/unit.s).value)

    # array to store snapshots of all positions (centers and stars)
    snapshots = np.zeros(shape=(N_snapshots+1,3,N1+N2+2))
    snapshots[0] = [np.append([X1,X2], x), np.append([Y1,Y2], y), np.append([Z1,Z2], z)]
    #print(snapshots.shape)

    # number of steps per snapshot
    div = max(int(N_steps/N_snapshots), 1)

    print("Solving equations of motion for two galaxies (Leapfrog integration)")

    for n in range(1,N_steps+1):

        # radial distances from centers with softening
        r1 = np.maximum(np.sqrt((X1 - x)**2 + (Y1 - y)**2 + (Z1 - z)**2), r_min1)
        r2 = np.maximum(np.sqrt((X2 - x)**2 + (Y2 - y)**2 + (Z2 - z)**2), r_min2)
        #print("\nr {:.6e} {:.6e} {:.6e} {:.6e}".format(r1[0],r2[0],r1[N1],r2[N1]))

        # update velocities of stars (acceleration due to gravity of centers)
        v_x += G.value*(M1*(X1 - x)/r1**3 + M2*(X2 - x)/r2**3) * dt
        v_y += G.value*(M1*(Y1 - y)/r1**3 + M2*(Y2 - y)/r2**3) * dt
        v_z += G.value*(M1*(Z1 - z)/r1**3 + M2*(Z2 - z)/r2**3) * dt
        #print("v_x {:.1f} {:.1f}".format(v_x[0],v_x[N1]))

        # update positions of stars
        x += v_x*dt
        y += v_y*dt
        z += v_z*dt
        #print("x {:.6e} {:.6e}".format(x[0],x[N1]))

        # distance between centers
        D_sqr_min = (r_min1+r_min2)**2
        D_cubed = (max((X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2, D_sqr_min))**(3/2)

        # gravitational acceleration of primary center
        A1_x = G.value*M2*(X2 - X1)/D_cubed
        A1_y = G.value*M2*(Y2 - Y1)/D_cubed
        A1_z = G.value*M2*(Z2 - Z1)/D_cubed

        # update velocities of centers (constant center-of-mass velocity)
        V1_x += A1_x*dt; V2_x -= (M1/M2)*A1_x*dt
        V1_y += A1_y*dt; V2_y -= (M1/M2)*A1_y*dt
        V1_z += A1_z*dt; V2_z -= (M1/M2)*A1_z*dt
        #print("V {:.1f} {:.1f} {:.1f}".format(V1_x,V2_x,(M1*V1_x+M2*V2_x)/(M1+M2)))

        # update positions of centers
        X1 += V1_x*dt; X2 += V2_x*dt
        Y1 += V1_y*dt; Y2 += V2_y*dt
        Z1 += V1_z*dt; Z2 += V2_z*dt
        #print("X {:.6e} {:.6e} {:.6e}".format(X1,X2,X1+X2))

        if n % div == 0:
            i = int(n/div)
            snapshots[i] = [np.append([X1,X2], x), np.append([Y1,Y2], y), np.append([Z1,Z2], z)]

        # fraction of computation done
        print("\r{:3d} %".format(int(100*n/N_steps)), end="")

    time = np.linspace(0*time_step, N_steps*time_step, N_snapshots+1, endpoint=True)
    print(" (stopped at t = {:.1f})".format(time[-1]))

    snapshots *= unit.m

    return time, snapshots.to(unit.kpc)


# VISUALIZATION
# =============

def show_orbits(stars, data):
    '''
    plots orbits of stars in xy-plane

    args: array of star indices,
          snapshots returned by evolve_disk
    '''
    fig, ax = plt.subplots(figsize=(10,10), dpi=100)
    
    ax.set_aspect('equal')

    for n in stars:
        orbit = data[:,:,n].transpose()
        ax.plot(orbit[0], orbit[1], lw=1)

    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)


def anim_orbits(stars, data, xlim, ylim, time=None, name='orbits'):
    '''
    animates orbits of stars in xy-plane

    args: array of star indices,
          snapshots returned by evolve_disk,
          plot range of x- and y-coordinates,
          snapshot times
          file name of video
    '''
    aspect_ratio = (ylim[1]-ylim[0])/(xlim[1]-xlim[0])

    fig, ax = plt.subplots(figsize=(10, 1.0+10*aspect_ratio), dpi=150)
    
    ax.set_aspect('equal')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)

    curves = []
    for n in stars:
        curve, = ax.plot([], [], lw=1)
        curves.append(curve)

    if time != None:
        title = ax.set_title('$t$ = {:.1f}'.format(time[0]))

    def update(i):
        for m,n in enumerate(stars):
            orbit = data[:,:,n].transpose()
            curves[m].set_data(orbit[0,0:i], orbit[1,0:i])
        if time != None:
            title.set_text('$t$ = {:.1f}'.format(time[i]))
        return curves

    anim = FuncAnimation(fig, update, frames=len(data), interval=25, blit=True, repeat=False)

    anim.save(name + '.mp4')


def anim_disk_2d(data, xlim, ylim, name='disk_2d'):
    '''
    animates single disk in xy-plane

    args: snapshots returned by evolve_disk,
          plot range of x- and y-coordinates,
          file name of video
    '''
    aspect_ratio = (ylim[1]-ylim[0])/(xlim[1]-xlim[0])

    fig, ax = plt.subplots(figsize=(10, 1.0+10*aspect_ratio), dpi=150)
    
    ax.set_aspect('equal')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)

    sct = ax.scatter(data[0,0,:], data[0,1,:], s=2, marker='.', color='blue')

    def update(i):
        sct.set_offsets(np.c_[data[i,0,:], data[i,1,:]])
        return sct,

    anim = FuncAnimation(fig, update, frames=len(data), interval=25, blit=True, repeat=False)

    anim.save(name+'.mp4')


def show_disk_3d(snapshot, xlim=None, ylim=None, zlim=None):
    '''
    plots stars for a single snapshot of an isolated disk in 3D

    args: snapshot data,
          plot range of x-, y- and z-coordinates
    '''
    fig = plt.figure(figsize=(10,10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)
    ax.set_zlabel(r'$z$ [kpc]', fontsize=12)

    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if zlim != None: ax.set_zlim(zlim[0], zlim[1])

    ax.scatter(snapshot[0,1:], snapshot[1,1:], snapshot[2,1:], marker='.', color='blue', s=2)


def anim_disk_3d(data, xlim=None, ylim=None, zlim=None, name='disk_3d'):
    '''
    animates isolated disk system in 3D

    args: snapshots returned by evolve_disk,
          plot range of x-, y- and z-coordinates,
          file name of video
    '''
    fig = plt.figure(figsize=(10,10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)
    ax.set_zlabel(r'$z$ [kpc]', fontsize=12)

    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if zlim != None: ax.set_zlim(zlim[0], zlim[1])

    sct = ax.scatter(data[0,0,1:], data[0,1,1:], data[0,2,1:], s=2, marker='.', color='blue')

    def update(i):
        sct._offsets3d  = (data[i,0,1:], data[i,1,1:], data[i,2,1:])
        return sct,

    anim = FuncAnimation(fig, update, frames=len(data), interval=25, blit=True, repeat=False)

    anim.save(name+'.mp4')


def show_two_disks_2d(snapshot, N1, xlim, ylim):
    '''
    plots centers and stars for a single snapshot of two disks

    args: snapshot data,
          number of stars in primary disk,
          plot range of x- and y-coordinates
    '''
    N2 = snapshot.shape[1]-2-N1
    aspect_ratio = (ylim[1]-ylim[0])/(xlim[1]-xlim[0])

    fig, ax = plt.subplots(figsize=(10, 1.0+10*aspect_ratio), dpi=100)

    ax.set_aspect('equal')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)

    ax.scatter(snapshot[0,0:2], snapshot[1,0:2], marker='+', color='black')
    ax.scatter(snapshot[0,2:N1+2], snapshot[1,2:N1+2], s=2, marker='.', color='blue')
    ax.scatter(snapshot[0,N1+2:N1+N2+2], snapshot[1,N1+2:N1+N2+2], s=2, marker='.', color='red')


def anim_two_disks_2d(data, N1, xlim, ylim, name='two_disks_2d'):
    '''
    animates two-disk system in xy-plane

    args: snapshots returned by evolve_two_disks,
          number of stars in primary disk,
          plot range of x- and y-coordinates,
          file name of video
    '''
    N2 = data.shape[2]-2-N1
    aspect_ratio = (ylim[1]-ylim[0])/(xlim[1]-xlim[0])

    fig, ax = plt.subplots(figsize=(10, 1.0+10*aspect_ratio), dpi=150)

    ax.set_aspect('equal')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)

    sct  = ax.scatter(data[0,0,0:2],          data[0,1,0:2],          marker='+', color='black')
    sct1 = ax.scatter(data[0,0,2:N1+2],       data[0,1,2:N1+2],       marker='.', color='blue', s=2)
    sct2 = ax.scatter(data[0,0,N1+2:N1+N2+2], data[0,1,N1+2:N1+N2+2], marker='.', color='red', s=2)

    def update(i):
        sct.set_offsets( np.c_[data[i,0,0:2],          data[i,1,0:2]])
        sct1.set_offsets(np.c_[data[i,0,2:N1+2],       data[i,1,2:N1+2]])
        sct2.set_offsets(np.c_[data[i,0,N1+2:N1+N2+2], data[i,1,N1+2:N1+N2+2]])
        return sct,sct1,sct2,

    anim = FuncAnimation(fig, update, frames=len(data), interval=25, blit=True, repeat=False)

    anim.save(name + '.mp4')


def show_two_disks_3d(snapshot, N1, xlim=None, ylim=None, zlim=None, time=None, name=None):
    '''
    plots centers and stars for a single snapshot of two disks in 3D

    args: snapshot data,
          number of stars in primary disk,
          plot range of x- and y-coordinates,
          snapshot time,
          file name
    '''
    N2 = snapshot.shape[1]-2-N1

    fig = plt.figure(figsize=(10,10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)
    ax.set_zlabel(r'$z$ [kpc]', fontsize=12)

    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if zlim != None: ax.set_zlim(zlim[0], zlim[1])

    if time != None:
        title = ax.set_title('$t$ = {:.1f}'.format(time))

    ax.scatter(snapshot[0,0:2],          snapshot[1,0:2],          snapshot[2,0:2], \
               marker='+', color='black')
    ax.scatter(snapshot[0,2:N1+2],       snapshot[1,2:N1+2],       snapshot[2,2:N1+2], \
               marker='.', color='blue', s=2)
    ax.scatter(snapshot[0,N1+2:N1+N2+2], snapshot[1,N1+2:N1+N2+2], snapshot[2,N1+2:N1+N2+2], \
               marker='.', color='red', s=2)

    if name != None:
        plt.savefig(name + '_{:.0f}.pdf'.format(time))

def anim_two_disks_3d(data, N1, xlim=None, ylim=None, zlim=None, time=None, name='two_disks_3d'):
    '''
    animates two-disk system in 3D

    args: snapshots returned by evolve_two_disks,
          number of stars in primary disk,
          plot range of x-, y- and z-coordinates,
          snapshot times,
          file name of video
    '''
    N2 = data.shape[2]-2-N1

    fig = plt.figure(figsize=(10,10), dpi=150)
    ax = fig.add_subplot(111, projection='3d')

    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel(r'$x$ [kpc]', fontsize=12)
    ax.set_ylabel(r'$y$ [kpc]', fontsize=12)
    ax.set_zlabel(r'$z$ [kpc]', fontsize=12)

    if xlim != None: ax.set_xlim(xlim[0], xlim[1])
    if ylim != None: ax.set_ylim(ylim[0], ylim[1])
    if zlim != None: ax.set_zlim(zlim[0], zlim[1])

    sct  = ax.scatter(data[0,0,0:2],          data[0,1,0:2],          data[0,2,0:2], \
                      marker='+', color='black')
    sct1 = ax.scatter(data[0,0,2:N1+2],       data[0,1,2:N1+2],       data[0,2,2:N1+2], \
                      marker='.', color='blue', s=2)
    sct2 = ax.scatter(data[0,0,N1+2:N1+N2+2], data[0,1,N1+2:N1+N2+2], data[0,2,N1+2:N1+N2+2], \
                      marker='.', color='red', s=2)
    if time != None:
        title = ax.set_title('$t$ = {:.1f}'.format(time[0]))

    def update(i):
        sct._offsets3d  = (data[i,0,0:2],          data[i,1,0:2],          data[i,2,0:2])
        sct1._offsets3d = (data[i,0,2:N1+2],       data[i,1,2:N1+2],       data[i,2,2:N1+2])
        sct2._offsets3d = (data[i,0,N1+2:N1+N2+2], data[i,1,N1+2:N1+N2+2], data[i,2,N1+2:N1+N2+2])
        if time != None:
            title.set_text('$t$ = {:.1f}'.format(time[i]))
        return sct,sct1,sct2,

    anim = FuncAnimation(fig, update, frames=len(data), interval=10, blit=True)

    anim.save(name + '.mp4')
