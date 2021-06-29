import os
import glob
from rockstar_structs import *


def read_header(f):
    """Helper function to read the headers of rockstar binary files.

    :param f: (file) open file object
    :returns: header
    :rtype: (arr [head_struct])

    """
    header = np.fromfile(f, dtype=head_struct, count=1)

    assert(header['particle_type'] > 0), '-- error: no particle ID information in outputs'

    return header


def read_halo(f):
    """Helper function to read the halo information from rockstar
    binary files.

    :param f: (file) open file object

    :returns: halo objects

    :rtype: (arr [halo_struct])
    """
    
    halo = np.fromfile(f, dtype=halo_struct, count=1)

    return halo


def read_ids(f, n):
    """Helper function to read the particle IDs from rockstar binary
    files.

    :param f: (file) open file object
    :param n: (int) number of IDs to read, from the 'num_p' file of
              the halo object

    :returns: particle IDs

    :rtype: (arr [np.int64])
    """
    ids = np.fromfile(f, dtype=np.int64, count=n)

    return ids


def read_binary_haloes(root, iout, sort_key='m', most_bound=False):
    """Reads in all of the halo_<iout>.*.bin files produced by
    rockstar.  Extracts the header information (defined in
    io/io_internal.h) halo objects (defined in halo.h) and particle
    IDs for each halo sorted into ascending order.  Also returns the
    header for the final object read, so don't trust the file-specific
    values in there, useful for cosmological parameters.

    Example
    -------

    iout = 13 root = './test_data/biased/'

    h, pid = read_binary_haloes(root, iout)

    for ih in h: print(ih['m'])

    :param root: (str) path to halo output files (not including the
        actual halo filenames)
    :param iout: (int) which output to look at
    :param sort_key: (str) field to sort haloes in descending order
        by, check halo_struct in rockstar_structs for a list of fields

    :returns: haloes, particle IDs, header for the final output read

    :rtype: (arr [object]), (arr [object])
    """
    fns = glob.glob(os.path.join(root, 'halos_{0:d}.*.bin'.format(iout)))

    haloes = np.empty(len(fns), dtype='object')
    part_ids = np.empty(len(fns), dtype='object')
    sort_vals = np.empty(len(fns), dtype='object')

    for j, ifn in enumerate(fns):
        with open(ifn, 'r') as f:
            header = read_header(f)
            
            # These store each halo for this file
            cur_haloes = np.empty(header['num_halos'], dtype='object')
            cur_part_ids = np.zeros(header['num_halos'], dtype='object')
            cur_sort_vals = np.empty(header['num_halos'], dtype='object')

            # Need two loops here because the halo data are stored
            # first, then the particle IDs follow
            for i in range(header['num_halos'][0]):
                cur_haloes[i] = read_halo(f)
                cur_sort_vals[i] = cur_haloes[i][sort_key]

            for i in range(header['num_halos'][0]):
                cur_part_ids[i] = read_ids(f, cur_haloes[i]['num_p'][0])

                
                if most_bound is not False:
                    pass
                                
                # Instead, do a sort on the particle IDs then we can just
                # check whether the first one is positive
                cur_part_ids[i] = np.sort(cur_part_ids[i])
                assert(cur_part_ids[i][0] > 0), '-- error: some particle IDs < 1: {0:d}'.format(cur_part_ids[i][0])

            haloes[j] = cur_haloes
            part_ids[j] = cur_part_ids
            sort_vals[j] = cur_sort_vals


    haloes = np.concatenate(haloes)
    part_ids = np.concatenate(part_ids)
    sort_vals = np.concatenate(sort_vals).astype(halo_struct[sort_key])
    assert(haloes.shape == part_ids.shape), '-- error: read in different amounts of data for haloes and particle IDs'
    assert(haloes.shape == sort_vals.shape), '-- error: read in different amounts of data for haloes and sort values'

    # Sort the arrays
    inds = np.argsort(sort_vals)
    inds = inds[::-1]
    
    return haloes[inds], part_ids[inds], header


def read_particle_haloes(root, iout, sort_key='mvir', most_bound=50):
    """Read the halos*.particles file produced by rockstar's
    FULL_PARTICLE_CHUNKS.  In order to get all of the particles out,
    make sure you choose FULL_PARTICLE_CHUNKS=NUM_WRITERS.  This will
    only output haloes that have at least most_bound particles.

    Example
    -------

    iout = 13 root = './test_data/biased/'

    h, pid = read_binary_haloes(root, iout)

    for ih in h: print(ih['m'])

    :param root: (str) path to halo output files (not including the
        actual halo filenames)
    :param iout: (int) which output to look at
    :param sort_key: (str) field to sort haloes in descending order
        by, check halo_struct in rockstar_structs for a list of fields
    :param most_bound: (int) number of most bound particles to look
        at, also the minimum number of particles in halo outputs

    :returns: haloes, particle IDs

    :rtype: (arr [object]), (arr [object])
    """
    
    fns = glob.glob(os.path.join(root, 'halos_{0:d}.*.particles'.format(iout)))
    haloes = np.empty(len(fns), dtype='object')
    part_ids = np.empty(len(fns), dtype='object')
    sort_vals = np.empty(len(fns), dtype='object')
    
    for ii, ifn, in enumerate(fns):
        # We don't know how many we'll have, but this should be
        # enough. If you run into index errors, this would be a good
        # place to look.
        cur_haloes = np.empty(10000, dtype=object)
        cur_sort_vals = np.empty(10000, dtype=object)
        parts = np.zeros((10000000, 11), dtype=float)
        # Get number of lines read
        nhead = 0
        nhalo = 0
        npart = 0
        with open(ifn, 'r') as f:
            # print(ifn)
            # Read up until the halo table begins
            l = ''
            while l != '#Halo table begins here:\n':
                l = f.readline()
                nhead += 1

            # Skip one more line to get to the halo table, then read it
            l = f.readline()
            while l[1] != 'P':
                cur_haloes[nhalo] = parse_halo_line(l.strip('#').split())
                cur_sort_vals[nhalo] = cur_haloes[nhalo][sort_key]
                l = f.readline()
                nhalo += 1

            # Cut halo array down to size
            cur_haloes = cur_haloes[0:nhalo]
            cur_sort_vals = cur_sort_vals[0:nhalo]
            #halo_ids = np.array([ih['id'] for ih in cur_haloes], dtype=np.int64)
            # halo_parts = np.zeros(shape=(halo_ids.shape[0], most_bound),
            #                       dtype=np.int64)
            # Use an object array for consistency with the binary reading
            cur_part_ids = np.empty_like(cur_haloes, dtype=object)
        
            # Skip one more line to get to the particle table, then read it
            l = f.readline()
            # readline() returns an empty string at the end of the file
            while len(l) > 0:
                parts[npart, :] = parse_part_line(l.split())
                l = f.readline()
                npart += 1

            # Cut parts array down to size
            parts = parts[0:npart, :]
            phids = parts[:, 9].astype(int)

            # Boolean array of haloes to keep, as some of the haloes
            # output in the particles files aren't kept
            keep_haloes = np.zeros(cur_haloes.shape[0], dtype=bool)

            # Loop over haloes and assign particles to haloes, keeping
            # only the most_bound particles
            for i in range(cur_haloes.shape[0]):
                # print('-- working on halo', i)
                hid = cur_haloes[i]['id']

                # Some of these haloes won't be printed out, so if
                # they don't have a proper ID, skip them. 
                if hid == -1:
                    continue
                
                hp = cur_haloes[i]['pos'][0][0:3]  # halo position (Mpc/h)
                hv = cur_haloes[i]['pos'][0][3:6]  # halo velocity (km/s)
                p = parts[phids == hid, :]  # particles for this halo

                # Also, some of the haloes won't have very many
                # particles so skip those too.
                if (len(p) < most_bound):
                    continue
                
                # ratio of (v_i/v_esc)^2
                pvvr = most_vel_bound(hp, hv, p, cur_haloes[i]['vmax'],
                                      cur_haloes[i]['rvmax'])
                
                # Sort based on ratio of (v_i/v_esc)^2
                idxs = np.argsort(pvvr)
                # print('------ v/vesc min: {0:.3f} max: {1:.3f}'.format(pvvr[idxs[0]], pvvr[idxs[-1]]))
                
                cur_part_ids[i] = p[idxs[0:most_bound], 6]  # particle_ids
                keep_haloes[i] = True
                
        del parts
        
        haloes[ii] = cur_haloes[keep_haloes]
        part_ids[ii] = cur_part_ids[keep_haloes]
        sort_vals[ii] = cur_sort_vals[keep_haloes]

    # Combine all the arrays from each file
    haloes = np.concatenate(haloes)
    part_ids = np.concatenate(part_ids)
    sort_vals = np.concatenate(sort_vals).astype(ascii_halo_struct[sort_key])
    
    # Sort the arrays based on the sort key
    inds = np.argsort(sort_vals)
    inds = inds[::-1]

    return haloes[inds], part_ids[inds]


def parse_halo_line(l):
    """Parse the halo table in the *.particles files.

    :param l: (list [str]) Split and stripped line from the .particles
              files

    :returns: halo data

    :rtype: (arr [ascii_halo_struct])
    """
    
    halo = np.array([(int(l[0]), int(l[1]), int(l[2]), float(l[3]), float(l[4]),
                      float(l[5]), float(l[6]), float(l[7]), float(l[8]),
                      (float(l[9]), float(l[10]), float(l[11]),
                       float(l[12]), float(l[13]), float(l[14])),
                      (float(l[15]), float(l[16]), float(l[17])), float(l[18]),
                      float(l[19]))], dtype=ascii_halo_struct)
    return halo


def parse_part_line(l):
    """Parse the particle table in the *.particles files.

    :param l: (list [str]) Split and stripped line from the .particles
              files

    :returns: particle data, last column is for particle's energy

    :rtype:(arr [float])
    """
    
    # Initially set the particle energy to zero, will recalculate
    # later
    # part = np.array([((float(l[0]), float(l[1]), float(l[2]),
    #                    float(l[3]), float(l[4]), float(l[5])),
    #                   int(l[6]), int(l[7]), int(l[8]), int(l[9]), 0.)],
    #                 dtype=ascii_part_struct)

    tp = [float(il) for il in l]
    tp.append((tp[3] ** 2.) + (tp[4] ** 2.) + (tp[5] ** 2.))  # v^2
    part = np.array(tp)
    
    return part


# @numba.jit(nopython=True)
def most_bound_parts(cur_parts, most_bound):
    # for j in range(cur_parts.shape[0]):
    #     xj = cur_parts[j, 0:3]  # position of current particle
    #     # positions of all other particles
    #     iids = cur_parts[:, 6] != cur_parts[j, 6]
    #     xi = cur_parts[iids, :]
    #     xi = xi[:, 0:3]
    #     # separation of current and all other particles
    #     rij = np.sum((xi - xj) ** 2., axis=1)
    #     uj = np.sum(-1. / rij)  # potential of current particle
    #     cur_parts[j, 10] += uj  # total energy of particle

    # # Sort based on energy, smallest energy is most bound
    # idxs = np.argsort(cur_parts[:, 10])
    # # Keep only the IDs
    # nmost_bound = min(len(idxs), most_bound)
    # cur_parts = cur_parts[idxs[0:nmost_bound], 9]

    cur_parts = cur_parts[:, 9]
    return cur_parts


def escape_velocity(r, vmax, rvmax):
    """Escape velocity (squared) for a particle in an NFW halo.  Taken
    from Kylpin+ (1997)

    V_esc(r)^2 = (2.15*V_max)^2 ln(1 + 2*r/r_max)/(r/r_max)

    :param r: (arr) distances from halo centre, input to V_esc (Mpc/h)
    :param vmax: (float) maximum velocity (km/s)
    :param rvmax: (float) radius of maximum velocity (Mpc/h)
    """
    rr = r / rvmax
    return (2.15 * vmax)**2. * np.log(1. + 2.*rr) / rr


def check_pvvr(pvvr):
    if np.any(np.isnan(pvvr)):
        print('------ [warning] NaNs found, probably Vmax is undefined!')
                    # print('vesc:', vr)
                    # print('vmax:', cur_haloes[i]['vmax'])
                    # print('rvmax:', cur_haloes[i]['rvmax'])
                    # print('np:', len(pvvr))
                    # print('p.shape:', p.shape)
                    # print('phids.shape:', phids.shape)
                    # print('np.sum(phids==hid):', np.sum(phids==hid))
                    # print('hid:', hid)

def most_vel_bound(hp, hv, p, vmax, rvmax):
    """Returns the ratio of the halo particles to the escape
    velocities.  Using this as a proxy for most bound particles.  I
    haven't made up my mind whether this actually makes sense, but
    it's a _lot_ quicker than summing over particles to find the
    potential.  Seems to work fine, based on halo properties.

    :param hp: (arr [3]) halo position in Mpc/h
    :param hv: (arr [3]) halo velocity in km/s
    :param p: (arr [npart, 11]) particle array
    :param vmax: (float) maxmimum velocity (km/s)
    :param rvmax: (float) radius of maximum velocity (km/s)

    :returns: (v_i/v_esc)^2

    :rtype: (arr [npart])
    """
    # particle distance relative to halo centre (Mpc/h)
    pr = np.sqrt(np.sum((p[:, 0:3] - hp) ** 2., axis=1))
    # particle velocity^2 relative to halo bulk velocity (km/s)
    pv = np.sum((p[:, 3:6] - hv) ** 2., axis=1)
    # escape velocity^2 for each particle
    vr = escape_velocity(pr, vmax, rvmax)
    pvvr = pv / vr    # ratio of particle velocity to escape velocity
    check_pvvr(pvvr)  # check for NaNs

    return pvvr
