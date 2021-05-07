import os
import sys
import glob
import numpy as np


def get_path(root, iout):
    return os.path.join(root, 'AHF/{0:03d}/halos/all_{0:03d}.'.format(iout))


def read_ahf_particles(path, most_bound=0):
    path += 'AHF_particles'
    
    fns = glob.glob(path)

    if len(fns) < 1:
        print('-- [error] {0} not found'.format(path))
        sys.exit()
        
    fn = fns[0]
    
    nhalo_tot = 0
    # hids = []
    pids = {}
    
    with open(fn, 'r') as f:
        while True:
            line = f.readline()
            # Check that we're still reading
            if not line:
                break

            nhalo = int(line.strip('\n'))
            nhalo_tot += nhalo
        
            for i in range(nhalo):
                l = f.readline().strip('\n').split()
                
                npart = int(l[0])
                hid = int(l[1])
                
                # We have to read these lines, even if we then throw them away
                # if npart < most_bound:
                #     continue
        
                # pids = np.zeros(npart, dtype=int)
                # ptypes = np.zeros(npart, dtype=int)
                data = np.zeros((npart, 2), dtype=np.int64)
            
                for j in range(npart):
                    l = f.readline().strip('\n').split()      
                    data[j, 0] = int(l[0])
                    data[j, 1] = int(l[1])

                # Keep only the DM pids (DM ptype=1)
                ii = data[:, 1] == 1
                pid = data[ii, 0]

                # We do a cut later on, so why bother here?
                if len(pid) >= most_bound:
                    # hids.append(hid)
                    pids[hid] = pid[0:most_bound]

                pids[hid] = pid

    print('-- total haloes read:', nhalo_tot)
    return pids


def read_ahf_haloes(root, iout, most_bound, contam_frac=0.999999,
                    subhaloes=False, bsph=None):
    """Reads in the AHF_particles files and also uses the AHF_halos
    file to do some filtering on the haloes.

    :param root: (str) full path containing the uppermost AHF
        directory (i.e. will be one of the main run directories)
    :param iout: (int) which output to look at
    :param most_bound: (int) how many particles from the centre to
        compare, also acts as a floor for the sizes of haloes that are
        analysed
    :param contam_frac: (float) minimum fraction of high-resolution
        particles in halo (haloes with f_Mhires < contam_frac are
        discarded)
    :param subhaloes: (bool) True: look only at subhaloes, False: look
        only at distinct haloes
    :param bsph: (arr, [xc, yc, zc, rc], optional) bounding sphere
        within which to select haloes, useful for selecting only a
        zoom region, units are AHF code units

    :returns: halo IDs (sorted by mass), particle IDs (halo IDs are
              dictionary keys)

    :rtype: (struct arr, dict of arr)
    """
    path = get_path(root, iout)
    parts = read_ahf_particles(path, most_bound)

    cols = [1, 3, 5, 6, 7, 4, 43, 63, 37]
    fields = ['host_id', 'm_tot', 'x', 'y', 'z', 'np_tot',
              'np_gas', 'np_star', 'f_mhr']

    # Create a dictionary of the fields' columns
    fd = {}
    for i in range(len(cols)):
        fd[fields[i]] = i
    
    haloes = np.loadtxt(path+'AHF_halos', usecols=cols)
    # Problem with int -> float -> int conversion?
    halo_ids = np.loadtxt(path+'AHF_halos', usecols=0, dtype=int)

    # Ignore subhaloes
    ii = haloes[:, fd['host_id']] <= 0
    if subhaloes:
        # Keep only subhaloes
        ii = np.logical_not(ii)
    haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)

    # Only keep uncontaminated haloes
    ii = haloes[:, fd['f_mhr']] > contam_frac
    haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)
    
    # Only keep haloes with most_bound number of DM parts
    np_dm = (haloes[:, fd['np_tot']] - haloes[:, fd['np_star']] -
             haloes[:, fd['np_gas']]) 
    ii = np_dm >= most_bound
    haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)
        
    if bsph is not None:
        # x0 = bbox[0]
        # x1 = bbox[1]
        # if len(bbox) == 2:
        #     y0 = x0
        #     z0 = x0
        #     y1 = x1
        #     z1 = x1
        # else:
        #     y0 = bbox[2]
        #     y1 = bbox[3]
        #     z0 = bbox[4]
        #     z1 = bbox[5]

        # xi = np.logical_and(haloes[:, fd['x']] > x0, haloes[:, fd['x']] < x1)
        # yi = np.logical_and(haloes[:, fd['y']] > y0, haloes[:, fd['y']] < y1)
        # zi = np.logical_and(haloes[:, fd['z']] > z0, haloes[:, fd['z']] < z1)

        pos = np.array([haloes[:, fd['x']],
                        haloes[:, fd['y']],
                        haloes[:, fd['z']]]).T
        cen = bsph[0:3]
        r = bsph[3]

        ii = in_sphere(pos, cen, r)
        
        haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)

    # Finally, sort
    ii = np.argsort(haloes[:, fd['m_tot']])[::-1]
    haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)
    
    hids = np.zeros(haloes.shape[0], dtype=np.dtype([('id', np.int64)]))
    pids = np.zeros(haloes.shape[0], dtype=object)

    for i in range(haloes.shape[0]):
        hids[i] = halo_ids[i]
        pids[i] = parts[halo_ids[i]][0:most_bound]


    dump_params(iout, most_bound, contam_frac, subhaloes, bsph)
        
    return hids, pids


def trim_haloes(haloes, halo_ids, ii):
    """Returns the haloes and halo_ids modified by either a Boolean
    array or a list or an array of indices.

    :param haloes: (arr) haloes array
    :param halo_ids: (arr) halo IDs array
    :param ii: (arr, [int or bool])

    :returns: haloes and halo IDs arrays

    :rtype: (arr, arr)
    """
    haloes = haloes[ii, :]
    halo_ids = halo_ids[ii]

    return haloes, halo_ids


def in_sphere(pos, cen, r):
    """Determines whether a coordinate (a row in pos) is within the
    sphere defined by cen, r.  Returns array of bools of size
    pos.shape[0].

    :param pos: (arr (N, 3)) coordinates
    :param cen: (arr (3)) sphere centre
    :param r: (float) sphere radius

    :returns: whether that point is in the spher

    :rtype: (arr, [bool] (N))
    """
    norm_pos = np.zeros_like(pos)

    # pos - cen
    for i in range(3):
        norm_pos[:, i] = pos[:, i] - cen[i]

    # |pos - cen|
    norm_pos = np.linalg.norm(norm_pos, axis=1)

    return norm_pos < r


def dump_params(iout, most_bound, contam_frac, subhaloes, bsph):
    out_fn = 'match_haloes_{0:d}.param'.format(iout)
    header = '# Matching distinct haloes\n'
    if subhaloes:
        out_fn = 'match_subhaloes_{0:d}.param'.format(iout)
        header = '# Matching subhaloes\n'

    if bsph is None:
        xc = -1
        yc = -1
        zc = -1
        r = -1
    else:
        xc = bsph[0]
        yc = bsph[1]
        zc = bsph[2]
        r = bsph[3]

    with open(out_fn, 'w') as f:
        f.write(header)
        f.write('most_bound {0:d}\n'.format(most_bound))
        f.write('contam_frac {0:.10f}\n'.format(contam_frac))
        f.write('xc {0:.10f}\n'.format(xc))
        f.write('yc {0:.10f}\n'.format(yc))
        f.write('zc {0:.10f}\n'.format(zc))
        f.write('r {0:.10f}\n'.format(r))
        
    
