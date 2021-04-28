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
        print('-- [error] all_{0:03d}.AHF_particles not found')
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


def read_ahf_haloes(root, iout, most_bound, bbox=None):
    """Reads in the AHF_particles files and also uses the AHF_halos
    file to do some filtering on the haloes.

    :param root: (str) full path containing the uppermost AHF
        directory (i.e. will be one of the main run directories)
    :param iout: (int) which output to look at
    :param most_bound: (int) how many particles from the centre to
        compare, also acts as a floor for the sizes of haloes that are
        analysed
    :param bbox: (arr, [x0, x1, <y1, y0, z1, z0>]) bounding box within
        which to select haloes, useful for selecting only a zoom
        region though AHF has the fM_hires parameter which is probably
        for selecting uncontaminated haloes

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
    haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)

    # Only keep uncontaminated haloes
    ii = haloes[:, fd['f_mhr']] > 0.999999
    haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)
    
    # Only keep haloes with most_bound number of DM parts
    np_dm = (haloes[:, fd['np_tot']] - haloes[:, fd['np_star']] -
             haloes[:, fd['np_gas']]) 
    ii = np_dm >= most_bound
    haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)
        
    if bbox is not None:
        x0 = bbox[0]
        x1 = bbox[1]
        if len(bbox) == 2:
            y0 = x0
            z0 = x0
            y1 = x1
            z1 = x1
        else:
            y0 = bbox[2]
            y1 = bbox[3]
            z0 = bbox[4]
            z1 = bbox[5]

        xi = np.logical_and(haloes[:, fd['x']] > x0, haloes[:, fd['x']] < x1)
        yi = np.logical_and(haloes[:, fd['y']] > y0, haloes[:, fd['y']] < y1)
        zi = np.logical_and(haloes[:, fd['z']] > z0, haloes[:, fd['z']] < z1)

        ii = np.logical_and(xi, yi)
        ii = np.logical_and(ii, zi)

        haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)

    # Finally, sort
    ii = np.argsort(haloes[:, fd['m_tot']])[::-1]
    haloes, halo_ids = trim_haloes(haloes, halo_ids, ii)
    
    hids = np.zeros(haloes.shape[0], dtype=np.dtype([('id', np.int64)]))
    pids = np.zeros(haloes.shape[0], dtype=object)

    for i in range(haloes.shape[0]):
        hids[i] = halo_ids[i]
        pids[i] = parts[halo_ids[i]][0:most_bound]

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
