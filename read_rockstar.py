import os
import glob
from rockstar_structs import *


def read_header(f):
    """Helper function to read the headers of rockstar binary files.

    :param f: (file) open file object
    :returns: header
    :rtype: (arr [head_struct])

    """
    # magic = np.fromfile(f, dtype=np.uint64, count=1)[0]
    # snap = np.fromfile(f, dtype=np.int64, count=1)[0]
    # chunk = np.fromfile(f, dtype=np.int64, count=1)[0]
    # scale = np.fromfile(f, dtype=np.float32, count=1)[0]
    # Om = np.fromfile(f, dtype=np.float32, count=1)[0]
    # Ol = np.fromfile(f, dtype=np.float32=1)[0]
    # h0 = np.fromfile(f, dtype=np.float32, count=1)[0]
    # bounds = np.fromfile(f, dtype=np.float32, count=6)
    # num_halos = np.fromfile(f, dtype=np.int64, count=1)[0]
    # num_parts = np.fromfile(f, dtype=np.int64, count=1)[0]
    # box_size = np.fromfile(f, dtype=np.float32, count=1)[0]
    # part_mass = np.fromfile(f, dtype=np.float32, count=1)[0]
    # part_type = np.fromfile(f, dtype=np.int64, count=1)[0]

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


def read_binary_haloes(root, iout, sort_key='m'):
    """Reads in all of the halo_<iout>.*.bin files produced by
    rockstar.  Extracts the header information (defined in
    io/io_internal.h) halo objects (defined in halo.h) and particle
    IDs for each halo.  Also returns the header for the final object
    read, so don't trust the file-specific values in there, useful for
    cosmological parameters.

    Example
    -------

    iout = 13
    root = './test_data/biased/'

    h, pid = read_binary_haloes(root, iout)

    for ih in h:
       print(ih['m'])


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
            cur_haloes = np.empty(header['num_halos'], dtype='object')
            cur_part_ids = np.empty(header['num_halos'], dtype='object')
            cur_sort_vals = np.empty(header['num_halos'], dtype='object')
            
            for i in range(header['num_halos'][0]):
                cur_haloes[i] = read_halo(f)
                cur_sort_vals[i] = cur_haloes[i][sort_key]
                
            for i in range(header['num_halos'][0]):
                cur_part_ids[i] = read_ids(f, cur_haloes[i]['num_p'][0])
                
            # Are IDs unique? This probably isn't that useful, since
            # IDs could also be duplicated outside this proccessor.
            assert(len(np.unique(cur_part_ids[i])) == len(cur_part_ids[i])), '-- error: particle IDs not unique in this halo'

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
    
    return haloes[inds[::-1]], part_ids[inds[::-1]], header
