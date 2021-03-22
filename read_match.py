import numpy as np

def write_match(iout, match):
    """Writes matched array to an ASCII file.

    :param iout: (int) output number
    :param match: (arr) matched outputs, first column is halo ID,
        second column is halo ID and final column is match fraction

    :returns: None

    :rtype: (NoneType)
    """
    
    np.savetxt('match_{0:d}.list'.format(iout), match, fmt='%d %d %.4f')


def read_match(iout):
    """Reads matched array from ASCII file.

    :param iout: (int) output number

    :returns: matched outputs, first column is halo ID, second
        column is halo ID and final column is match fraction

    :rtype: (arr)
    """
    
    # match = np.loadtxt('match_{0:d}.list'.format(iout), fmt='%d %d %.4f')
    match = np.genfromtxt('match_{0:d}.list'.format(iout))
#                          dtype=[('id1', int), ('id2', int),
#                                 ('match_frac', float)])
    return match
