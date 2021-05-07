"""
The format of the output file is each row is the ID of matched halo,
with each column being the different simulations.  The final column is
always the minimum maatched fraction for that halo.  The header of the
output file should have the form

# <match1> <match2> ... <matchN>

/without/ <match_frac> on the end.  <matchi> should be the identifier
of the run, i.e. same_vtf.
"""

import numpy as np

def get_dtype_match3(fields):
    dtype = [(f, 'i8') for f in fields]
    dtype.append(('match_frac', 'f4'))
    dtype_match3 = np.dtype(dtype)

    return dtype_match3


def write_match(iout, match):
    """Writes matched array to an ASCII file.

    :param iout: (int) output number
    :param match: (arr) matched outputs, first column is halo ID,
        second column is halo ID and final column is match fraction

    :returns: None

    :rtype: (NoneType)
    """
    
    np.savetxt('match_{0:d}.list'.format(iout), match, fmt='%d %d %.4f')


def write_match_three(iout, match, match_frac, fields=None,
                      subhaloes=False):
    """Writes matched array to an ASCII file.

    :param iout: (int) output number
    :param match: (arr) matched outputs, first column is halo ID,
        second column is halo ID, third column is halo ID and final
        column is match fraction
    :param subhaloes: (bool) are we looking at just haloes (False) or
        just subhaloes (True)?
w
    :returns: None

    :rtype: (NoneType)
    """

    if fields is None:
        fields = ['match1', 'match2', 'match3']

    header = ' '.join(fields)
    dtype_match3 = get_dtype_match3(fields)
    
    out = np.zeros((match_frac.shape[0]), dtype=dtype_match3)
    for i, f in enumerate(fields):
        out[f] = match[:, i]
        # out[f] = match[:, i]
        # out[f] = match[:, i]
    out['match_frac'] = match_frac

    match_fn = get_match_fn(iout, subhaloes)
    np.savetxt(match_fn, out, fmt='%d %d %d %.4f',
               header=header)


def read_match(iout):
    """Reads matched array from ASCII file.

    :param iout: (int) output number

    :returns: matched outputs, first column is halo ID, second
        column is halo ID and final column is match fraction

    :rtype: (arr)
    """
    

    match = np.genfromtxt('match_{0:d}.list'.format(iout))

    return match


def read_match3(iout, subhaloes):
    """Reads matched array from ASCII file.

    :param iout: (int) output number
    :param subhaloes: (bool) are we looking at just haloes (False) or
        just subhaloes (True)?

    :returns: matched outputs, first column is halo ID, second
        column is halo ID and final column is match fraction

    :rtype: (struct arr)
    """

    # First regenerate the field names from the header
    match_fn = get_match_fn(iout, subhaloes)
    fields = read_header(iout, subhaloes)
    dtype_match3 = get_dtype_match3(fields)
    # Now load the data
    match = np.genfromtxt(match_fn, dtype=dtype_match3)

    return match


def read_header(iout, subhaloes):
    match_fn = get_match_fn(iout, subhaloes)
    with open(match_fn, 'r') as f:
        header = f.readline()
        
    fields = header[1:].strip('\n').split()

    return fields


def get_match_fn(iout, subhaloes=False):
    obj = 'haloes'
    if subhaloes: obj = 'subhaloes'
    return 'match_{0}_{1:d}.list'.format(obj, iout)
