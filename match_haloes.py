import time
import numba
import random
from read_match import *
from read_rockstar import *

match_frac = 0.5
match_num = 50

def run_binary_match_two(roots, iouts):
    root1 = roots[0]
    root2 = roots[1]
    iout1 = iouts[0]
    iout2 = iouts[1]

    print('-- reading in haloes')
    h1, pid1, header1 = read_binary_haloes(root1, iout1)
    h2, pid2, header2 = read_binary_haloes(root2, iout2)

    match = match_two([h1, h2], [pid1, pid2])

    # Write matched array to file
    write_match(iout1, match)


    
def run_ascii_match_two(roots, iouts, most_bound=50):
    root1 = roots[0]
    root2 = roots[1]
    iout1 = iouts[0]
    iout2 = iouts[1]

    print('-- reading in haloes')
    h1, pid1 = read_particle_haloes(root1, iout1, most_bound=most_bound)
    h2, pid2 = read_particle_haloes(root2, iout2, most_bound=most_bound)

    match = match_two([h1, h2], [pid1, pid2])

    # Write matched array to file
    write_match(iout1, match)

    

def match_two(hs, pids):
    h1 = hs[0]
    h2 = hs[1]
    pid1 = pids[0]
    pid2 = pids[1]
    
    match = np.zeros((np.min([pid1.shape[0], pid2.shape[0]]), 3))
    k = 0

    ni = pid1.shape[0]
    nj = pid2.shape[0]

    for i in range(ni):
        print('---- working on halo {0:d}/{1:d}'.format(i+1, ni))
        np1 = pid1[i].shape[0]

        for j in range(nj):
            # If we've matched this halo, we don't need to check it again
            if h2[j]['id'] in match[0:k, 1]:
                continue
        
            # How does this work on a random subset of the particle IDs?
            np2 = pid2[j].shape[0]
            # nm = 0
            # for jj in range(np2):
            #     if pid2[j][jj] in pid1[i]:
            #         nm += 1
            
            # Q: I wonder if this numpy function is quicker? A: yes,
            # about three times quicker!
            nm = np.sum(np.isin(pid1[i], pid2[j], assume_unique=True),
                        dtype=int)
        
            # Calculate the matched fraction for this halo, use the
            # largest num particles in order to get the smallest
            # matched fraction
            mf = nm / max(np1, np2)
        
            if mf > match_frac:
                print('---- found a match!')
                match[k, 0] = h1[i]['id']
                match[k, 1] = h2[j]['id']
                match[k, 2] = mf
                k += 1

                break

    # Trim match array
    match = match[0:k, :]

    return match


if __name__ == '__main__':
    iout1 = 13
    root1 = './test_data/biased/'

    iout2 = 13
    root2 = './test_data/same_vtf/'

    roots = [root1, root2]
    iouts = [iout1, iout2]

    s = time.time()
    run_ascii_match_two(roots, iouts)
    f = time.time()

    print('Took', f-s, 'seconds')
