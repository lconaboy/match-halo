import time
# import numba
import random
from read_ahf import *
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


def run_ahf_match_two(roots, iouts, bbox=None, most_bound=100):
    root1 = roots[0]
    root2 = roots[1]
    iout1 = iouts[0]
    iout2 = iouts[1]

    print('-- reading in haloes')
    h1, pid1 = read_ahf_haloes(root1, iout1, bbox=bbox, most_bound=most_bound)
    h2, pid2 = read_ahf_haloes(root2, iout2, bbox=bbox, most_bound=most_bound)

    match = match_two([h1, h2], [pid1, pid2])

    # Write matched array to file
    write_match(iout1, match)


def run_ahf_match_three(roots, iouts, bsph=None, most_bound=100,
                        fields=None, subhaloes=False):
    root1 = roots[0]
    root2 = roots[1]
    root3 = roots[2]
    iout1 = iouts[0]
    iout2 = iouts[1]
    iout3 = iouts[2]

    print('-- reading in haloes')
    h1, pid1 = read_ahf_haloes(root1, iout1, bsph=bsph, most_bound=most_bound,
                               subhaloes=subhaloes)
    h2, pid2 = read_ahf_haloes(root2, iout2, bsph=bsph, most_bound=most_bound,
                               subhaloes=subhaloes)
    h3, pid3 = read_ahf_haloes(root3, iout3, bsph=bsph, most_bound=most_bound,
                               subhaloes=subhaloes)

    # Run two match_twos then compare them
    match2, matchf2 = match_two([h1, h2], [pid1, pid2])  # match between sim 1 and 2
    match3, matchf3 = match_two([h1, h3], [pid1, pid3])  # match between sim 1 and 3

    # Check whether we actually found any matches
    if (len(match2) < 1) or (len(match3) < 1):
        print('-- some simulations found no matches, finishing')
        sys.exit()

    
    max_match = np.max([match2.max(), match3.max()])

    # Find the indices of match 2 that are in match 3
    ii2 = np.isin(match2[:, 0], match3[:, 0])
    # vice versa
    ii3 = np.isin(match3[:, 0], match2[:, 0])

    # Pick out these values
    match2_new = match2[ii2, :]
    matchf2_new = matchf2[ii2]
    match3_new = match3[ii3, :]
    matchf3_new = matchf3[ii3]

    # Sort on sim 1 IDs (first column for both is sim 1)
    ii2 = np.argsort(match2_new[:, 0])
    ii3 = np.argsort(match3_new[:, 0])
    match2_new = match2_new[ii2, :]
    matchf2_new = matchf2_new[ii2]
    match3_new = match3_new[ii3, :]
    matchf3_new = matchf3_new[ii3]

    match = np.array([match2_new[:, 0], match2_new[:, 1],
                      match3_new[:, 1]], dtype=int).T
    matchf = np.min(np.array([matchf2_new, matchf3_new]).T, axis=1)
    
    # Write matched array to file
    write_match_three(iout=iout1, match=match, match_frac=matchf,
                      fields=fields, subhaloes=subhaloes)

    
def match_two(hs, pids):
    h1 = hs[0]
    h2 = hs[1]
    pid1 = pids[0]
    pid2 = pids[1]
    
    match = np.zeros((np.min([pid1.shape[0], pid2.shape[0]]), 2), dtype=int)
    matchf = np.zeros((np.min([pid1.shape[0], pid2.shape[0]])), dtype=float)
    k = 0

    ni = pid1.shape[0]
    nj = pid2.shape[0]

    for i in range(ni):
        print('---- working on halo {0:d}/{1:d}'.format(i+1, ni))
        np1 = pid1[i].shape[0]

        for j in range(nj):
            # If we've matched this halo, we don't need to check it again
            # if h2[j]['id'] in match[0:k, 1]:
            #     continue
            if np.any(np.isin(h2[j]['id'], match[0:k, 1], assume_unique=True)):
                continue
        
            np2 = pid2[j].shape[0]
            
            # Q: I wonder if this numpy function is quicker than
            # looping through particles? A: yes, about three times
            # quicker!
            nm = np.sum(np.isin(pid1[i], pid2[j], assume_unique=True),
                        dtype=int)

            # Calculate the matched fraction for this halo, use the
            # largest num particles in order to get the smallest
            # matched fraction
            mf = nm / max(np1, np2)
            
            # if mf > 0.0:
            #     print('mf greater than 0!', mf, nm, np1, np2)

            # print('mf:', mf)

            if mf > match_frac:
                print('------ found a match!')
                match[k, 0] = h1[i]['id']
                match[k, 1] = h2[j]['id']
                matchf[k] = mf
                k += 1

                break

    # Trim match array
    match = match[0:k, :]
    matchf = matchf[0:k]

    return match, matchf


if __name__ == '__main__':
    iout = 25
    root = '/snap7/scratch/dp004/dc-cona1/bd/runs/v50_512_32768/'
    fields = ['same_vtf', 'unbiased', 'biased']

    roots = [root + f + '/' for f in fields]
    iouts = [iout, iout, iout]


    s = time.time()
    run_ahf_match_three(roots, iouts, fields=fields, most_bound=100)
                        #bbox=[50000-500, 50000+500])
    f = time.time()

    print('Took', f-s, 'seconds')
