from read_rockstar import *

match_frac = 0.5
match_num = 50

iout1 = 13
root1 = './test_data/biased/'

iout2 = 13
root2 = './test_data/same_vtf/'

print('-- reading in haloes')
h1, pid1, header1 = read_binary_haloes(root1, iout1)
h2, pid2, header2 = read_binary_haloes(root2, iout2)

match = np.zeros((min(pid1.shape[0], pid2.shape[0]), 3))
k = 0

ni = 100 # pid1.shape[0]
nj = pid2.shape[0]

for i in range(ni):
    print('---- working on halo {0:d}/{1:d}'.format(i+1, ni))
    np1 = pid1[i].shape[0]

    for j in range(nj):
        # If we've matched this halo, we don't need to check it again
        if h2[j]['id'] in match[0:k, 1]:
            continue
        
        nm = 0
        np2 = pid2[j].shape[0]
        for jj in range(np2):
            if pid2[j][jj] in pid1[i]:
                nm += 1

        # Calculate the matched fraction for this halo, use the
        # largest num particles in order to get the smallest matched
        # fraction
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

# Write matched array to file
np.savetxt('match_{0:d}.list'.format(iout1), match, fmt='%d %d %.4f')
