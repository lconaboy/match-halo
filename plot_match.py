from read_match import *
from read_rockstar import *
import matplotlib.pyplot as plt

iout1 = 13
root1 = './test_data/biased/'
iout2 = 13
root2 = './test_data/same_vtf/'
print('-- reading in haloes')
h1, pid1, header1 = read_binary_haloes(root1, iout1)
h2, pid2, header2 = read_binary_haloes(root2, iout2)

m = read_match(iout1)
m1 = m[:, 0].astype(int)
m2 = m[:, 1].astype(int)
mf = m[:, 2]

inds1 = np.zeros(mf.shape[0], dtype=object)
inds2 = np.zeros(mf.shape[0], dtype=object)

j = 0
for i in range(h1.shape[0]):
    if h1[i]['id'] in m1:
        inds1[j] = h1[i]
        j += 1
        
j = 0
for i in range(h2.shape[0]):
    if h2[i]['id'] in m2:
        inds2[j] = h2[i]     
        j += 1
        
fig, ax = plt.subplots(figsize=(6.5, 6))
xy = np.zeros((mf.shape[0], 2))
for i in range(mf.shape[0]):
    xy[i] = [inds1[i]['m'], inds2[i]['m']]
xl = [np.min(xy), np.max(xy)]
ax.loglog(xl, xl, c='k')
sc = ax.scatter(xy[:, 0], xy[:, 1], cmap='viridis', c=mf)
cb = fig.colorbar(mappable=sc)
ax.set_xlabel('M$_{{1}}$ (h$^{{-1}}$ M$_\\odot$)')
ax.set_ylabel('M$_{{2}}$ (h$^{{-1}}$ M$_\\odot$)')
ax.set_xlim(xl)
ax.set_ylim(xl)
cb.set_label('matched fraction')
fig.savefig('match_test.pdf', bbox_inches='tight')
