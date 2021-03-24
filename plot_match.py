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


# Plot mass
fig, ax = plt.subplots(figsize=(7, 6))
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
fig.savefig('match_test_m.pdf', bbox_inches='tight')

# Plot positions
# xyz_str = 'xyz'
def plot_pos(inds1, inds2, p='x'):
    if p == 'x':
        xyz_str = ['x', 'y', 'z']
        i0 = 0  # offset in pos array
        units = ' (h$^{{-1}}$ Mpc)'
        label = 's'
    else:
        xyz_str = ['vx', 'vy', 'vz']
        i0 = 3
        units = ' (km/s)'
        label = 'v'
    
    s = np.zeros((mf.shape[0], 2))  # magnitude
    for ii, ix in enumerate(xyz_str):
        fig, ax = plt.subplots(figsize=(7, 6))
        xy = np.zeros((mf.shape[0], 2))
        for i in range(mf.shape[0]):
            xy[i] = [inds1[i]['pos'][0][ii+i0], inds2[i]['pos'][0][ii+i0]]
            s[i] += xy[i] ** 2.
        #xy = np.abs(xy)
        xl = [np.min(xy), np.max(xy)]
        ax.plot(xl, xl, c='k')
        sc = ax.scatter(xy[:, 0], xy[:, 1], cmap='viridis', c=mf)
        cb = fig.colorbar(mappable=sc)
        ax.set_xlabel(ix+units)
        ax.set_ylabel(ix+units)
        ax.set_xlim(xl)
        ax.set_ylim(xl)
        cb.set_label('matched fraction')
        fig.savefig('match_test_'+ix+'.pdf', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(7, 6))
    xy = np.sqrt(s)
    ix = label
    xl = [np.min(xy), np.max(xy)]
    ax.plot(xl, xl, c='k')
    sc = ax.scatter(xy[:, 0], xy[:, 1], cmap='viridis', c=mf)
    cb = fig.colorbar(mappable=sc)
    ax.set_xlabel(ix+'$_1$' + units)
    ax.set_ylabel(ix+'$_2$' + units)
    ax.set_xlim(xl)
    ax.set_ylim(xl)
    cb.set_label('matched fraction')
    fig.savefig('match_test_'+ix+'.pdf', bbox_inches='tight')


plot_pos(inds1, inds2, p='x')
plot_pos(inds1, inds2, p='v')
