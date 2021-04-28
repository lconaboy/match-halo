from read_match import *
import sys
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic

sys.path.insert(0, '/cosma/home/dp004/dc-cona1/drft-pp/')
from lines import *

iout = 25
root = '/snap7/scratch/dp004/dc-cona1/bd/runs/v50_512_32768/'
fmt = '{0}/AHF/{1:03d}/halos/all_{1:03d}.AHF_halos'
fields = read_header(iout)
matches = read_match3(iout)
fb = {}
mvir = {}
mdm = {}
rvir = {}
pos = {}

for f in fields:
    path = root + fmt.format(f, iout)
    hids = np.loadtxt(path, usecols=0, dtype=int)
    mtot, x, y, z, rv, mgas, mstar = np.loadtxt(path,
                                                  usecols=(3, 5, 6, 7,
                                                           11, 44, 64),
                                                  unpack=True)
    fb[f] = np.zeros(matches[f].shape[0], dtype=float)
    mvir[f] = np.zeros(matches[f].shape[0], dtype=float)
    mdm[f] = np.zeros(matches[f].shape[0], dtype=float)
    rvir[f] = np.zeros(matches[f].shape[0], dtype=float)
    pos[f] = np.zeros((matches[f].shape[0], 3), dtype=float)
    
    for i, match in enumerate(matches[f]):
        ii = hids == match
        fb[f][i] = (mgas[ii] + mstar[ii]) / mtot[ii]
        mvir[f][i] =  mtot[ii]
        mdm[f][i] = mtot[ii] - mgas[ii] - mstar[ii]
        pos[f][i, :] = [x[ii], y[ii], z[ii]]
        rvir[f][i] = rv[ii]
        
plt.figure(figsize=(6, 6))
bins = 10.**np.linspace(np.log10(2e5), np.log10(2e7), num=11)
for f in ['unbiased', 'biased']:
    x = mvir[f]
    y = fb[f]-fb['same_vtf']
    xb = 0.5 * (bins[1:] + bins[:-1])
    yb, _, _ = binned_statistic(x, y, statistic='mean', bins=bins)
    plt.scatter(x, y, marker='.', color=c_dict[f], label=f)
    plt.semilogx(xb, yb, c=c_dict[f], ls=ls_dict[f], label='__none__')
plt.xscale('log')
plt.legend()
plt.savefig('fb_comp.png', bbox_inches='tight', dpi=600)

# Plot m_dm
def plot_m(m, qty):
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    axs = axs.ravel()
    for ax, f in zip(axs, ['unbiased', 'biased']):
        ax.plot(np.log10(m['same_vtf']), np.log10(m[f]), 'k.')
        mima = [np.log10(np.min([m[f], m['same_vtf']])),
                np.log10(np.max([m[f], m['same_vtf']]))]
        ax.set_xlabel('log$_{{10}}$('+qty+') '+'same\_vtf')
        ax.set_ylabel('log$_{{10}}$('+qty+') '+f)
        ax.set_xlim(mima)
        ax.set_ylim(mima)
        ax.plot(mima, mima, c='r')

    fig.savefig(qty+'_comp.pdf', bbox_inches='tight')


def plot_pos(m, qty):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(12, 18))
    xyz = 'xyz'
    for i, f in enumerate(['unbiased', 'biased']):
        for j in range(3):
            axs[j, i].plot(np.log10(m[f][:, j]), np.log10(m['same_vtf'][:, j]),
                           'k.', label=xyz[j])
            mima = [np.log10(np.min([m[f][:, j], m['same_vtf'][:, j]])),
                    np.log10(np.max([m[f][:, j], m['same_vtf'][:, j]]))]
            axs[j, i].set_xlim(mima)
            axs[j, i].set_ylim(mima)
            axs[j, i].plot(mima, mima, c='r')
            axs[j, i].set_ylabel('log$_{{10}}$('+xyz[j]+') '+f)
            axs[j, i].set_xlabel('log$_{{10}}$('+xyz[j]+') '+'same\_vtf')
            axs[j, i].set_xlabel('log$_{{10}}$('+xyz[j]+') '+'same\_vtf')
            axs[j, i].legend()
    
    fig.savefig(qty+'_comp.pdf', bbox_inches='tight')

plot_m(mdm, 'mdm')
plot_m(rvir, 'rvir')
plot_pos(pos, 'pos')
