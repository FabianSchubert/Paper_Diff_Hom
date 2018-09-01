#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.stats import linregress as linreg
from scipy.stats import pearsonr
import scipy.stats as stats

from plot_setting import *

import pdb

def gauss(x,sigm):

	return np.exp(-x**2/(2.*sigm**2))/(2.*np.pi*sigm**2)


M = loadmat("firingrates_and_distances.mat")

D = M["KO_P30P40_distances"]
F = M["KO_P30P40_movies"]

n_measurements = D.shape[0]

f_avg = []
D_mat = []

for k in range(n_measurements):

	f_avg.append(F[k,0].mean(axis=0))
	D_mat.append(D[k,0])

f_avg_append = []
dens_append = []

n_sweep = 100
sigm_sweep_limits = [1.,100.]
sigm_list = np.linspace(sigm_sweep_limits[0],sigm_sweep_limits[1],n_sweep)

for k in range(n_measurements):
	
	f_avg_append.extend(f_avg[k].tolist())

	corr_list = np.ndarray((n_sweep))

	for l in range(n_sweep):

		corr_list[l] = np.corrcoef(f_avg[k],1./gauss(D_mat[k],sigm_list[l]).sum(axis=0))[1,0]

	sigm_max_corr = sigm_list[corr_list.argmax()]

	dens_append.extend(gauss(D_mat[k],sigm_max_corr).sum(axis=0).tolist())

f_avg_append = np.array(f_avg_append)
dens_append = np.array(dens_append)

r_pears = np.corrcoef(1./dens_append,f_avg_append)[1,0]

p_pears = pearsonr(1./dens_append,f_avg_append)

print("pearson corr coef.: " + str(r_pears))
print("p-value: " + str(p_pears))

print("")

r_spearman, p_spearman = stats.spearmanr(1./dens_append,f_avg_append)

print("spearman corr coef.: " + str(r_spearman))
print("p-value: " + str(p_spearman))

fig, ax = plt.subplots(figsize=(4,3))

ax.plot(1./dens_append,f_avg_append,'.')

ax.set_xlabel("1/Neuron Dens. [$\\rm \\mu m^{-2}$]")
ax.set_ylabel("Mean Firing Rate [Hz]")

props = dict(facecolor='white')
#pdb.set_trace()
textstr = "pearson corr. coef.: " + str(round(r_pears,2))  + "\np-value: " + str(round(p_pears[1]*10**28,2)) + "$\\cdot 10^{-28}$" + "\n\nspearman corr. coef.: " + str(round(r_spearman,2))  + "\np-value: " + str(round(p_spearman*10**24,2)) + "$\\cdot 10^{-24}$"

# place a text box in upper left in axes coords
#ax.text(0.55, 0.95, textstr, transform=ax.transAxes, fontsize=10,
#        verticalalignment='top',horizontalalignment='left', bbox=props)


plt.savefig("inv_dens_vs_fir_rates.png",dpi=300)

plt.show()

pdb.set_trace()