import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from scipy.optimize import curve_fit
from plot_setting import *

savefolder = plots_base_folder + "paper/"
plot_filename = "syn_weight_dist"

def gauss(x,A,mu,sigm):
	
	return A*np.exp(-(x-mu)**2/(2.*sigm**2))/(np.sqrt(2.*np.pi)*sigm)


def plot_syn_weights(filename,index_t,bin_range,n_bins,label_w,color,ax):
	
	W = np.load(filename)
	
	N = W.shape[1]
	
	#W_cut = W[index_t*N:(index_t+1)*N,:].toarray()
	
	ind_syn = np.where(W > 0)
	
	w = W[ind_syn[0],ind_syn[1]]
	
	bins_w = np.linspace(bin_range[0],bin_range[1],n_bins+1)
	
	h = np.histogram(np.log10(w*1000),bins=bins_w)
	
	#pdb.set_trace()
	
	x = (h[1][1:] + h[1][:-1])/2. # mid-binning
	dx = x[1]-x[0]
	
	y = h[0]/(h[0].sum()*dx)
	
	fit,err = curve_fit(gauss,x,y,[1.,0.,1.])
	print("A: "+ str(fit[0]) + "\n" + "mu: " + str(fit[1]) + "\n" + "sigm: " + str(fit[2]))
	
	x_fine = np.linspace(bin_range[0],bin_range[1],1000)
	
	h_p, = ax.plot(x,y,'.',label=label_w,c=color)
	fit_p, = ax.plot(x_fine,gauss(x_fine,fit[0],fit[1],fit[2]),c=color)
	

fig, ax = plt.subplots(figsize=(default_fig_width,default_fig_width*0.4))

plot_syn_weights(sim_data_base_folder + "complete_diff_long/W_eTOe.npy",999,[-3.,1.5],30,"Diffusive H.",mpl.rcParams['axes.color_cycle'][0],ax)
plot_syn_weights(sim_data_base_folder + "complete_non_diff_long/W_eTOe.npy",999,[-3.,1.5],30,"Non-Diffusive H.",mpl.rcParams['axes.color_cycle'][1],ax)

ax.set_xlabel(r"$\mathrm{log_{10}(Weight) \, [log_{10}(mV)]}$")
ax.set_ylabel("Prob. Dens.")
ax.legend(loc=2)

for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)

plt.show()
