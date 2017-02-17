import numpy as np
import cPickle as pickle
import matplotlib.pyplot as plt
from frequ_from_spikes import *
from plot_setting import *
import sys

files_diff = []
files_non_diff = []
files_instant_diff = []

for k in xrange(10):
	files_diff.append(sim_data_base_folder + "spiketime_dataset_diff/set_"+str(k)+"/")
	files_non_diff.append(sim_data_base_folder + "spiketime_dataset_non_diff/set_"+str(k)+"/")


savefold = plots_base_folder + "firing_rate_dists/"

n_bins = 40

plot_pop = sys.argv[1]#"exc"

if plot_pop == "exc":
	#### excitatory
	regular_bin_range = [0.,8.]
	log_bin_range = [-.3,1.1]
	y_lim_reg = [0.,.7]
	y_lim_log = [0.,4.]
	title_reg = "A"
	title_log = "B"
	plot_filename = "fir_rate_dist_e_compare"
	filename = "spiketimes_e.p"
elif plot_pop == "inh":
	#### inhibitory
	regular_bin_range = [0.,25.]
	log_bin_range = [0.,1.5]
	y_lim_reg = [0.,.3]
	y_lim_log = [0.,3.]
	title_reg = "A*"
	title_log = "B*"
	plot_filename = "fir_rate_dist_i_compare"
	filename = "spiketimes_i.p"
else:
	print "Wrong Population Argument"
	sys.exit()




def plot_dist(files,filename,t_range,bins_hist,plot_type,label_hist,ax):

	spt_sets = []
	spt_sets_joint = []


	for k in xrange(len(files)):
	
		spt_sets.append(pickle.load(open(files[k]+filename)).values())
		spt_sets_joint.extend(spt_sets[-1])


	n_total = len(spt_sets_joint)
	if plot_type == "log":
		f_arr = np.log10(frequ_vec(spt_sets_joint,t_range[0],t_range[1]))
	elif plot_type == "regular":
		f_arr = frequ_vec(spt_sets_joint,t_range[0],t_range[1])
	else:
		print("Error - Wrong plot type argument")
	skew = (((f_arr-f_arr.mean())/f_arr.std())**3.).mean()
	print(label_hist + " " + plot_type + " plot, skewness: " + str(skew))
	
	ax.hist(f_arr,bins=bins_hist,normed=True,histtype="step",label=label_hist)

fig, ax = plt.subplots(1,2,figsize=(default_fig_width,default_fig_width*0.4))

bins_h_regular = np.linspace(regular_bin_range[0],regular_bin_range[1],n_bins+1)
bins_h_log = np.linspace(log_bin_range[0],log_bin_range[1],n_bins+1)

plot_dist(files_non_diff,filename,[1000.,1500.],bins_h_regular,"regular","Non-Diff. H.",ax[0])
plot_dist(files_diff,filename,[1000.,1500.],bins_h_regular,"regular","Diff. H.",ax[0])
if plot_pop == "exc":
	ax[0].legend()
ax[0].set_title(title_reg,loc="left")
ax[0].set_ylim(y_lim_reg)
ax[0].set_xlim(regular_bin_range[0],regular_bin_range[1])
ax[0].set_xlabel("f [Hz]")
ax[0].set_ylabel("Prob. Dens.")

plot_dist(files_non_diff,filename,[1000.,1500.],bins_h_log,"log","Non-Diff. H.",ax[1])
plot_dist(files_diff,filename,[1000.,1500.],bins_h_log,"log","Diff. H.",ax[1])
#ax[1].legend()
ax[1].set_title(title_log,loc="left")
ax[1].set_ylim(y_lim_log)
ax[1].set_xlim(log_bin_range[0],log_bin_range[1])
ax[1].set_xlabel(r"$\mathrm{log_{10}(f) \, [log_{10}(Hz)]}$")
ax[1].set_ylabel("Prob. Dens.")




for format in [".png",".svg",".eps"]:
	plt.savefig(savefold+plot_filename+format)

plt.show()



#import pdb
#pdb.set_trace()






