import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import numpy as np
import cPickle as pickle
from plot_setting import *
#plt.style.use('ggplot')

savefolder = plots_base_folder + "paper/"
plot_filename = "CF_plot"

#0:diff + topol
#1:non-diff + topol
#2:diff + no topol
#3:non-diff + no topol

source = ["CF_dataset/CF_eTOe.csv",
			"CF_dataset_non_diff/CF_eTOe.csv",
			"CF_dataset_no_topology/CF_eTOe.csv",
			"CF_dataset_non_diff_no_topology/CF_eTOe.csv"]
labels = ["Diffusive Homeostasis, topology",
			"Non-Diffusive Homeostasis, topology",
			"Diffusive Homeostasis, no topology",
			"Non-Diffusive Homeostasis, no topology"]
CF = []
import pdb
for k in xrange(4):
	cf = []
	with open(sim_data_base_folder+source[k],"rb") as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			cf.append(np.array(row).astype("float"))
	cf = np.array(cf).T
	#pdb.set_trace()
	cf_m = cf.mean(axis=1)
	cf_err = cf.std(axis=1)/np.sqrt(cf.shape[1])

	CF.append([cf_m,cf_err])

t = range(CF[0][0].shape[0])

fig = plt.figure(figsize=(default_fig_width*0.5,default_fig_width*0.3))

for k in xrange(4):
	plt.fill_between(t,CF[k][0]-CF[k][1],CF[k][0]+CF[k][1],color=mpl.rcParams['axes.color_cycle'][k])
	plt.plot(t,CF[k][0],c=mpl.rcParams['axes.color_cycle'][k],label=labels[k],lw=0.5)



plt.xlabel("t [s]")
plt.ylabel("Connection Fraction")
plt.xlim([t[0],t[-1]])
plt.ylim([0.,0.13])
#plt.legend(loc=4)

for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)


plt.show()

import pdb
pdb.set_trace()
