import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import numpy as np
import cPickle as pickle
from plot_setting import *
#plt.style.use('ggplot')

savefolder = plots_base_folder + "paper/"
plot_filename = "RC_CFB_plot"

#0:diff + topol
#1:non-diff + topol
#2:diff + no topol
#3:non-diff + no topol

source = ["CF_dataset/RC_CFB_eTOe.csv",
			"CF_dataset_non_diff/RC_CFB_eTOe.csv",
			"CF_dataset_no_topology/RC_CFB_eTOe.csv",
			"CF_dataset_non_diff_no_topology/RC_CFB_eTOe.csv"]
labels = ["Diffusive Homeostasis, topology",
			"Non-Diffusive Homeostasis, topology",
			"Diffusive Homeostasis, no topology",
			"Non-Diffusive Homeostasis, no topology"]
RC_CFB = []
import pdb
for k in xrange(4):
	rc_cfb = []
	with open(sim_data_base_folder+source[k],"rb") as csvfile:
		reader = csv.reader(csvfile)
		for row in reader:
			rc_cfb.append(np.array(row).astype("float"))
	rc_cfb = np.array(rc_cfb).T
	#pdb.set_trace()
	rc_cfb_m = rc_cfb.mean(axis=1)
	rc_cfb_err = rc_cfb.std(axis=1)/np.sqrt(rc_cfb.shape[1])

	RC_CFB.append([rc_cfb_m,rc_cfb_err])

t = range(RC_CFB[0][0].shape[0])

fig = plt.figure(figsize=(default_fig_width*0.5,default_fig_width*0.5))

for k in xrange(4):
	plt.fill_between(t,RC_CFB[k][0]-RC_CFB[k][1],RC_CFB[k][0]+RC_CFB[k][1],color=mpl.rcParams['axes.color_cycle'][k])
	plt.plot(t,RC_CFB[k][0],c=mpl.rcParams['axes.color_cycle'][k],label=labels[k],lw=0.5)



plt.xlabel("t [s]")
plt.ylabel("Ratio/Chance of Bidirectional Connections")
plt.xlim([t[0],t[-1]])
plt.ylim([0.,4.])
#plt.legend()

for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)


plt.show()

import pdb
pdb.set_trace()
