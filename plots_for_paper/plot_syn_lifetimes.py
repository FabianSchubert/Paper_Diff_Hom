import numpy as np
import cPickle as pickle
#import matplotlib as mpl
#mpl.use("Agg")
import matplotlib.pyplot as plt
import pdb
from plot_setting import *

savefolder = plots_base_folder + "paper/"
plot_filename = "syn_lifetimes"

def analyze_and_plot(filename,pltlabel,ax):
	
	## case 1: ...[0,0]...
	case1=np.zeros((2,400,400))
	## case 2: ...[0,1]...
	case2=np.zeros((2,400,400))
	case2[1,:,:]=1
	## case 3: ...[1,1]...
	case3=np.ones((2,400,400))
	## case 4: ...[1,0]...
	case4=np.zeros((2,400,400))
	case4[0,:,:]=1


	W=np.load(filename)
	
	W_bin = 1.*(W!=0)
	W_bin=np.append(np.zeros((1,400,400)),W_bin,axis=0)

	lifetimes=np.array([])
	
	count = np.zeros((400,400))
	
	ones=[]
	
	for k in range(750,1501):
		
		#pdb.set_trace()
		
		#print k
		
		count+=  (W_bin[[k-1,k],:,:]==case2).prod(axis=0) + (W_bin[[k-1,k],:,:]==case3).prod(axis=0)
		
		
		lifetimes = np.append(lifetimes,count[np.where((W_bin[[k-1,k],:,:]==case4).prod(axis=0)==1)])
		
		#ones.append((count[np.where((W_bin[[k-1,k],:,:]==case4).prod(axis=0)==1)]==1).sum())
		
		count[np.where((W_bin[[k-1,k],:,:]==case4).prod(axis=0)==1)] = 0
		
	n=np.histogram(lifetimes,bins=np.linspace(1,750,101))
	x = (n[1][0:-1]+n[1][1:])/2.
	y=n[0]
	fit_ind_max = 20
	fit = np.polyfit(np.log10(x[:fit_ind_max]),np.log10(y[:fit_ind_max]),deg=1)
	print("Slope " + pltlabel + ": " + str(fit[0]))
	#plt.plot(ones)
	datplot = ax.plot(x,y,'.',label=pltlabel)
	#pdb.set_trace()
	col = datplot.get_color()
	pdb.set_trace()
	fitplot = ax.plot(np.array([x[0],x[-1]]),10**(fit[0]*np.log10(np.array([x[0],x[-1]]))+fit[1]),c=datplot.get_color())
	



fig,ax = plt.subplots(1,1,figsize=(default_fig_width,default_fig_width*0.4))

print("Analyzing diff. case")
analyze_and_plot(sim_data_base_folder + "complete_diff_long/W_eTOe_record.npy","Diffusive",ax)
print("Analyzing non-diff. case")
analyze_and_plot(sim_data_base_folder + "complete_non_diff_long/W_eTOe_record.npy","Non-Diffusive",ax)


plt.xscale('log')
plt.yscale('log')
plt.xlabel('Syn. Lifetime [s]')
plt.ylabel('Count in Bins')
plt.legend()
plt.xlim([4.,800.])

for f_f in file_format:
	plt.savefig(savefolder+plot_filename+f_f)
plt.show()

pdb.set_trace()
	
