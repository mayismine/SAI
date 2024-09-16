import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.ticker as ticker

plt.rc('font',family='Times New Roman')

RF_path = r'analyze and figure\SHAP\RF\PDP_LAI_SM.xlsx'
Attention_path = r'analyze and figure\SHAP\SA\PDP_LAI_SM.xlsx'
Expert_path = r'analyze and figure\SHAP\MTMS\PDP_LAI_SM.xlsx'
Influnce_path = r'analyze and figure\SHAP\SAI\PDP_LAI_SM.xlsx'

RF_LAI = np.array(pd.read_excel(RF_path)[['X']])
RF_SHAP = np.array(pd.read_excel(RF_path)[['Y']])
RF_SM = np.array(pd.read_excel(RF_path)[['Z']])
Attention_LAI = np.array(pd.read_excel(Attention_path)[['X']]) * 6
Attention_SHAP = np.array(pd.read_excel(Attention_path)[['Y']]) * (9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Attention_SM = np.array(pd.read_excel(Attention_path)[['Z']]) * (0.484 - 0.031) + 0.031
Expert_LAI = np.array(pd.read_excel(Expert_path)[['X']]) * 6
Expert_SHAP = np.array(pd.read_excel(Expert_path)[['Y']]) * (9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Expert_SM = np.array(pd.read_excel(Expert_path)[['Z']]) * (0.484 - 0.031) + 0.031
Influnce_LAI = np.array(pd.read_excel(Influnce_path)[['X']]) * 6
Influnce_SHAP = np.array(pd.read_excel(Influnce_path)[['Y']]) * (9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Influnce_SM = np.array(pd.read_excel(Influnce_path)[['Z']]) * (0.484 - 0.031) + 0.031
abnormal = np.argmax(Expert_SHAP)
Expert_LAI = np.delete(Expert_LAI,abnormal)
Expert_SHAP = np.delete(Expert_SHAP,abnormal)
Expert_SM = np.delete(Expert_SM,abnormal)
print(np.max([np.max(RF_SM),np.max(Attention_SM),np.max(Expert_SM),np.max(Influnce_SM)]))
print(np.min([np.min(RF_SM),np.min(Attention_SM),np.min(Expert_SM),np.min(Influnce_SM)]))

fig=plt.figure()
fig.subplots_adjust(hspace=0.2,right=0.9)
ax1=fig.add_subplot(2,2,1)
ax1.scatter(RF_LAI, RF_SHAP, s=10, c=RF_SM, cmap='Spectral')
ax1.text(0,0.9,'RF',fontsize=20)
ax2=fig.add_subplot(2,2,2)
ax2.scatter(Attention_LAI, Attention_SHAP, s=10, c=Attention_SM, cmap='Spectral')
ax2.text(0,0.9,'SA',fontsize=20)
ax3=fig.add_subplot(2,2,3)
ax3.scatter(Expert_LAI, Expert_SHAP, s=10, c=Expert_SM, cmap='Spectral')
ax3.text(0,0.9,'MTMS',fontsize=20)
ax4=fig.add_subplot(2,2,4)
ax4.scatter(Influnce_LAI, Influnce_SHAP, s=10, c=Influnce_SM, cmap='Spectral')
ax4.text(0,0.9,'SAI',fontsize=20)

ax1.set_ylim(-1.3,1.3)
ax2.set_ylim(-1.3,1.3)
ax3.set_ylim(-1.3,1.3)
ax4.set_ylim(-1.3,1.3)
ax1.set_yticks([-1.2,-0.8,-0.4,0,0.4,0.8,1.2])
ax2.set_yticks([-1.2,-0.8,-0.4,0,0.4,0.8,1.2])
ax3.set_yticks([-1.2,-0.8,-0.4,0,0.4,0.8,1.2])
ax4.set_yticks([-1.2,-0.8,-0.4,0,0.4,0.8,1.2])
ax1.set_xticks([0,1,2,3,4,5,6])
ax2.set_xticks([0,1,2,3,4,5,6])
ax3.set_xticks([0,1,2,3,4,5,6])
ax4.set_xticks([0,1,2,3,4,5,6])
ax1.set_yticklabels(labels=[-1.2,-0.8,-0.4,0,0.4,0.8,1.2],fontsize = 16)
ax2.set_yticklabels(labels=[-1.2,-0.8,-0.4,0,0.4,0.8,1.2],fontsize = 16)
ax3.set_yticklabels(labels=[-1.2,-0.8,-0.4,0,0.4,0.8,1.2],fontsize = 16)
ax4.set_yticklabels(labels=[-1.2,-0.8,-0.4,0,0.4,0.8,1.2],fontsize = 16)
ax1.set_xticklabels(labels=[0,1,2,3,4,5,6],fontsize = 16)
ax2.set_xticklabels(labels=[0,1,2,3,4,5,6],fontsize = 16)
ax3.set_xticklabels(labels=[0,1,2,3,4,5,6],fontsize = 16)
ax4.set_xticklabels(labels=[0,1,2,3,4,5,6],fontsize = 16)
ax1.set_ylabel('SHAP (mm/d)', fontsize=20)
ax3.set_ylabel('SHAP (mm/d)', fontsize=20)
ax1.set_xlabel('LAI (m$^{2}$/m$^{2}$)', fontsize=20)
ax2.set_xlabel('LAI (m$^{2}$/m$^{2}$)', fontsize=20)
ax3.set_xlabel('LAI (m$^{2}$/m$^{2}$)', fontsize=20)
ax4.set_xlabel('LAI (m$^{2}$/m$^{2}$)', fontsize=20)
cbar_ax = fig.add_axes([0.92,0.15,0.008,0.8])
cb = matplotlib.colorbar.ColorbarBase(ax = cbar_ax,cmap = matplotlib.cm.get_cmap('Spectral_r'),ticks=[])
tick_locator = ticker.MaxNLocator(nbins=1)
cb.locator = tick_locator
cb.set_ticks([])
cb.update_ticks()
plt.text(0.1,0, "     0.045", fontsize=16, color='black')
plt.text(0.1,0.48, "     SM (m$^{3}$/m$^{3}$)", fontsize=16, color='black')
plt.text(0.1,0.97, "     0.322", fontsize=16, color='black')
# plt.suptitle('PDP Interaction of LAI and SM',fontsize=30)
plt.subplots_adjust(top=0.95, bottom=0.15, right=0.9, left=0.1,hspace=0.35)
plt.margins(0, 0)
plt.show()