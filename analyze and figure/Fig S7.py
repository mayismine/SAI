import copy

import matplotlib
import matplotlib as mpl
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.interpolate import make_interp_spline

from general.draw_heatmap import heatmap_basin
import pandas as pd


plt.rc('font', family='Times New Roman')
plt.rcParams['font.size'] = 15
fig = plt.figure(figsize=(26, 14))
plt.subplots_adjust(wspace=0, hspace=0)

gstext = gridspec.GridSpec(4,1)
gstext.update(top=0.94,bottom=0.05,left=0.02,right=0.08)
gs1l = gridspec.GridSpec(2, 4)
gs1l.update(top=0.94, bottom=0.74,left=0.16,right=0.35)
gs1r = gridspec.GridSpec(2, 1)
gs1r.update(top=0.94, bottom=0.74,left=0.45,right=0.95)
gs2l = gridspec.GridSpec(2, 4)
gs2l.update(top=0.71, bottom=0.51,left=0.16,right=0.35)
gs2r = gridspec.GridSpec(2, 1)
gs2r.update(top=0.71, bottom=0.51,left=0.45,right=0.95)
gs3l = gridspec.GridSpec(2, 4)
gs3l.update(top=0.48, bottom=0.28,left=0.16,right=0.35)
gs3r = gridspec.GridSpec(2, 1)
gs3r.update(top=0.48, bottom=0.28,left=0.45,right=0.95)
gs4l = gridspec.GridSpec(2, 4)
gs4l.update(top=0.25, bottom=0.05,left=0.16,right=0.35)
gs4r = gridspec.GridSpec(2, 1)
gs4r.update(top=0.25, bottom=0.05,left=0.45,right=0.95)
gscolorbar = gridspec.GridSpec(1, 4)
gscolorbar.update(top=0.94,bottom=0.05,left=0.36,right=0.37)

ax1 = plt.subplot(gs1l[:,:])
ax2 = plt.subplot(gs1r[:,:])
ax3 = plt.subplot(gs2l[:,:])
ax4 = plt.subplot(gs2r[:,:])
ax5 = plt.subplot(gs3l[:,:])
ax6 = plt.subplot(gs3r[:,:])
ax7 = plt.subplot(gs4l[:,:])
ax8 = plt.subplot(gs4r[:,:])
ax_colorbar = plt.subplot(gscolorbar[:,:])
ax_text = plt.subplot(gstext[:,:])

basin_1 = 0
basin_2 = 6
basin_3 = 3
basin_4 = 29
heat_maps = heatmap_basin(basin_1,6)
columns = ['RF','SA','MTMS','SAI','GLEAM','FLUXCOM']
norm2 = mpl.colors.Normalize(vmin=0, vmax=60)
sns.heatmap(heat_maps, annot=True, fmt='.2f',xticklabels=columns,yticklabels=columns,cmap='Reds',norm=norm2,ax=ax1, annot_kws={"fontsize":10,"color":'black'}, cbar=False)
heat_maps = heatmap_basin(basin_2,6)
sns.heatmap(heat_maps, annot=True, fmt='.2f',xticklabels=columns,yticklabels=columns,cmap='Reds',norm=norm2,ax=ax3, annot_kws={"fontsize":10,"color":'black'}, cbar=False)
heat_maps = heatmap_basin(basin_3,6)
sns.heatmap(heat_maps, annot=True, fmt='.2f',xticklabels=columns,yticklabels=columns,cmap='Reds',norm=norm2,ax=ax5, annot_kws={"fontsize":10,"color":'black'}, cbar=False)
heat_maps = heatmap_basin(basin_4,6)
sns.heatmap(heat_maps, annot=True, fmt='.2f',xticklabels=columns,yticklabels=columns,cmap='Reds',norm=norm2,ax=ax7, annot_kws={"fontsize":10,"color":'black'}, cbar=False)
ax1.set_xticklabels(columns,rotation=0,fontsize=8)
ax3.set_xticklabels(columns,rotation=0,fontsize=8)
ax5.set_xticklabels(columns,rotation=0,fontsize=8)
ax7.set_xticklabels(columns,rotation=0,fontsize=8)
ax1.set_yticklabels(columns,rotation=0,fontsize=8)
ax3.set_yticklabels(columns,rotation=0,fontsize=8)
ax5.set_yticklabels(columns,rotation=0,fontsize=8)
ax7.set_yticklabels(columns,rotation=0,fontsize=8)

RF_data = pd.read_excel(r'results\basin\common\RF_common.xlsx')
Attention_data = pd.read_excel(r'results\basin\common\Attention_common.xlsx')
MTMS_data = pd.read_excel(r'results\basin\common\MTMS_common.xlsx')
SAI_data = pd.read_excel(r'results\basin\common\SAI_common.xlsx')
fluxcom_data = pd.read_excel(r'results\basin\common\Fluxcom_common.xlsx')
gleam_data = pd.read_excel(r'results\basin\common\GLEAM_common.xlsx')


def draw_basin_trend(ax,row):
    RF_basin_data = np.array(RF_data.loc[:,'200205':'201612'])[row,:].flatten()
    Attention_basin_data = np.array(Attention_data.loc[:,'200205':'201612'])[row,:].flatten()
    MTMS_basin_data = np.array(MTMS_data.loc[:,'200205':'201612'])[row,:].flatten()
    SAI_basin_data = np.array(SAI_data.loc[:,'200205':'201612'])[row,:].flatten()
    fluxcom_basin_data = np.array(fluxcom_data.loc[:,'200205':'201612'])[row,:].flatten()
    gleam_basin_data = np.array(gleam_data.loc[:,'200205':'201612'])[row,:].flatten()

    datas = []
    datas.append(RF_basin_data)
    datas.append(Attention_basin_data)
    datas.append(MTMS_basin_data)
    datas.append(SAI_basin_data)
    datas.append(fluxcom_basin_data)
    datas.append(gleam_basin_data)

    linewidth = 2
    plt.rcParams.update({"font.size":10})
    colors = ["#A5405E", "#FEA600", "#BF7533", "#FF4400", "#638DEE", "#AA66EB"]
    labels = ['RF', 'SA', 'MTMS', 'SAI', 'Fluxcom', 'GLEAM']
    x = np.arange(len(np.array(RF_data.loc[row, '200205':'201612'])))

    for i in range(6):
        plot_data = datas[i]
        m = make_interp_spline(x, plot_data)
        xs = np.linspace(0, len(x)-1, 500)
        ys = m(xs)
        ax.plot(xs, ys,color=colors[i],linewidth=linewidth,markersize=6, alpha=0.4,label=labels[i])

    ax.set_xticks([0,20,40,60,80,100,120,140,160,175])
    ax.set_xticklabels(['200205','200401','200509','200705','200901','201009','201205','201401','201509','201612'])
    ax.set_xlabel('Time')
    ax.set_ylabel('ET (mm/month)')
    ax.legend(ncol=6)

draw_basin_trend(ax2,basin_1)
draw_basin_trend(ax4,basin_2)
draw_basin_trend(ax6,basin_3)
draw_basin_trend(ax8,basin_4)


im3 = matplotlib.cm.ScalarMappable(norm=norm2, cmap='Reds')
cbar3 = fig.colorbar(
    im3, cax=ax_colorbar, orientation='vertical',
    label='RMSE colorbar (mm/month)',
)

ax_text.get_xaxis().set_visible(False)
ax_text.get_yaxis().set_visible(False)
ax_text.set_xticks([])
ax_text.set_yticks([])
ax_text.spines['top'].set_visible(False)
ax_text.spines['bottom'].set_visible(False)
ax_text.spines['left'].set_visible(False)
ax_text.spines['right'].set_visible(False)
plt.rcParams.update({"font.size":15})
ax_text.text(0.5,0.88,'AMAZON')
ax_text.text(0.5,0.62,'YENISEY')
ax_text.text(0.5,0.36,'MISSISSIPPI')
ax_text.text(0.5,0.11,'DANUBE')

ax2.set_ylim(50,160)
ax2.set_yticks([50,80,110,140])
ax4.set_ylim(-5,15)
ax4.set_yticks([0,5,10])
ax6.set_ylim(0,155)
ax6.set_yticks([0,30,60,90,120])
ax8.set_ylim(0,150)
ax8.set_yticks([0,30,60,90,120])

plt.show()
