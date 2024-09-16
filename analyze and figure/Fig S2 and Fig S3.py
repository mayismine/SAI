import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import gridspec
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde
import os
import math
import seaborn as sns
import matplotlib

plt.rc('font',family='Times New Roman')

def get_data(path):
    list = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        filedata = pd.read_csv(filepath)
        list.append(filedata)
    data = list[0]
    for i in range(len(list)):
        if i == 0:
            continue
        data = pd.concat([data, list[i]])
    return data

def get_data1(path):
    list = []
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        filedata = pd.read_excel(filepath)
        list.append(filedata)
    data = list[0]
    for i in range(len(list)):
        if i == 0:
            continue
        data = pd.concat([data, list[i]])
    return data

which = 'CN'
RF_root = r'results\exp2\RF'
Attention_root = r'results\exp2\SA'
MSMT_root = r'results\exp2\MTMS'
MSMT_Rain_root = r'results\exp2\SAI'
true = r'experiment\exp2\test'
station = pd.read_excel(r'data\site_info\loc_research_train_test.xlsx')
empty = pd.DataFrame([])
dict_RF = {'GRA': empty, 'WSA': empty, 'SAV': empty, 'EBF': empty, 'DBF': empty, 'MF': empty, 'ENF': empty, 'WET': empty,
        'CRO': empty, 'OSH': empty, 'CSH': empty}
dict_Attention = {'GRA': empty, 'WSA': empty, 'SAV': empty, 'EBF': empty, 'DBF': empty, 'MF': empty, 'ENF': empty, 'WET': empty,
        'CRO': empty, 'OSH': empty, 'CSH': empty}
dict_MSMT = {'GRA': empty, 'WSA': empty, 'SAV': empty, 'EBF': empty, 'DBF': empty, 'MF': empty, 'ENF': empty, 'WET': empty,
        'CRO': empty, 'OSH': empty, 'CSH': empty}
dict_Rain = {'GRA': empty, 'WSA': empty, 'SAV': empty, 'EBF': empty, 'DBF': empty, 'MF': empty, 'ENF': empty, 'WET': empty,
        'CRO': empty, 'OSH': empty, 'CSH': empty}
dict_truth = {'GRA': empty, 'WSA': empty, 'SAV': empty, 'EBF': empty, 'DBF': empty, 'MF': empty, 'ENF': empty,
            'WET': empty, 'CRO': empty, 'OSH': empty, 'CSH': empty}
for file in os.listdir(RF_root):
    RF_filepath = os.path.join(RF_root, file)
    Attention_filepath = os.path.join(Attention_root,file)
    MSMT_filepath = os.path.join(MSMT_root, file)
    Rain_filepath = os.path.join(MSMT_Rain_root, file)
    truepath = os.path.join(true, file)
    RF_file_data = pd.read_excel(RF_filepath)
    Attention_file_data = pd.read_excel(Attention_filepath)
    MSMT_file_data = pd.read_excel(MSMT_filepath)
    Rain_file_data = pd.read_excel(Rain_filepath)
    true_data = pd.read_csv(truepath.replace('.xlsx', '.csv'))
    name = str(file)[4:10]
    cate = station.loc[3, name]
    dict_RF[str(cate)] = pd.concat([dict_RF[str(cate)], RF_file_data])
    dict_Attention[str(cate)] = pd.concat([dict_Attention[str(cate)], Attention_file_data])
    dict_MSMT[str(cate)] = pd.concat([dict_MSMT[str(cate)], MSMT_file_data])
    dict_Rain[str(cate)] = pd.concat([dict_Rain[str(cate)], Rain_file_data])
    dict_truth[str(cate)] = pd.concat([dict_truth[str(cate)], true_data[["ET"]]])

all_data = []
name = []
for key in dict_truth:
    if len(dict_truth[key]) == 0:
        continue
    if key == 'DBF':
        continue
    IGBP_data = []
    name.append(str(key))
    truth = np.array(dict_truth[key])[:, 0:1]
    RF = np.array(dict_RF[key])[:, 1:2] - truth
    Attention = np.array(dict_Attention[key])[:, 1:2] - truth
    MSMT = np.array(dict_MSMT[key])[:, 1:2] - truth
    Rain = np.array(dict_Rain[key])[:, 1:2] - truth
    IGBP_data.append(RF)
    IGBP_data.append(Attention)
    IGBP_data.append(MSMT)
    IGBP_data.append(Rain)
    all_data.append(IGBP_data)

fig=plt.figure(figsize=(12,11))
fig.subplots_adjust(right=0.9)
gstop = gridspec.GridSpec(2, 6)
# gstop = gridspec.GridSpec(1, 5)
gstop.update(top=0.9, bottom=0.62, wspace=0.05)
gs = gridspec.GridSpec(4,6)
# gs = gridspec.GridSpec(4,5)
gs.update(top=0.57, bottom=0.1)
ax1=plt.subplot(gstop[:,:])
labels = ["RF","SA","MTMS","SAI"]
colors = ["lightgreen","khaki","lightblue","orange"]
i = 0
for key in dict_truth:
    if len(dict_truth[key]) == 0:
        continue
    if key == 'DBF':
        continue
    bplot = ax1.boxplot(np.array(all_data[i]).squeeze().transpose(1,0), notch=True,showfliers=False, patch_artist=True, labels=labels, positions=(2*i+0.6, 2*i+1, 2*i+1.4,2*i+1.8), widths=0.3)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    i = i + 1

x_position = []
x_position_fmt = []
i = 0
for key in dict_truth:
    if len(dict_truth[key]) == 0:
        continue
    if key == 'DBF':
        continue
    x = 2 * i + 1.2
    x_position.append(x)
    x_position_fmt.append(str(key))
    # x_position_fmt.append('')
    i = i + 1
ax1.set_xticks([i for i in x_position])
ax1.set_xticklabels(x_position_fmt,fontsize=16)

ax1.set_ylabel('Bias Distribution (mm/d)', fontsize=16)
ax1.grid(linestyle="--", alpha=0.3)
ax1.legend(bplot['boxes'], labels, loc='lower left',ncol=4,fontsize=15)

i = 0
RMSE_data = []
R2_data = []
ax_data = []
for key in dict_truth:
    if len(dict_truth[key]) == 0:
        continue
    if key == 'DBF':
        continue
    truth = np.array(dict_truth[key])[:, 0:1]
    RF = np.array(dict_RF[key])[:, 1:2]
    Attention = np.array(dict_Attention[key])[:, 1:2]
    MSMT = np.array(dict_MSMT[key])[:, 1:2]
    Rain = np.array(dict_Rain[key])[:, 1:2]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.3)
    # =========================================RF=============================================================
    # ax2 = fig.add_subplot(5, 5, 6 + i)
    ax2 = plt.subplot(gs[0, i])
    x = truth.reshape(-1)
    xy = np.vstack([x, RF.reshape(-1)])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, RF, z = x[idx], RF.reshape(-1)[idx], z[idx]
    #ax2.scatter(x,RF,c=z,cmap='Spectral_r')
    sns.kdeplot(x=x, y=RF, fill=True, cmap='Spectral_r' ,ax=ax2,norm=norm)
    miny = np.min(truth)
    maxy = np.max(truth)
    minx = np.min(RF)
    maxx = np.max(RF)
    y_x = np.arange(-1,8.5,1)   #(-1, max(maxy, maxx)+1, 1)
    ax2.plot(y_x, y_x, color='black', alpha=0.5, label='y=x',linestyle='dashed')
    ax2.set_xlim((-1,8.5))  #((-1,max(maxy, maxx)))
    ax2.set_ylim((-1,8.5))  #((-1, max(maxy, maxx)))
    ax2.set_xticks([0,2,4,6,8])  #(np.arange(0,math.floor(max(maxy, maxx))+1,2))
    ax2.set_yticks([0,2,4,6,8])  #(np.arange(0,math.floor(max(maxy, maxx))+1,2))
    ax2.tick_params(pad=0)
    regressor = LinearRegression()
    regressor = regressor.fit(truth, np.array(dict_RF[key])[:, 1:2])
    RMSE = np.sqrt(mean_squared_error(truth.reshape(-1), np.array(dict_RF[key])[:, 1:2]))
    RMSE_data.append(RMSE)
    R2_data.append(str(round(regressor.coef_[0, 0], 2)))
    ax_data.append(ax2)
    # =========================================Attention=============================================================
    # ax3 = fig.add_subplot(5, 5, 11 + i)
    ax3 = plt.subplot(gs[1, i])
    x = truth.reshape(-1)
    xy = np.vstack([x, Attention.reshape(-1)])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, Attention, z = x[idx], Attention.reshape(-1)[idx], z[idx]
    flag = sns.kdeplot(x=x, y=Attention, fill=True, cmap='Spectral_r', ax=ax3, norm=norm)
    minx = np.min(Attention)
    maxx = np.max(Attention)
    #y_x = np.arange(-1, max(maxy, maxx)+1, 1)
    ax3.plot(y_x, y_x, color='black', alpha=0.5, label='y=x',linestyle='dashed')
    ax3.set_xlim((-1,8.5))  #((-1, max(maxy, maxx)))
    ax3.set_ylim((-1,8.5))  #((-1, max(maxy, maxx)))
    ax3.set_xticks([0,2,4,6,8])  #(np.arange(0,math.floor(max(maxy, maxx))+1,2))
    ax3.set_yticks([0,2,4,6,8])  #(np.arange(0,math.floor(max(maxy, maxx))+1,2))
    ax3.tick_params(pad=0)
    regressor = regressor.fit(truth, np.array(dict_Attention[key])[:, 1:2])
    RMSE = np.sqrt(mean_squared_error(truth, np.array(dict_Attention[key])[:, 1:2]))
    RMSE_data.append(RMSE)
    R2_data.append(str(round(regressor.coef_[0, 0], 2)))
    ax_data.append(ax3)
    #=========================================MSMT=============================================================
    # ax4 = fig.add_subplot(5, 5, 16 + i)
    ax4 = plt.subplot(gs[2, i])
    x = truth.reshape(-1)
    xy = np.vstack([x, MSMT.reshape(-1)])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, MSMT, z = x[idx], MSMT.reshape(-1)[idx], z[idx]
    sns.kdeplot(x=x, y=MSMT, fill=True, cmap='Spectral_r', ax=ax4, norm=norm)
    minx = np.min(MSMT)
    maxx = np.max(MSMT)
    #y_x = np.arange(-1, max(maxy, maxx)+1, 1)
    ax4.plot(y_x, y_x, color='black', alpha=0.5, label='y=x',linestyle='dashed')
    ax4.set_xlim((-1,8.5))  #((-1, max(maxy, maxx)))
    ax4.set_ylim((-1,8.5))  #((-1, max(maxy, maxx)))
    ax4.set_xticks([0,2,4,6,8])  #(np.arange(0,math.floor(max(maxy, maxx))+1,2))
    ax4.set_yticks([0,2,4,6,8])  #(np.arange(0,math.floor(max(maxy, maxx))+1,2))
    ax4.tick_params(pad=0)
    regressor = regressor.fit(truth, np.array(dict_MSMT[key])[:, 1:2])
    # ax4.plot(np.reshape(y_x, (-1, 1)), regressor.predict(np.reshape(y_x, (-1, 1))), color='black', alpha=0.9,
    #          label='y=' + str(round(regressor.coef_[0, 0], 2)) + 'x+' + str(round(regressor.intercept_[0], 2)),linewidth=2,linestyle='dotted')
    RMSE = np.sqrt(mean_squared_error(truth, np.array(dict_MSMT[key])[:, 1:2]))
    RMSE_data.append(RMSE)
    R2_data.append(str(round(regressor.coef_[0, 0], 2)))
    ax_data.append(ax4)
    # =========================================MSMT_Rain=============================================================
    # ax5 = fig.add_subplot(5, 5, 21 + i)
    ax5 = plt.subplot(gs[3, i])
    x = truth.reshape(-1)
    xy = np.vstack([x, Rain.reshape(-1)])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, Rain, z = x[idx], Rain.reshape(-1)[idx], z[idx]
    sns.kdeplot(x=x, y=Rain, fill=True, cmap='Spectral_r', ax=ax5, norm=norm)
    minx = np.min(Rain)
    maxx = np.max(Rain)
    #y_x = np.arange(-1, max(maxy, maxx)+1, 1)
    ax5.plot(y_x, y_x, color='black', alpha=0.5, label='y=x',linestyle='dashed')
    ax5.set_xlim((-1,8.5))  #((-1, max(maxy, maxx)))
    ax5.set_ylim((-1,8.5))  #((-1, max(maxy, maxx)))
    ax5.set_xticks([0,2,4,6,8])  #(np.arange(0,math.floor(max(maxy, maxx))+1,2))
    ax5.set_yticks([0,2,4,6,8])  #(np.arange(0,math.floor(max(maxy, maxx))+1,2))
    ax5.tick_params(pad=0)
    regressor = regressor.fit(truth, np.array(dict_Rain[key])[:, 1:2])
    RMSE = np.sqrt(mean_squared_error(truth, np.array(dict_Rain[key])[:, 1:2]))
    RMSE_data.append(RMSE)
    R2_data.append(str(round(regressor.coef_[0, 0], 2)))
    ax_data.append(ax5)
    i = i + 1
# cbar_ax = fig.add_axes([0.92,0.11,0.008,0.505])
text_ax = fig.add_axes([0.92,0.6,0.02,0.25])
cbar_ax = fig.add_axes([0.92,0.1,0.008,0.47])
text_ax.set_xticks([])
text_ax.set_yticks([])
text_ax.spines['top'].set_visible(False)
text_ax.spines['right'].set_visible(False)
text_ax.spines['bottom'].set_visible(False)
text_ax.spines['left'].set_visible(False)
text_ax.text(0.0,-0.04,'Density Colorbar',fontsize=14,rotation=90)
cb = matplotlib.colorbar.ColorbarBase(ax = cbar_ax,cmap = matplotlib.cm.get_cmap('Spectral_r'),norm=norm)
#==============================lack:RMSE/R2======================================================================
RMSE_fontsize = 10
if which == 'AU':
    #=================GRA==================
    ax_data[0].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[0]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[0].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[0]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[0].set_ylabel('RF', fontsize=14)
    ax_data[1].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[1]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[1].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[1]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[1].set_ylabel('SA', fontsize=14)
    ax_data[2].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[2]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[2].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[2]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[2].set_ylabel('MTMS', fontsize=14)
    ax_data[3].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[3]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[3].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[3]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[3].set_ylabel('SAI', fontsize=14)
    # ax_data[3].set_xlabel('Obs', fontsize=16)
    #=================WSA==================
    ax_data[4].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[4]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[4].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[4]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[5].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[5]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[5].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[5]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[6].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[6]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[6].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[6]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[7].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[7]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[7].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[7]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[7].set_xlabel('Obs', fontsize=16)
    #=================SAV==================
    ax_data[8].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[8]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[8].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[8]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[9].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[9]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[9].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[9]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[10].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[10]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[10].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[10]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[11].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[11]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[11].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[11]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[11].set_xlabel('Obs', fontsize=16)
    #=================EBF==================
    ax_data[12].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[12]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[12].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[12]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[13].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[13]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[13].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[13]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[14].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[14]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[14].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[14]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[15].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[15]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[15].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[15]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[15].set_xlabel('Obs', fontsize=16)
    #=================WET==================
    ax_data[16].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[16]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[16].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[16]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[17].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[17]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[17].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[17]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[18].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[18]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[18].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[18]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[19].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[19]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[19].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[19]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[19].set_xlabel('Obs', fontsize=16)
#==============================rich:RMSE/R2======================================================================
if which == 'CN':
    #=================GRA==================
    ax_data[0].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[0]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[0].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[0]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[0].set_ylabel('RF', fontsize=14)
    ax_data[1].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[1]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[1].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[1]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[1].set_ylabel('SA', fontsize=14)
    ax_data[2].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[2]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[2].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[2]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[2].set_ylabel('MTMS', fontsize=14)
    ax_data[3].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[2]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[3].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[2]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[3].set_ylabel('SAI', fontsize=14)
    # ax_data[3].set_xlabel('Obs', fontsize=16)
    #=================EBF==================
    ax_data[4].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[4]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[4].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[4]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[5].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[5]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[5].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[5]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[6].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[6]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[6].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[6]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[7].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[7]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[7].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[7]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[7].set_xlabel('Obs', fontsize=16)
    #=================M F==================
    ax_data[8].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[8]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[8].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[8]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[9].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[9]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[9].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[9]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[10].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[10]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[10].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[10]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[11].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[11]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[11].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[11]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[11].set_xlabel('Obs', fontsize=16)
    #=================ENF==================
    ax_data[12].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[12]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[12].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[12]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[13].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[13]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[13].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[13]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[14].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[14]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[14].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[14]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[15].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[15]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[15].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[15]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[15].set_xlabel('Obs', fontsize=16)
    #=================WET==================
    ax_data[16].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[16]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[16].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[16]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[17].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[17]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[17].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[17]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[18].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[18]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[18].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[18]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[19].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[19]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[19].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[19]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[19].set_xlabel('Obs', fontsize=16)
    #=================CRO==================
    ax_data[20].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[20]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[20].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[20]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[21].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[21]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[21].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[21]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[22].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[22]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[22].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[22]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[23].text(0, 5, '$\it{R^{2}}$=' + str('%.2f'%round(float(R2_data[23]), 2)), fontsize=RMSE_fontsize, color='black')
    ax_data[23].text(0, 6.5, 'RMSE=' + str('%.2f'%round(float(RMSE_data[23]), 2)), fontsize=RMSE_fontsize, color='black')
    # ax_data[23].set_xlabel('Obs', fontsize=16)

# ax_data[16].set_xlim((-1,5))
# ax_data[16].set_ylim((-1, 5))
# ax_data[16].set_xticks(np.arange(0,5,2))
# ax_data[16].set_yticks(np.arange(0,5,2))
# ax_data[24].set_xlim((-1,4.2))
# ax_data[24].set_ylim((-1, 4.2))
# ax_data[24].set_xticks(np.arange(0,4,2))
# ax_data[24].set_yticks(np.arange(0,4,2))
# ax_data[25].set_xlim((-1,4.2))
# ax_data[25].set_ylim((-1, 4.2))
# ax_data[25].set_xticks(np.arange(0,4,2))
# ax_data[25].set_yticks(np.arange(0,4,2))
# ax_data[26].set_xlim((-1,4.2))
# ax_data[26].set_ylim((-1, 4.2))
# ax_data[26].set_xticks(np.arange(0,4,2))
# ax_data[26].set_yticks(np.arange(0,4,2))
# ax_data[27].set_xlim((-1,4.2))
# ax_data[27].set_ylim((-1, 4.2))
# ax_data[27].set_xticks(np.arange(0,4,2))
# ax_data[27].set_yticks(np.arange(0,4,2))

plt.subplots_adjust(top=1, bottom=0, right=0.9, left=0.08)
# plt.suptitle('Australia Test Bias Distribution and Scatter Density',fontsize=20)
ax_text1 = fig.add_axes([0.035,0.14,0.02,0.2])
ax_text1.spines['top'].set_visible(False)
ax_text1.spines['left'].set_visible(False)
ax_text1.spines['right'].set_visible(False)
ax_text1.spines['bottom'].set_visible(False)
ax_text1.set_xticks([])
ax_text1.set_yticks([])
ax_text1.text(0.05,0.5,'ET estimation (mm/d)',rotation=90,fontsize=16)
ax_text2 = fig.add_axes([0.29,0.03,0.2,0.05])
ax_text2.spines['top'].set_visible(False)
ax_text2.spines['left'].set_visible(False)
ax_text2.spines['right'].set_visible(False)
ax_text2.spines['bottom'].set_visible(False)
ax_text2.set_xticks([])
ax_text2.set_yticks([])
ax_text2.text(0,0,'ET Observation (mm/d)',fontsize=16)
plt.margins(0, 0)
plt.show()

#========================================================================