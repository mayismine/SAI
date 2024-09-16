import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

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


# =============================  rich  =========================================================
RF_root = r'results\exp1\RF\test_rich'
SA_root = r'results\exp1\SA\test_rich'
MTMS_root = r'results\exp1\MTMS\test_rich'
SAI_root = r'results\exp1\SAI\test_rich'
true = r'experiment\exp1\test_rich'
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
    SA_filepath = os.path.join(SA_root,file)
    MTMS_filepath = os.path.join(MTMS_root, file)
    SAI_filepath = os.path.join(SAI_root, file)
    truepath = os.path.join(true, file)
    RF_file_data = pd.read_excel(RF_filepath)
    SA_file_data = pd.read_excel(SA_filepath)
    MTMS_file_data = pd.read_excel(MTMS_filepath)
    SAI_file_data = pd.read_excel(SAI_filepath)
    true_data = pd.read_csv(truepath.replace('.xlsx', '.csv'))
    name = str(file)[4:10]
    cate = station.loc[3, name]
    dict_RF[str(cate)] = pd.concat([dict_RF[str(cate)], RF_file_data])
    dict_Attention[str(cate)] = pd.concat([dict_Attention[str(cate)], SA_file_data])
    dict_MSMT[str(cate)] = pd.concat([dict_MSMT[str(cate)], MTMS_file_data])
    dict_Rain[str(cate)] = pd.concat([dict_Rain[str(cate)], SAI_file_data])
    dict_truth[str(cate)] = pd.concat([dict_truth[str(cate)], true_data[["ET"]]])

all_rich_data = []
name = []
for key in dict_truth:
    if len(dict_truth[key]) == 0:
        continue
    IGBP_data = []
    name.append(str(key))
    truth = np.array(dict_truth[key])[:, 0:1]
    RF = np.array(dict_RF[key])[:, 1:2]
    Attention = np.array(dict_Attention[key])[:, 1:2]
    MSMT = np.array(dict_MSMT[key])[:, 1:2]
    Rain = np.array(dict_Rain[key])[:, 1:2]
    IGBP_data.append(truth)
    IGBP_data.append(RF)
    IGBP_data.append(Attention)
    IGBP_data.append(MSMT)
    IGBP_data.append(Rain)
    all_rich_data.append(IGBP_data)


# =============================  lack  =========================================================
RF_root = r'results\exp1\RF\test_lack'
SA_root = r'results\exp1\SA\test_lack'
MTMS_root = r'results\exp1\MTMS\test_lack'
SAI_root = r'results\exp1\SAI\test_lack'
true = r'experiment\exp1\test_lack'
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
    Attention_filepath = os.path.join(SA_root,file)
    MSMT_filepath = os.path.join(MTMS_root, file)
    Rain_filepath = os.path.join(SAI_root, file)
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

all_lack_data = []
name = []
for key in dict_truth:
    if len(dict_truth[key]) == 0:
        continue
    IGBP_data = []
    name.append(str(key))
    truth = np.array(dict_truth[key])[:, 0:1]
    RF = np.array(dict_RF[key])[:, 1:2]
    Attention = np.array(dict_Attention[key])[:, 1:2]
    MSMT = np.array(dict_MSMT[key])[:, 1:2]
    Rain = np.array(dict_Rain[key])[:, 1:2]
    IGBP_data.append(truth)
    IGBP_data.append(RF)
    IGBP_data.append(Attention)
    IGBP_data.append(MSMT)
    IGBP_data.append(Rain)
    all_lack_data.append(IGBP_data)

plt.rc('font',family='Times New Roman')
fig=plt.figure()
ax1=fig.add_subplot(211)
ax2=fig.add_subplot(212)
labels = ["In Situ Measurement", "RF","SA","MTMS","SAI"]
colors = ["lightgray","lightgreen","khaki","lightblue","orange"]
i = 0
for key in dict_truth:
    if len(dict_truth[key]) == 0:
        continue
    bplot = ax1.boxplot(np.array(all_rich_data[i]).squeeze().transpose(1,0), notch=True,patch_artist=True, labels=labels, positions=(2*i+0.6, 2*i+0.9, 2*i+1.2,2*i+1.5,2*i+1.8), widths=0.25)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    bplot = ax2.boxplot(np.array(all_lack_data[i]).squeeze().transpose(1, 0), notch=True,patch_artist=True, labels=labels,
                        positions=(2*i+0.6, 2*i+0.9, 2*i+1.2,2*i+1.5,2*i+1.8), widths=0.25)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
    i = i + 1

x_position = []
x_position_fmt = []
i = 0
for key in dict_truth:
    if len(dict_truth[key]) == 0:
        continue
    x = 2 * i + 1.2
    x_position.append(x)
    x_position_fmt.append(str(key))
    i = i + 1
ax1.set_xticks([i for i in x_position])
ax1.set_xticklabels(x_position_fmt,fontsize=16)
ax2.set_xticks([i for i in x_position])
ax2.set_xticklabels(x_position_fmt,fontsize=16)
ax1.text(0.3,6.7,'(A) ',fontsize=18,fontweight='bold')
ax2.text(0.3,6.7,'(B) ',fontsize=18,fontweight='bold')
ax1.text(0.6,6.7,'Data-rich region test',fontsize=18)
ax2.text(0.6,6.7,'Data-poor region test',fontsize=18)
ax1.set_ylabel('ET Distribution (mm/d)',fontsize=16)
ax2.set_ylabel('ET Distribution (mm/d)',fontsize=16)
ax1.grid(linestyle="--", alpha=0.3)
ax2.grid(linestyle="--", alpha=0.3)
ax1.set_ylim(-0.5,7.5)
ax2.set_ylim(-0.5,7.5)
ax1.set_yticks([0,1,2,3,4,5,6,7])
ax1.set_yticklabels([0,1,2,3,4,5,6,7])
ax2.set_yticks([0,1,2,3,4,5,6,7])
ax2.set_yticklabels([0,1,2,3,4,5,6,7])
ax1.legend(bplot['boxes'], labels, loc='upper right',ncol=5,fontsize=15)
ax2.legend(bplot['boxes'], labels, loc='upper right',ncol=5,fontsize=15)
# plt.suptitle('Rich Region and Poor Region Test ET Distribution',fontsize=20)
plt.subplots_adjust(top=1, bottom=0.05, right=0.9, left=0.05)
plt.margins(0, 0)
plt.show()
