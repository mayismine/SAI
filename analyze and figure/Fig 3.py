import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

plt.rc('font',family='Times New Roman')
fontsize = 15

features_name = ["acc_rain7","WS","Ta", "Press", "RH", "SW_IN",'SM','LAI','SIF',"Month",'DEM','CLAY','SAND','SILT']
Random_forest_rank = []
Attention_rank = []
Expert_rank = []
Influnce_rank = []
Random_forest_pd_data = pd.read_excel(r'analyze and figure\SHAP\RF\SHAP.xlsx')[features_name]
Random_forest_data = np.array(Random_forest_pd_data)
Attention_pd_data = pd.read_excel(r'analyze and figure\SHAP\SA\SHAP.xlsx')[features_name]
Attention_data = np.array(Attention_pd_data)
Expert_pd_data = pd.read_excel(r'analyze and figure\SHAP\MTMS\SHAP.xlsx')[features_name]
Expert_data = np.array(Expert_pd_data)
Influnce_pd_data = pd.read_excel(r'analyze and figure\SHAP\SAI\SHAP.xlsx')[features_name]
Influnce_data = np.array(Influnce_pd_data)

# SHAP_Bar_value
Random_forest_SHAP_Bar_value = np.average(np.abs(Random_forest_data),axis=0)
Attention_SHAP_Bar_value = np.average(np.abs(Attention_data),axis=0) * (9.989863199201565 - 1.0560459715518086e-06)
Expert_SHAP_Bar_value = np.average(np.abs(Expert_data),axis=0) * (9.989863199201565 - 1.0560459715518086e-06)
Influnce_SHAP_Bar_value = np.average(np.abs(Influnce_data),axis=0) * (9.989863199201565 - 1.0560459715518086e-06)

fig=plt.figure(figsize=(12,11))
fig.text(0.22,0.95,'(A) ',fontsize=20,fontweight='bold')
fig.text(0.72,0.95,'(B) ',fontsize=20,fontweight='bold')
fig.text(0.214,0.45,'(C) ',fontsize=20,fontweight='bold')
fig.text(0.72,0.45,'(D) ',fontsize=20,fontweight='bold')
fig.text(0.24,0.95,'     RF',fontsize=20)
fig.text(0.74,0.95,'     SA',fontsize=20)
fig.text(0.234,0.45,'     MTMS',fontsize=20)
fig.text(0.74,0.45,'     SAI',fontsize=20)
gstl = gridspec.GridSpec(3, 3)
gstl.update(top=0.93, bottom=0.55, left=0.05,right=0.45)
gstr = gridspec.GridSpec(3,3)
gstr.update(top=0.93, bottom=0.55, left=0.55,right=0.95)
gsdl = gridspec.GridSpec(3, 3)
gsdl.update(top=0.43, bottom=0.05, left=0.05,right=0.45)
gsdr = gridspec.GridSpec(3,3)
gsdr.update(top=0.43, bottom=0.05, left=0.55,right=0.95)
ax_RF_Bar = plt.subplot(gstl[:,0])
ax_RF = plt.subplot(gstl[:,1:])
ax_Attention_Bar = plt.subplot(gstr[:,0])
ax_Attention = plt.subplot(gstr[:,1:])
ax_Expert_Bar = plt.subplot(gsdl[:,0])
ax_Expert = plt.subplot(gsdl[:,1:])
ax_Influnce_Bar = plt.subplot(gsdr[:,0])
ax_Influnce = plt.subplot(gsdr[:,1:])



# SHAP Bar
ticks = np.arange(14)
Random_forest_rank = [features_name[i] for i in np.argsort(Random_forest_SHAP_Bar_value)[::-1]]
Random_forest_SHAP_Bar_value = Random_forest_SHAP_Bar_value[np.argsort(Random_forest_SHAP_Bar_value)[::-1]]
ax_RF_Bar.invert_xaxis()
ax_RF_Bar = sns.barplot(Random_forest_SHAP_Bar_value,Random_forest_rank,ax = ax_RF_Bar,color='lightgrey')
ax_RF_Bar.set_yticklabels(Random_forest_rank,fontsize=14)
Attention_rank = [features_name[i] for i in np.argsort(Attention_SHAP_Bar_value)[::-1]]
Attention_SHAP_Bar_value = Attention_SHAP_Bar_value[np.argsort(Attention_SHAP_Bar_value)[::-1]]
# ax_Attention_Bar.yaxis.tick_right()
ax_Attention_Bar.invert_xaxis()
ax_Attention_Bar = sns.barplot(Attention_SHAP_Bar_value,Attention_rank,ax = ax_Attention_Bar,color='lightgrey')
ax_Attention_Bar.set_yticklabels(Attention_rank,fontsize=14)
Expert_rank = [features_name[i] for i in np.argsort(Expert_SHAP_Bar_value)[::-1]]
Expert_SHAP_Bar_value = Expert_SHAP_Bar_value[np.argsort(Expert_SHAP_Bar_value)[::-1]]
ax_Expert_Bar.invert_xaxis()
ax_Expert_Bar = sns.barplot(Expert_SHAP_Bar_value,Expert_rank,ax = ax_Expert_Bar,color='lightgrey')
ax_Expert_Bar.set_yticklabels(Expert_rank,fontsize=14)
Influnce_rank = [features_name[i] for i in np.argsort(Influnce_SHAP_Bar_value)[::-1]]
Influnce_SHAP_Bar_value = Influnce_SHAP_Bar_value[np.argsort(Influnce_SHAP_Bar_value)[::-1]]
# ax_Influnce_Bar.yaxis.tick_right()
ax_Influnce_Bar.invert_xaxis()
ax_Influnce_Bar = sns.barplot(Influnce_SHAP_Bar_value,Influnce_rank,ax = ax_Influnce_Bar,color='lightgrey')
ax_Influnce_Bar.set_yticklabels(Influnce_rank,fontsize=14)
for i,v in enumerate(Random_forest_SHAP_Bar_value):
    if i == 0:
        ax_RF_Bar.text(v-0.1,i+0.2,str(round(v,3)),ha='center',fontsize=14)
    else:
        ax_RF_Bar.text(v+0.1,i+0.2,str(round(v,3)),ha='center',fontsize=14)
for i,v in enumerate(Attention_SHAP_Bar_value):
    if i == 0 or i == 1 or i == 2:
        ax_Attention_Bar.text(v-0.07,i+0.2,str(round(v,3)),ha='center',fontsize=14)
    else:
        ax_Attention_Bar.text(v+0.07,i+0.2,str(round(v,3)),ha='center',fontsize=14)
for i,v in enumerate(Expert_SHAP_Bar_value):
    if i == 0 or i == 1:
        ax_Expert_Bar.text(v-0.07,i+0.2,str(round(v,3)),ha='center',fontsize=14)
    else:
        ax_Expert_Bar.text(v+0.07,i+0.2,str(round(v,3)),ha='center',fontsize=14)
for i,v in enumerate(Influnce_SHAP_Bar_value):
    if i == 0 or i == 1 or i == 2:
        ax_Influnce_Bar.text(v-0.07,i+0.2,str(round(v,3)),ha='center',fontsize=14)
    else:
        ax_Influnce_Bar.text(v+0.07,i+0.2,str(round(v,3)),ha='center',fontsize=14)
ax_RF_Bar.set_xlabel('|SHAP| mean value (mm/d)',fontsize=15)
ax_RF_Bar.set_xticks([0.0,0.2,0.4,0.6])
ax_RF_Bar.set_xticklabels([0.0,0.2,0.4,0.6],fontsize=14)
ax_Attention_Bar.set_xlabel('|SHAP| mean value (mm/d)',fontsize=15)
ax_Attention_Bar.set_xticks([0.0,0.1,0.2,0.3,0.4])
ax_Attention_Bar.set_xticklabels([0.0,0.1,0.2,0.3,0.4],fontsize=14)
ax_Expert_Bar.set_xlabel('|SHAP| mean value (mm/d)',fontsize=15)
ax_Expert_Bar.set_xticks([0.0,0.1,0.2,0.3,0.4,0.5])
ax_Expert_Bar.set_xticklabels([0.0,0.1,0.2,0.3,0.4,0.5],fontsize=14)
ax_Influnce_Bar.set_xlabel('|SHAP| mean value (mm/d)',fontsize=15)
ax_Influnce_Bar.set_xticks([0.0,0.1,0.2,0.3,0.4])
ax_Influnce_Bar.set_xticklabels([0.0,0.1,0.2,0.3,0.4],fontsize=14)

# violin
ax_RF.get_yaxis().set_visible(False)
ax_Attention.get_yaxis().set_visible(False)
ax_Expert.get_yaxis().set_visible(False)
ax_Influnce.get_yaxis().set_visible(False)
ax_RF.set_yticks(ticks)
ax_RF.set_yticklabels(Random_forest_rank)
ax_Attention.set_yticks(ticks)
ax_Attention.set_yticklabels(Attention_rank)
ax_Expert.set_yticks(ticks)
ax_Expert.set_yticklabels(Expert_rank)
ax_Influnce.set_yticks(ticks)
ax_Influnce.set_yticklabels(Influnce_rank)
sns.violinplot(x="value",y='type',data=pd.read_excel(r'analyze and figure\SHAP\RF\SHAP_row.xlsx'), orient="h",ax = ax_RF,size=0.5)
sns.violinplot(x="value",y='type',data=pd.read_excel(r'analyze and figure\SHAP\SA\SHAP_row.xlsx'), orient="h",ax = ax_Attention,size=0.5)
sns.violinplot(x="value",y='type',data=pd.read_excel(r'analyze and figure\SHAP\MTMS\SHAP_row.xlsx'), orient="h",ax = ax_Expert,size=0.5)
sns.violinplot(x="value",y='type',data=pd.read_excel(r'analyze and figure\SHAP\SAI\SHAP_row.xlsx'), orient="h",ax = ax_Influnce,size=0.5)
ax_RF.axvline(x=0,c='black',ls='--')
ax_Attention.axvline(x=0,c='black',ls='--')
ax_Expert.axvline(x=0,c='black',ls='--')
ax_Influnce.axvline(x=0,c='black',ls='--')
ax_RF.set_xlabel('SHAP  value (mm/d)',fontsize=18)
ax_RF.set_xticks([-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0])
ax_RF.set_xticklabels([-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0],fontsize=14)
ax_Attention.set_xlabel('SHAP  value (mm/d)',fontsize=18)
ax_Attention.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0])
ax_Attention.set_xticklabels([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0],fontsize=14)
ax_Expert.set_xlabel('SHAP  value (mm/d)',fontsize=18)
ax_Expert.set_xticks([-2,-1,0,1,2,3,4,5,6])
ax_Expert.set_xticklabels([-2,-1,0,1,2,3,4,5,6],fontsize=14)
ax_Influnce.set_xlabel('SHAP  value (mm/d)',fontsize=18)
ax_Influnce.set_xticks([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5])
ax_Influnce.set_xticklabels([-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5],fontsize=14)
ax_RF.spines['right'].set_visible(False)
ax_RF.spines['left'].set_visible(False)
ax_RF.spines['top'].set_visible(False)
ax_Attention.spines['right'].set_visible(False)
ax_Attention.spines['left'].set_visible(False)
ax_Attention.spines['top'].set_visible(False)
ax_Expert.spines['right'].set_visible(False)
ax_Expert.spines['left'].set_visible(False)
ax_Expert.spines['top'].set_visible(False)
ax_Influnce.spines['right'].set_visible(False)
ax_Influnce.spines['left'].set_visible(False)
ax_Influnce.spines['top'].set_visible(False)

# plt.subplots_adjust(top=0.95, bottom=0.05, right=0.95, left=0.05,wspace=0.2)
# plt.margins(0, 0)
plt.show()
# sns.swarmplot()