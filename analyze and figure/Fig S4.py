import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rc('font',family='Times New Roman')

# features_name = ["acc_rain7","WS","Ta", "Press", "RH", "SW_IN",'SM','LAI','SIF',"Month",'DEM','CLAY','SAND','SILT']
# features_X = ["acc_rain7_X","WS_X","Ta_X", "Press_X", "RH_X", "SW_IN_X",'SM_X','LAI_X','SIF_X',"Month_X",'DEM_X','CLAY_X','SAND_X','SILT_X']
RF_path = r'analyze and figure\SHAP\RF\PDP.xlsx'
Attention_path = r'analyze and figure\SHAP\SA\PDP.xlsx'
Expert_path = r'analyze and figure\SHAP\MTMS\PDP.xlsx'
Influnce_path = r'analyze and figure\SHAP\SAI\PDP.xlsx'
# Random_forest PDP
RF_CLAY_data = np.array(pd.read_excel(RF_path)[['CLAY']])
RF_CLAY_X = np.array(pd.read_excel(RF_path)[['CLAY_X']])
RF_DEM_data = np.array(pd.read_excel(RF_path)[['DEM']])
RF_DEM_X = np.array(pd.read_excel(RF_path)[['DEM_X']])
RF_acc_rain7_data = np.array(pd.read_excel(RF_path)[['acc_rain7']])
RF_acc_rain7_X = np.array(pd.read_excel(RF_path)[['acc_rain7_X']])
RF_Month_data = np.array(pd.read_excel(RF_path)[['Month']])
RF_Month_X = np.array(pd.read_excel(RF_path)[['Month_X']])
RF_SIF_data = np.array(pd.read_excel(RF_path)[['SIF']])
RF_SIF_X = np.array(pd.read_excel(RF_path)[['SIF_X']])
RF_LAI_data = np.array(pd.read_excel(RF_path)[['LAI']])
RF_LAI_X = np.array(pd.read_excel(RF_path)[['LAI_X']])
RF_SW_IN_data = np.array(pd.read_excel(RF_path)[['SW_IN']])
RF_SW_IN_X = np.array(pd.read_excel(RF_path)[['SW_IN_X']])
RF_RH_data = np.array(pd.read_excel(RF_path)[['RH']])
RF_RH_X = np.array(pd.read_excel(RF_path)[['RH_X']])
# Attention PDP
Attention_CLAY_data = np.array(pd.read_excel(Attention_path)[['CLAY']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Attention_CLAY_X = np.array(pd.read_excel(Attention_path)[['CLAY_X']])*(57-3)+3
Attention_DEM_data = np.array(pd.read_excel(Attention_path)[['DEM']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Attention_DEM_X = np.array(pd.read_excel(Attention_path)[['DEM_X']])*(4319+7)-7
Attention_acc_rain7_data = np.array(pd.read_excel(Attention_path)[['acc_rain7']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Attention_acc_rain7_X = np.array(pd.read_excel(Attention_path)[['acc_rain7_X']])*1837.4
Attention_Month_data = np.array(pd.read_excel(Attention_path)[['Month']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Attention_Month_X = np.array(pd.read_excel(Attention_path)[['Month_X']])*(12-1)+1
Attention_SIF_data = np.array(pd.read_excel(Attention_path)[['SIF']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Attention_SIF_X = np.array(pd.read_excel(Attention_path)[['SIF_X']])*(0.7697+0.0633)-0.0633
Attention_LAI_data = np.array(pd.read_excel(Attention_path)[['LAI']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Attention_LAI_X = np.array(pd.read_excel(Attention_path)[['LAI_X']])*6
Attention_SW_IN_data = np.array(pd.read_excel(Attention_path)[['SW_IN']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Attention_SW_IN_X = np.array(pd.read_excel(Attention_path)[['SW_IN_X']])*421.688
Attention_RH_data = np.array(pd.read_excel(Attention_path)[['RH']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Attention_RH_X = np.array(pd.read_excel(Attention_path)[['RH_X']])*(100.0-3.708541667)+3.708541667
# Expert PDP
Expert_CLAY_data = np.array(pd.read_excel(Expert_path)[['CLAY']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Expert_CLAY_X = np.array(pd.read_excel(Expert_path)[['CLAY_X']])*(57-3)+3
Expert_DEM_data = np.array(pd.read_excel(Expert_path)[['DEM']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Expert_DEM_X = np.array(pd.read_excel(Expert_path)[['DEM_X']])*(4319+7)-7
Expert_acc_rain7_data = np.array(pd.read_excel(Expert_path)[['acc_rain7']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Expert_acc_rain7_X = np.array(pd.read_excel(Expert_path)[['acc_rain7_X']])*1837.4
Expert_Month_data = np.array(pd.read_excel(Expert_path)[['Month']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Expert_Month_X = np.array(pd.read_excel(Expert_path)[['Month_X']])*(12-1)+1
Expert_SIF_data = np.array(pd.read_excel(Expert_path)[['SIF']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Expert_SIF_X = np.array(pd.read_excel(Expert_path)[['SIF_X']])*(0.7697+0.0633)-0.0633
Expert_LAI_data = np.array(pd.read_excel(Expert_path)[['LAI']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Expert_LAI_X = np.array(pd.read_excel(Expert_path)[['LAI_X']])*6
Expert_SW_IN_data = np.array(pd.read_excel(Expert_path)[['SW_IN']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Expert_SW_IN_X = np.array(pd.read_excel(Expert_path)[['SW_IN_X']])*421.688
Expert_RH_data = np.array(pd.read_excel(Expert_path)[['RH']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Expert_RH_X = np.array(pd.read_excel(Expert_path)[['RH_X']])*(100.0-3.708541667)+3.708541667
# Influnce PDP
Influnce_CLAY_data = np.array(pd.read_excel(Influnce_path)[['CLAY']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Influnce_CLAY_X = np.array(pd.read_excel(Influnce_path)[['CLAY_X']])*(57-3)+3
Influnce_DEM_data = np.array(pd.read_excel(Influnce_path)[['DEM']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Influnce_DEM_X = np.array(pd.read_excel(Influnce_path)[['DEM_X']])*(4319+7)-7
Influnce_acc_rain7_data = np.array(pd.read_excel(Influnce_path)[['acc_rain7']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Influnce_acc_rain7_X = np.array(pd.read_excel(Influnce_path)[['acc_rain7_X']])*1837.4
Influnce_Month_data = np.array(pd.read_excel(Influnce_path)[['Month']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Influnce_Month_X = np.array(pd.read_excel(Influnce_path)[['Month_X']])*(12-1)+1
Influnce_SIF_data = np.array(pd.read_excel(Influnce_path)[['SIF']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Influnce_SIF_X = np.array(pd.read_excel(Influnce_path)[['SIF_X']])*(0.7697+0.0633)-0.0633
Influnce_LAI_data = np.array(pd.read_excel(Influnce_path)[['LAI']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Influnce_LAI_X = np.array(pd.read_excel(Influnce_path)[['LAI_X']])*6
Influnce_SW_IN_data = np.array(pd.read_excel(Influnce_path)[['SW_IN']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Influnce_SW_IN_X = np.array(pd.read_excel(Influnce_path)[['SW_IN_X']])*421.688
Influnce_RH_data = np.array(pd.read_excel(Influnce_path)[['RH']])*(9.989863199201565-1.0560459715518086e-06)+1.0560459715518086e-06
Influnce_RH_X = np.array(pd.read_excel(Influnce_path)[['RH_X']])*(100.0-3.708541667)+3.708541667

fig=plt.figure()
step = 3
fig.subplots_adjust(hspace=0.2)
ax1=fig.add_subplot(2,4,1)
ax1.plot(RF_DEM_X,RF_DEM_data,color='lightgreen',label='RF')
ax1.plot(Attention_DEM_X[::step],Attention_DEM_data[::step],color="khaki",label='SA')
ax1.plot(Expert_DEM_X[::step],Expert_DEM_data[::step],color="lightblue",label='MTMS')
ax1.plot(Influnce_DEM_X[::step],Influnce_DEM_data[::step],color="orange",label='SAI')

ax2=fig.add_subplot(2,4,2)
ax2.plot(RF_CLAY_X,RF_CLAY_data,color='lightgreen')
ax2.plot(Attention_CLAY_X[::step],Attention_CLAY_data[::step],color="khaki")
ax2.plot(Expert_CLAY_X[::step],Expert_CLAY_data[::step],color="lightblue")
ax2.plot(Influnce_CLAY_X[::step],Influnce_CLAY_data[::step],color="orange")

ax3=fig.add_subplot(2,4,3)
ax3.plot(RF_acc_rain7_X,RF_acc_rain7_data,color='lightgreen')
ax3.plot(Attention_acc_rain7_X[::step],Attention_acc_rain7_data[::step],color="khaki")
ax3.plot(Expert_acc_rain7_X[::step],Expert_acc_rain7_data[::step],color="lightblue")
ax3.plot(Influnce_acc_rain7_X[::step],Influnce_acc_rain7_data[::step],color="orange")

ax4=fig.add_subplot(2,4,4)
ax4.plot(RF_Month_X,RF_Month_data,color='lightgreen')
ax4.plot(Attention_Month_X[::step],Attention_Month_data[::step],color="khaki")
ax4.plot(Expert_Month_X[::step],Expert_Month_data[::step],color="lightblue")
ax4.plot(Influnce_Month_X[::step],Influnce_Month_data[::step],color="orange")

ax5=fig.add_subplot(2,4,5)
ax5.plot(RF_SIF_X,RF_SIF_data,color='lightgreen')
ax5.plot(Attention_SIF_X[::step],Attention_SIF_data[::step],color="khaki")
ax5.plot(Expert_SIF_X[::step],Expert_SIF_data[::step],color="lightblue")
ax5.plot(Influnce_SIF_X[::step],Influnce_SIF_data[::step],color="orange")

ax6=fig.add_subplot(2,4,6)
ax6.plot(RF_LAI_X,RF_LAI_data,color='lightgreen')
ax6.plot(Attention_LAI_X[::step],Attention_LAI_data[::step],color="khaki")
ax6.plot(Expert_LAI_X[::step],Expert_LAI_data[::step],color="lightblue")
ax6.plot(Influnce_LAI_X[::step],Influnce_LAI_data[::step],color="orange")

ax7=fig.add_subplot(2,4,7)
ax7.plot(RF_SW_IN_X,RF_SW_IN_data,color='lightgreen')
ax7.plot(Attention_SW_IN_X[::step],Attention_SW_IN_data[::step],color="khaki")
ax7.plot(Expert_SW_IN_X[::step],Expert_SW_IN_data[::step],color="lightblue")
ax7.plot(Influnce_SW_IN_X[::step],Influnce_SW_IN_data[::step],color="orange")

ax8=fig.add_subplot(2,4,8)
ax8.plot(RF_RH_X,RF_RH_data,color='lightgreen')
ax8.plot(Attention_RH_X[::step],Attention_RH_data[::step],color="khaki")
ax8.plot(Expert_RH_X[::step],Expert_RH_data[::step],color="lightblue")
ax8.plot(Influnce_RH_X[::step],Influnce_RH_data[::step],color="orange")

ax1.set_ylabel('SHAP (mm/d)', fontsize=16)
ax1.set_xlabel('DEM (m)', fontsize=16)
ax2.set_xlabel('CLAY (%)', fontsize=16)
ax3.set_xlabel('acc_rain7 (mm)', fontsize=16)
ax4.set_xlabel('Month', fontsize=16)
ax5.set_ylabel('SHAP (mm/d)', fontsize=16)
ax5.set_xlabel('SIF (W/m$^{2}$·μm·sr)', fontsize=16)
ax6.set_xlabel('LAI (m$^{2}$/m$^{2}$)', fontsize=16)
ax7.set_xlabel('SW_IN (W/m$^{2}$)', fontsize=16)
ax8.set_xlim(0,100)
ax8.set_xlabel('RH (%)', fontsize=16)
ax1.legend(fontsize=15)

plt.subplots_adjust(top=0.95, bottom=0.15, right=0.95, left=0.05,hspace=0.2)
plt.margins(0, 0)
plt.show()