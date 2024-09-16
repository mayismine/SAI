import copy
import matplotlib as mpl
from matplotlib import gridspec
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal, osr, ogr
from scipy.interpolate import make_interp_spline
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
import matplotlib.patches as mpatches

gstl = gridspec.GridSpec(2, 4)
gstl.update(top=0.88, bottom=0.14,left=0.06,right=0.85)
gstr = gridspec.GridSpec(2, 1)
gstr.update(top=0.877, bottom=0.143,left=0.87,right=0.92)
gsdl = gridspec.GridSpec(1,4)
gsdl.update(top=0.12, bottom=0.05,left=0.06,right=0.85)

def add_north(ax, labelsize=20, loc_x=0.98, loc_y=1.04, width=0.02, height=0.06, pad=0.14):
    minx, maxx = ax.get_xlim()
    miny, maxy = ax.get_ylim()
    ylen = maxy - miny
    xlen = maxx - minx
    left = [minx + xlen*(loc_x - width*.5), miny + ylen*(loc_y - pad)]
    right = [minx + xlen*(loc_x + width*.5), miny + ylen*(loc_y - pad)]
    top = [minx + xlen*loc_x, miny + ylen*(loc_y - pad + height)]
    center = [minx + xlen*loc_x, left[1] + (top[1] - left[1])*.4]
    triangle = mpatches.Polygon([left, top, right, center], color='k')
    ax.text(s='N',
            x=minx + xlen*loc_x,
            y=miny + ylen*(loc_y - pad + height),
            fontsize=labelsize,
            horizontalalignment='center',
            verticalalignment='bottom')
    ax.add_patch(triangle)


def add_scalebar(ax,lon0, lat0, length):
    ax.hlines(y=lat0, xmin=lon0, xmax=lon0 + length / 111, colors="black", ls="-", lw=1.5, label='%d km' % (length))
    ax.vlines(x=lon0, ymin=lat0 - 0.45, ymax=lat0 + 2, colors="black", ls="-", lw=1.5)
    ax.vlines(x=lon0 + length / 2 / 111, ymin=lat0 - 0.45, ymax=lat0 + 2, colors="black", ls="-", lw=1.5)
    ax.vlines(x=lon0 + length / 111, ymin=lat0 - 0.45, ymax=lat0 + 2, colors="black", ls="-", lw=1.5)
    ax.text(lon0 + length / 111, lat0 + 3, '%d' % (length), horizontalalignment='center',fontsize=15)
    ax.text(lon0 + length / 2 / 111, lat0 + 3, '%d' % (length / 2), horizontalalignment='center',fontsize=15)
    ax.text(lon0, lat0 + 3, '0', horizontalalignment='center',fontsize=15)
    ax.text(lon0 + length / 111 * 1.15, lat0+3, 'km', horizontalalignment='center',fontsize=15)

def ET_month_tif_2_png():
    year = 2000
    path = r'results\global\SAI\tiff'
    for year_off in range(22):
        year = 2000 + year_off
        for i in range(12):
            plt.rc('font', family='Times New Roman')
            plt.rcParams['font.size'] = 15
            fig = plt.figure(figsize=(26, 14))
            plt.subplots_adjust(wspace=0, hspace=0)
            input_path = path + '\\' + str(year) + str(i+1).zfill(2) + '.tif'
            data_set = gdal.Open(input_path).ReadAsArray(0, 0, 1440, 720)
            ax1 = fig.add_axes([0.06, 0.14, 0.79, 0.74], projection=ccrs.PlateCarree())
            ax1.coastlines()
            ax1.add_feature(cfeature.COASTLINE, lw=0.1)
            ax1.add_feature(cfeature.RIVERS, lw=0.05)
            cmap2 = copy.copy(mpl.cm.RdYlBu)
            cmap2.set_under('red')
            cmap2.set_over('blue')
            norm2 = mpl.colors.TwoSlopeNorm(vmin=0,vcenter=50, vmax=150)
            im2 = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2)
            ax1.set_xticks([-180,-120,-60,0,60,120,180])
            ax1.set_xticklabels(['180°E', '120°E', '160°E', '0°', '60°W', '120°W', '180°W'],fontsize=20)
            ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90])
            ax1.set_yticklabels(['90°N', '60°N', '30°N', '0°', '30°S', '60°S', '90°S'],fontsize=20)
            ax1.grid(True, color="black",linestyle='--')
            ax1.imshow(data_set, cmap=cmap2, extent=[-180, 180, -90, 90], norm=norm2)
            patch = ax1.patch
            patch.set_color("whitesmoke")
            data_set[data_set<=1] = np.nan

            histogram_row = np.nanmean(data_set,axis=1)
            histogram_row[np.isnan(histogram_row)] = 0
            histogram_column = np.nanmean(data_set, axis=0)
            histogram_column[np.isnan(histogram_column)] = 0
            ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax2 = plt.subplot(gstr[:,:])
            x_row = np.arange(720)
            model = make_interp_spline(x_row,histogram_row)
            y_row = model(x_row)
            ax2.set_ylim(0, 720)
            ax2.invert_yaxis()
            ax2.set_yticks([])
            ax2.set_yticks([len(histogram_row)/6,len(histogram_row)/6*2,len(histogram_row)/6*3,len(histogram_row)/6*4,len(histogram_row)/6*5])
            ax2.set_yticklabels([])
            ax2.tick_params(axis="y", direction="in")
            ax2.set_xticks(np.arange(0, int((np.max(histogram_row) // 50) + 1) * 50+1, 50))
            ax2.set_xticklabels(np.arange(0, int((np.max(histogram_row) // 50) + 1) * 50+1, 50))
            ax2.set_xlim(0,int((np.max(histogram_row) // 50) + 1) * 50+1)
            ax2.grid(True, color ="black",linestyle='--')
            ax2.plot(np.where(y_row <= 1, np.nan, y_row), x_row,color ='black', linewidth=3)
            patch = ax2.patch
            patch.set_color("whitesmoke")
            ax3 = plt.subplot(gsdl[:, :])
            x_column = np.arange(1440)
            model = make_interp_spline(x_column, histogram_column)
            y_column = model(x_column)
            ax3.set_xlim(0,1440)
            ax3.set_ylim(0,int((np.max(histogram_column)//50)+1)*50+1)
            ax3.invert_yaxis()
            ax3.yaxis.tick_right()
            ax3.set_xticks([len(histogram_column)/6,len(histogram_column)/6*2,len(histogram_column)/6*3,len(histogram_column)/6*4,len(histogram_column)/6*5])
            ax3.set_xticklabels([])
            ax3.tick_params(axis="x", direction="in")
            ax3.set_yticks(np.arange(0,int((np.max(histogram_column)//50)+1)*50+1,50))
            ax3.set_yticklabels(np.arange(0,int((np.max(histogram_column)//50)+1)*50+1,50))
            ax3.grid(True, color="black",linestyle='--')
            ax3.plot(x_column, np.where(y_column <=1 , np.nan, y_column),color ='black', linewidth=3)
            patch = ax3.patch
            patch.set_color("whitesmoke")
            ax4 = fig.add_axes([0.55, 0.28, 0.28, 0.02])
            plt.rcParams['font.size'] = 20
            cbar2 = fig.colorbar(
                im2, cax=ax4, orientation='horizontal',
                extend='both', ticks=np.linspace(0, 250, 11),
                label='ET Colorbar (mm/month)',
            )
            plt.suptitle(str(year)+str(i+1).zfill(2)+' Global ET (mm/month)',x = 0.5,y = 0.93,fontsize = 25)
            ax5 = fig.add_axes([0.872, 0.05, 0.05, 0.05])
            ax5.text(0,0.5,'ET (mm/month)',fontsize=20)
            ax5.spines['top'].set_visible(False)
            ax5.spines['left'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            ax5.spines['bottom'].set_visible(False)
            ax5.set_xticks([])
            ax5.set_yticks([])
            add_north(ax1)
            add_scalebar(ax1, 95, -80, 6000)
            plt.savefig(r'', dpi=100)
            # plt.show()

def ET_MK_tif_2_png():
    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.size'] = 15
    fig = plt.figure(figsize=(26, 14))
    plt.subplots_adjust(wspace=0, hspace=0)
    input_path = r'analyze and figure\MK\slope.tif'
    data_set = gdal.Open(input_path).ReadAsArray(0, 0, 1440, 720)
    ax1 = fig.add_axes([0.06, 0.14, 0.79, 0.74], projection=ccrs.PlateCarree())
    ax1.coastlines()
    ax1.add_feature(cfeature.COASTLINE, lw=0.1)
    ax1.add_feature(cfeature.RIVERS, lw=0.05)
    cmap2 = copy.copy(mpl.cm.coolwarm_r)
    cmap2.set_under('red')
    cmap2.set_over('blue')
    norm2 = mpl.colors.TwoSlopeNorm(vmin=-8, vcenter=0, vmax=8)
    im2 = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2)
    ax1.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax1.set_xticklabels(['180°E', '120°E', '160°E', '0°', '60°W', '120°W', '180°W'],fontsize=20)
    ax1.set_yticks([-90, -60, -30, 0, 30, 60, 90])
    ax1.set_yticklabels(['90°N', '60°N', '30°N', '0°', '30°S', '60°S', '90°S'],fontsize=20)
    ax1.grid(True, color="black", linestyle='--')
    ax1.imshow(data_set, cmap=cmap2, extent=[-180, 180, -90, 90], norm=norm2)
    patch = ax1.patch
    patch.set_color("whitesmoke")

    histogram_row = np.nanmean(data_set, axis=1)
    index_row = np.isnan(histogram_row)
    histogram_row[np.isnan(histogram_row)] = 0
    histogram_column = np.nanmean(data_set, axis=0)
    index_col = np.isnan(histogram_column)
    histogram_column[np.isnan(histogram_column)] = 0
    ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2 = plt.subplot(gstr[:, :])
    x_row = np.arange(720)
    model = make_interp_spline(x_row, histogram_row)
    y_row = model(x_row)
    ax2.set_ylim(0, 720)
    ax2.invert_yaxis()
    ax2.set_yticks([])
    ax2.set_yticks(
        [len(histogram_row) / 6, len(histogram_row) / 6 * 2, len(histogram_row) / 6 * 3, len(histogram_row) / 6 * 4,
         len(histogram_row) / 6 * 5])
    ax2.set_yticklabels([])
    ax2.tick_params(axis="y", direction="in")
    ax2.set_xticks(np.array([-2,0,2,4,6]))
    ax2.set_xlim(-2, 6)
    ax2.grid(True, color="black", linestyle='--')
    ax2.plot(np.where(index_row, np.nan, y_row), x_row, color='black', linewidth=3)
    patch = ax2.patch
    patch.set_color("whitesmoke")
    ax3 = plt.subplot(gsdl[:, :])
    x_column = np.arange(1440)
    model = make_interp_spline(x_column, histogram_column)
    y_column = model(x_column)
    ax3.set_xlim(0, 1440)
    ax3.set_ylim(-2, 6)
    ax3.invert_yaxis()
    ax3.yaxis.tick_right()
    ax3.set_xticks([len(histogram_column) / 6, len(histogram_column) / 6 * 2, len(histogram_column) / 6 * 3,
                    len(histogram_column) / 6 * 4, len(histogram_column) / 6 * 5])
    ax3.set_xticklabels([])
    ax3.tick_params(axis="x", direction="in")
    ax3.set_yticks(np.array([-2,0,2,4,6]))
    ax3.grid(True, color="black", linestyle='--')
    ax3.plot(x_column, np.where(index_col, np.nan, y_column), color='black', linewidth=3)
    patch = ax3.patch
    patch.set_color("whitesmoke")
    # ax4 = plt.subplot(gsdr[:, :])
    ax4 = fig.add_axes([0.55, 0.28, 0.28, 0.02])
    cbar2 = fig.colorbar(
        im2, cax=ax4, orientation='horizontal',
        extend='both', ticks=np.linspace(-8, 8, 5),
        label='ET Colorbar (mm/year)'
    )
    plt.suptitle('Global ET MK slope (mm/year)', x=0.5, y=0.93, fontsize=25)
    ax5 = fig.add_axes([0.872, 0.05, 0.05, 0.05])
    ax5.text(0, 0.5, 'ET (mm/year)',fontsize=20)
    ax5.spines['top'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.set_xticks([])
    ax5.set_yticks([])
    add_north(ax1)
    add_scalebar(ax1, 95, -80, 6000)
    plt.savefig(r'', dpi=100)
    plt.show()

ET_month_tif_2_png()



