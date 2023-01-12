from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

# scaling function
def scaleMinMax(x):
    return((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)))

# opening multi band image
ds = gdal.Open('full.tif')
ds1=gdal.Open('band1.TIF')
f = ds1.GetRasterBand(1).ReadAsArray()
f = scaleMinMax(f)


print('bands',ds1.RasterCount,'rows',ds1.RasterYSize,'columns',ds1.RasterXSize)
plt.figure()
plt.imshow(f)
plt.show()

# extracting each band and reading it as matrix
#rgb:123
r = ds.GetRasterBand(1).ReadAsArray()
g = ds.GetRasterBand(2).ReadAsArray()
b = ds.GetRasterBand(3).ReadAsArray()

# scaling , Reason : float values should be between 0 and 1 (0 -255 for integers) ,
# any values > than 1 are clipped to 1 so image get changed.
# In order to avoid this normalizes your data by scaling.
rMinMax = scaleMinMax(r)
gMinMax = scaleMinMax(g)
bMinMax = scaleMinMax(b)


# Task1:visualize single band


# Task2:concatenate:
# stacking all images upon each other
rgbMinMax = np.dstack((rMinMax,gMinMax,bMinMax))
plt.figure()
plt.imshow(rgbMinMax)
plt.show()



# task3: crop
# original coordiantes
# Upper Left  (    0.0,    0.0)
# Lower Left  (    0.0, 1041.0)
# Upper Right ( 1024.0,    0.0)
# Lower Right ( 1024.0, 1041.0)
# Center      (  512.0,  520.5)

upper_left_x = 216
upper_left_y = 49
lower_right_x = 813
lower_right_y = 995
window = (upper_left_x,upper_left_y,lower_right_x,lower_right_y)
#gdal.translate(dest,src,args)
gdal.Translate('cropped.tif','full.tif',projWin=window)

