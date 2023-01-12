from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

# scaling function
def scaleMinMax(x):
    return((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)))
# def scaleCC(x):
#     return((x - np.nanpercentile(x,2))/(np.nanpercentile(x,98) - np.nanpercentile(x,2)))
ds = gdal.Open('full.tif')
# ds1 = gdal.Open('band1.tif')
# f1= ds1.GetRasterBand(1).ReadAsArray()
# ds2 = gdal.Open('band2.tif')
# f2= ds2.GetRasterBand(1).ReadAsArray()
# ds3 = gdal.Open('band3.tif')
# f3= ds3.GetRasterBand(1).ReadAsArray()
# f1=scaleMinMax(f1)
# f2=scaleMinMax(f2)
# f3=scaleMinMax(f3)
# allfff = np.dstack((f1,f2,f3))
# print(allfff)
# plt.figure()
# plt.imshow(allfff)
# plt.show()

r = ds.GetRasterBand(1).ReadAsArray()
g = ds.GetRasterBand(2).ReadAsArray()
b = ds.GetRasterBand(3).ReadAsArray()
print('bands', ds.RasterCount, 'rows', ds.RasterYSize, 'columns', ds.RasterXSize)
# print('bands',ds1.RasterCount,'rows',ds1.RasterYSize,'columns',ds1.RasterXSize)
# print('bands',ds2.RasterCount,'rows',ds2.RasterYSize,'columns',ds2.RasterXSize)
# print('bands',ds3.RasterCount,'rows',ds3.RasterYSize,'columns',ds3.RasterXSize)
ds = None
resultArray =[]
rMinMax = scaleMinMax(r)
gMinMax = scaleMinMax(g)
bMinMax = scaleMinMax(b)
rgbMinMax = np.dstack((rMinMax,gMinMax,bMinMax))
plt.figure()
plt.imshow(rgbMinMax)
plt.show()

a,b,c = rgbMinMax.shape
print('Original matrix is ',rgbMinMax)


array1 = np.array(rMinMax)
reshapedR=array1.reshape(1065984)
# here 1065984= size(rows*cols) of a single band
array2 = np.array(gMinMax)
reshapedG=array2.reshape(1065984)


array3 = np.array(bMinMax)
reshapedB=array3.reshape(1065984)

Mean = (reshapedB+reshapedG+reshapedR)/3
reshapedB=reshapedB-Mean
reshapedG=reshapedG-Mean
reshapedR=reshapedR-Mean
resultArray.append(reshapedB)
resultArray.append(reshapedG)
resultArray.append(reshapedR)

Matrix = np.array(resultArray)
Matrix = Matrix.transpose()

arrayN = np.array(Matrix)
n,col=arrayN.shape
print(col)
print(arrayN)
print('\n\n Mean is \n',Mean )
print('\nCompressed Matrix is :',Matrix)
X_Meaned = Matrix
Covariance = (((X_Meaned).transpose()).dot(X_Meaned))/(n-1)
print('\n\nRows and cols of Covariance matrix  are ',np.shape(Covariance))
print('\n Covariance Matrix is \n' , Covariance)

eigen_val,eigen_vec = np.linalg.eigh(Covariance)
print("\n\n Eigen Values are  \n ",eigen_val)
print("\n\n Eigen Vectors are \n ",eigen_vec)

#sorting
# def sortArray(arr):
#      for i in range(len(arr)):
#          for j in range (i+1,len(arr)):
#              if(arr[i] < arr[j]):
#                  temp = arr[i]
#                  arr[i] = arr[j]
#                  arr[j] = temp
#      return arr
# SortedEigVal = sortArray(eigen_val)
# print('\nSorted eigen values are \n ',SortedEigVal)
sorted_index = np.argsort(eigen_val)[::-1]
sortedEigVal = eigen_val[sorted_index]
sortedEigenVectors = eigen_vec[:,sorted_index]
print('\nSorted eigen values are \n ',sortedEigVal)
print('\nSorted eigen vectors are \n ',sortedEigenVectors)

nComponents = 3
principalComp = sortedEigenVectors[:,0:nComponents]
# X_Meaned=X_Meaned[:,0:nComponents]
reducedMatrix = np.dot(principalComp.transpose(), X_Meaned.transpose()).transpose()
# x=1041
# y=1024
# pro= x*y
# pro= nComponents*nComponents
# # pro=pro/3
# pro=int(pro)
# for i in range(2,pro+1):
#     if((pro%(i*3))==0):
#         row=(pro/(i*3))
#         col=i
#         if(abs(row-col)==17):
#          print("The smallest divisor is:",i)   # print the Smallest divisor
#          break
# row=int(row)
# reducedMatrix_reshaped=reducedMatrix.reshape(row,col,3)

reducedMatrix_reshaped=reducedMatrix.reshape(1041,1024,3)
reducedMatrix_reshaped=scaleMinMax(reducedMatrix_reshaped)
print('\nReduced Matrix is \n\n',reducedMatrix_reshaped)
print('\nprincipal Components are \n\n',principalComp)

plt.figure()
plt.imshow(reducedMatrix_reshaped)
plt.show()







# Error analysis

sum = sortedEigVal[0]+sortedEigVal[1]+sortedEigVal[2]
pc1Explained= (sortedEigVal[0])/sum
pc1Explained=pc1Explained*100
print('\n\nVariance explained by PC1',pc1Explained,'%')
pc2Explained= (sortedEigVal[1])/sum
pc2Explained=pc2Explained*100
print('Variance explained by PC2',pc2Explained,'%')
pc3Explained= (sortedEigVal[2])/sum
pc3Explained=pc3Explained*100
print('Variance explained by PC3',pc3Explained,'%')


MSE = np.square(np.subtract(rgbMinMax,reducedMatrix_reshaped)).mean()
# print('\n\nMean squared error with ',nComponents,' is',MSE)
print('\n\nMean squared error with 1 principal components is 0.1867047580111931')
print('\n\nMean squared error with 3 principal components is 0.14318196553479803')
