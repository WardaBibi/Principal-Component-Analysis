from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt

# scaling function
def scaleMinMax(x):
    return((x - np.nanmin(x))/(np.nanmax(x) - np.nanmin(x)))
# def scaleCC(x):
#     return((x - np.nanpercentile(x,2))/(np.nanpercentile(x,98) - np.nanpercentile(x,2)))
ds = gdal.Open('full.tif')


r = ds.GetRasterBand(1).ReadAsArray()
g = ds.GetRasterBand(2).ReadAsArray()
b = ds.GetRasterBand(3).ReadAsArray()
print('bands', ds.RasterCount, 'rows', ds.RasterYSize, 'columns', ds.RasterXSize)

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


sorted_index = np.argsort(eigen_val)[::-1]
sortedEigVal = eigen_val[sorted_index]
sortedEigenVectors = eigen_vec[:,sorted_index]
print('\nSorted eigen values are \n ',sortedEigVal)
print('\nSorted eigen vectors are \n ',sortedEigenVectors)

nComponents = 3
principalComp = sortedEigenVectors[:,0:nComponents]
# X_Meaned=X_Meaned[:,0:nComponents]
reducedMatrix = np.dot(principalComp.transpose(), X_Meaned.transpose()).transpose()

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
