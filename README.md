# APS1070_Project_3: COVID-19 2019 Dataset – PCA

**Key Takeaways:**
-	Time series of COVID cases, need to standardize for population sizes, can see the trends of how different countries handle it
-	Apply PCA by calculating the covariance matrix, eigenvalues, and eigenvectors with np.linalg.eigh, determining how many principal components are required to cover the dataset variance by cumulative variance
-	Covariance Matrix, C = np.dot(cases_std.T, cases_std) / (n-1), 
-	Eigen Decomposition, eigenValues, eigenVectors = np.linalg.eigh(C), sort to determine the largest eigenvalues
-	Data Reconstruction, reconstruction of incremental PCs, along with residual errors and RMSE
-	W = eigenVectors[:, 0:i] #Using number of PCs
-	    projX = np.dot(cases_std, W) #Projection
-	    ReconX = np.dot(projX, W.T) #Reconstruct data
-	Incremental reconstruction of the original time series
-	    InvReconX = fit.inverse_transform(ReconX)
-	    InvReconstruction.append(InvReconX[list(cases_raw.index).index(country)]) #Add all the cumulative eigenvectors to a list
-	Apply PCA for faster computations
-	MNIST Dataset, reshape each to a 28x28 matrix to see the shape
o	Calculate the covariance, Eigen decomposition, and sort
o	PCA helps by filtering out the noise, not as blurry

Three different datasets to analyze:
1. Covid-19 dataset to analyze the number of total cases for different countries at the end of each day (Part 1-3)
2. The COVID-19 dataset is the total number of recovered cases for each country at the end of each day. (Part 4-5)
3. PCA to images using the MNIST dataset of handwritten digits. (Part 6)

**Part 1: Getting Started**

The first data set is 192 rows (countries) and 397 columns, representing the days since COVID-19 started
- Time series are plotted for some countries, and each country is normalized.
  
![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/03777cfd-e314-4ab1-94f2-4a4919fd34ac)

Standardizing the data: 

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/01ff076d-4228-4189-a570-0b018f685917)

Plotting standardized time series on multiple countries:

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/16b21c72-c45a-4b83-9a89-ca4a848a6de3)

By standardizing the data, the mean of each day is 0, if the curve goes up, the number of cases per day (ie. the rate) increases relative to the day before.

**Part 2: Applying Principal Component Analysis (PCA)**
-The dataset is 192 countries x 397 days (192, 397)

Calculating the covariance matrix:

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/d3f24c44-a6dd-4f2e-9279-680dd4adaebf)

Sort from highest to lowest:
- args = (-eigenValues).argsort()
- eigenValues = eigenValues[args]
- eigenVectors = eigenVectors[:, args]

Can determine how much of the variance can be explained with the first few eigenvectors (ie. the largest ones):

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/3825c457-2772-4c41-8dc9-b3107b61d652)

Can plot cumExpVar[0:10], only need 4 Principal Components (PCs) to cover 99% of the variance

Plotting the first 16 PC's as time series to see if it mimics the shape

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/6e853b0c-04b9-4bf6-ab7b-7d2c24cd05b3)
![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/10bd4b36-57ae-4097-9ffd-52c454460a56)

**Part 3: Data Reconstruction**

- Create a function that accepts the country name and plots the original and time series for the country, incremental reconstruction of the original time series, and the residual error (Root Mean Square Error)
  
![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/cac0c68f-e019-4dc8-bdf9-e7180ec9b53d)

- PC's amount used are 1, 2, 4, 8, and 16.
- Get only that country's data:
-   cases_raw.iloc[list(cases_raw.index).index(country)].plot(ax=axes[0,0], title = 'Raw Time Series'
-   cases_std_df.iloc[list(cases_std_df.index).index(country)].plot(ax=axes[0,1], title = 'Standardized Time Series')
- Find the W, which is the eigenvectors, projection which is a dot product of the data and its eigenvectors
- Reconstruct data by finding the dot product of Projection and W^Transposed
- Add all cumulative eigenvectors to a list
- Incremental reconstruction of the original time series by fit.inverse_transform(ReconX)
- Calculate Residual error by the differences in estimated and actual values
- RMSE = np.sqrt(np.sum(CumResiduals[i]**2)/m)
  
**Part 4: Time Series Analysis of Recovered Cases**

Preprocess Data: Using ffill method, using the previous data as the value

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/e8640da0-b8e1-4871-bdec-2f4a4d22f87b)

**Part 5: PCA on the dimensions of the samples using Eigen Decomposition**

Need to apply PCA since the dates (number of features) are greater than the number of samples (countries)

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/6275c17f-38ef-4516-be60-60feabb73fda)

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/53467724-beb3-4618-9881-c83753291278)

Another way to transform the data, calculate the covariance matrix using the transposed data. Calculated the eigenValues and the eigenVectors.
Same inverse function as above.

**Part 6: MNIST Dataset**

Dataset for hand-written digit recognition, each image has 28x28 pixels (array of 784), want to sue PCA to compress the images.
The dataset has 1000 arrays and 1000 targets with the correct number.

Find the covariance, and decomposition, and sort in descending error:

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/3294211e-ee5c-4c46-9073-a40fb0e19b4a)

The first 10 eigenvectors:

![image](https://github.com/Chengalex96/APS1070_Project_3/assets/81919159/3738534f-dd51-4fc3-acbb-c56ae7961087)

Takes the number of PC's from the list and tries to reconstruct the data of a random sample point. From the graphs displayed, you can distinguish the number around 20 PCs, therefore the compression ratio is:

CompressionRatio = xelements*ximages / (xelements*PCreq + ximages*PCreq + mu.shape[0]) #1000 * 784 / 20(784 + 1000 + 784) add the mean face vector used to standardize

About 21.5 times.





