import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

'''
filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_base.csv"
df = pd.read_csv(filepath, header=None)
headers = ['',"Manufacturer", "Category", "Screen", "GPU", "OS", "CPU_core", "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg" and "Price"]
df.columns = headers
'''
path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/automobileEDA.csv"
df = pd.read_csv(path)

lm = LinearRegression() #model using "engine-size" as the independent variable and "price" as the dependent variable
X = df[['engine-size']]
Y = df['price']
lm.fit(X, Y)
cd = lm.score(X, Y)
b0 = lm.intercept_
b1 = lm.coef_[0]
print(f'intercept = {b0}, coef = {b1}, coeff_0f_determination = {cd}')
print(f'evacuation is:')
print(f'Yhat = {b0} + {b1} * df["engine-size"]')

lm1 = LinearRegression() #Multiple Linear Regression
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
lm1.fit(Z, Y)
cd1 = lm1.score(Z, Y)
b0 = lm1.intercept_
b1, b2, b3, b4 = lm1.coef_[0], lm1.coef_[1], lm1.coef_[2], lm1.coef_[3]
print(f'intercept = {b0}, coef = {lm.coef_}, coeff_0f_determination = {cd1}')
print(f'evacuation is:')
print(f'Yhat = {b0} + {b1} * horsepower + {b2} * curb-weight + {b3} * engine-size + {b4} * highway-mpg')

lm2 = LinearRegression() #the response variable is "price", and the predictor variable is "normalized-losses" and "highway-mpg"
Z2 = df[['normalized-losses', 'highway-mpg']]
lm2.fit(Z2, Y)
cd2 = lm2.score(Z2, Y)
b0 = lm2.intercept_
b1, b2 = lm2.coef_[0], lm2.coef_[1]
print(f'intercept = {b0}, coef = {lm.coef_}, coeff_0f_determination = {cd2}')
print(f'evacuation is:')
print(f'Yhat = {b0} + {b1} * normalized-losses + {b2} * highway-mpg')

df1 = df[["peak-rpm", "highway-mpg", 'price']]
print(df1.corr(numeric_only=True))

width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(x=df['highway-mpg'], y=df['price']) #Residual Plot
plt.show()

sns.regplot(x="highway-mpg", y="price", data=df) #Regression Plot
plt.show()

# Polynomial Regression and Pipelines
#use the following function to plot the data:

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()

x = df['highway-mpg']
y = df['price']
f = np.polyfit(x, y, 3) #fit the polynomial 3th order using the function polyfit, then
p = np.poly1d(f)
print('p:', p)
PlotPolly(p, x, y, 'highway-mpg') #use the function poly1d to display the polynomial function.

#fit the polynomial 4th order
f11 = np.polyfit(x, y, 4)
p11 = np.poly1d(f11)
print('p11:', p11)
PlotPolly(p11, x, y, 'noname123')

#multivariate polynomial 2nd order
pr = PolynomialFeatures(degree=2)
Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
Z_pr = pr.fit_transform(Z)
print(Z.shape)
print(Z_pr.shape)
print(Z, Z_pr)

#Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline.
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)
Z = Z.astype(float)
pipe.fit(Z, y)
ypipe = pipe.predict(Z)
print('ypipe')
print(ypipe)

#Measures for In-Sample Evaluation

#Model 1: Simple Linear Regression
lm.fit(X, Y) #highway_mpg_fit
# Find the R^2
print('Model 1: Simple Linear Regression')
print('The R-square is: ', lm.score(X, Y))
Yhat = lm.predict(X)
print('The output of the first four predicted value is: ', Yhat[0:4])
mse = mean_squared_error(df['price'], Yhat)
print('The mean square error of price and predicted value is: ', mse)

#Model 2: Multiple Linear Regression

# fit the model
lm.fit(Z, df['price'])
# Find the R^2
print('Model 2: Multiple Linear Regression')
print('The R-square is: ', lm.score(Z, df['price']))
Y_predict_multifit = lm.predict(Z)
print('The mean square error of price and predicted value using multifit is: ', \
      mean_squared_error(df['price'], Y_predict_multifit))

#Model 3: Polynomial Fit
from sklearn.metrics import r2_score
r_squared = r2_score(y, p(x))
print('Model 3: Polynomial Fit')
print('The R-square value is: ', r_squared)
mse = mean_squared_error(df['price'], p(x))
print('The mean square error of price and predicted value using Polynomial Fit is: ', mse)