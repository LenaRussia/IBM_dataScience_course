import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from ipywidgets import interact, interactive, fixed, interact_manual


#import dataset cars
df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/Data%20files/module_5_auto.csv')
df = df._get_numeric_data()      #First, let's only use numeric data:
df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)     #Let's remove the columns 'Unnamed:0.1' and 'Unnamed:0' since they do not provide any value to the models.
#print(df.head())

# Functions for Plotting

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()

def PollyPlot(xtrain, xtest, y_train, y_test, lr, poly_transform):     # training data / testing data / lr:  linear regression object / poly_transform:  polynomial transformation object
    width = 12
    height = 10
    plt.figure(figsize=(width, height))

    xmax = max([xtrain.values.max(), xtest.values.max()])
    xmin = min([xtrain.values.min(), xtest.values.min()])

    x = np.arange(xmin, xmax, 0.1)

    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    plt.close()


#PART 1: TRAINING AND TESTING

y_data = df['price']   # the target data is 'price'
x_data = df.drop('price', axis=1)   # Drop y_data ('price') in dataframe x_data

# randomly (random_state=1) split data into training and testing data:
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=1)
#print("number of test samples :", y_test.shape[0])
#print("number of training samples:", y_train.shape[0])

lre = LinearRegression()   #create Multiple Linear Regression object
lre.fit(x_train[['horsepower']], y_train)   #We fit the model using the feature "horsepower":
r2_train = lre.score(x_train[['horsepower']], y_train)   #R^2 on the train data
r2_test = lre.score(x_test[['horsepower']], y_test)   #R^2 on the test data calculated to predict how this model will perform in the real world
#print(f'Coefficient of determination calculated on train data is {r2_train}')
#print(f'Coefficient of determination calculated on test data is {r2_test}') #r2 is low means that we maybe don't have enough of test data

# CROSS-VALIDATION SCORE
from sklearn.model_selection import cross_val_score
Rcross = cross_val_score(lre, x_data[['horsepower']], y_data, cv=4) # The default scoring is R^2. Each element in the array has the average R^2 value for the fold
#print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std()) # calculate the average and standard deviation of our estimate
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error') # We can use negative squared error as a score by setting the parameter  'scoring' metric to 'neg_mean_squared_error'.

# PREDICT THE OUTPUT
from sklearn.model_selection import cross_val_predict
yhat = cross_val_predict(lre, x_data[['horsepower']], y_data, cv=4) #The function splits up the data into the specified number of folds, with one fold for testing and the other folds are used for training.




#PART 2: OVERFITTING, UNDERFITTING AND MODEL SELECTION
lr = LinearRegression()     #create Multiple Linear Regression object
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)
yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
#print(yhat_train[0:5])
yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
#print(yhat_test[0:5])


Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title) #Plot of predicted values using the training data compared to the actual values of the training data.

Title = 'Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, yhat_test, "Actual Values (Test)", "Predicted Values (Test)", Title)    #Plot of predicted value using the test data compared to the actual values of the test data.

#Let's see if polynomial regression also exhibits a drop in the prediction accuracy when analysing the test dataset
x_train_p, x_test_p, y_train_p, y_test_p = train_test_split(x_data, y_data, test_size=0.45, random_state=0)
pr = PolynomialFeatures(degree=5)
x_train_pr = pr.fit_transform(x_train_p[['horsepower']]) #perform a degree 5 polynomial transformation on the feature 'horsepower' for the train set of x
x_test_pr = pr.fit_transform(x_test_p[['horsepower']]) #perform a degree 5 polynomial transformation on the feature 'horsepower' for the test set of x

poly = LinearRegression() #create a Linear Regression model "poly"
poly.fit(x_train_pr, y_train_p) #train it with transformed x_train_p with a degree 5 polynomial transformation
yhat_pr = poly.predict(x_test_pr) #the output of our model using the method "predict"

#print("Predicted values:", yhat_pr[0:4])
#print("True values:", y_test_p[0:4].values)

PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly,pr)  #use the function "PollyPlot" to display the training data, testing data, and the predicted function.
R2_poly_train = poly.score(x_train_pr, y_train_p)  #R2 for the training data transformed with 5 degree polynomial transformation
R2_poly_test = poly.score(x_test_pr, y_test_p)   #R2 for the testing data transformed with 5 degree polynomial transformation
#print(f"R2 for train data: {R2_poly_train}")
#print(f"R2 for test data: {R2_poly_test}") #A negative R^2 is a sign of overfitting.

#Let's see how the R^2 changes on the test data for different order polynomials and then plot the results:
Rsqu_test = []
order = [1, 2, 3, 4]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train_p[['horsepower']])
    x_test_pr = pr.fit_transform(x_test_p[['horsepower']])
    lr.fit(x_train_pr, y_train_p)
    Rsqu_test.append(lr.score(x_test_pr, y_test_p))
plt.plot(order, Rsqu_test)
plt.xlabel('order')
plt.ylabel('R^2')
plt.title('R^2 Using Test Data')
plt.text(3, 0.75, 'Maximum R^2 ')
plt.show()


#The following interface allows you to experiment with different polynomial orders and different amounts of data:
def f(order, test_data):
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_data, random_state=0)
    pr = PolynomialFeatures(degree=order)
    x_train_pr = pr.fit_transform(x_train[['horsepower']])
    x_test_pr = pr.fit_transform(x_test[['horsepower']])
    poly = LinearRegression()
    poly.fit(x_train_pr, y_train)
    PollyPlot(x_train['horsepower'], x_test['horsepower'], y_train, y_test, poly, pr)

interact(f, order=3, test_data=0.2)




# POLYNOMIAL TRANSFORMATIONS WITH MORE THAN ONE FEATURE
pr1 = PolynomialFeatures(degree=2) #Create a "PolynomialFeatures" object "pr1" of degree two
x_train_pr1 = pr1.fit_transform(x_train_p[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
x_test_pr1 = pr1.fit_transform(x_test_p[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(x_train_p.shape, x_test_p.shape, x_train_pr1.shape, x_test_pr1.shape) #shape shows how many dimensions does the new feature have, there are now 15 features
poly1 = LinearRegression()   #create linear regression object
poly1.fit(x_train_pr1, y_train_p)   #train linear regression object with training set of data
yhat_p1 = poly1.predict(x_test_pr1) #predicts an output of polynomial features using testing set of data


Title = 'Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test, yhat_p1, "Actual Values (Test)", "Predicted Values (Test)", Title) #use the predefined upper function to display the plot


#PART 3: RIDGE REGRESSION

from sklearn.linear_model import Ridge
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
x_test_pr = pr.fit_transform(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg','normalized-losses','symboling']])
RigeModel = Ridge(alpha=1)  #create a Ridge regression object, setting the regularization parameter (alpha) to 0.1
RigeModel.fit(x_train_pr, y_train)  #fit the model using the method fit, like regular regression
yhat_ridge = RigeModel.predict(x_test_pr)
print('predicted:', yhat_ridge[0:4])
print('test set :', y_test[0:4].values)
print(f'the R^2: {RigeModel.score(x_test_pr, y_test)}')


#PART 4: GRID SEARCH

from sklearn.model_selection import GridSearchCV
parameters1 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]   #create a dictionary of parameter values
RR = Ridge()  #Create a Ridge regression object
Grid1 = GridSearchCV(RR, parameters1, cv=4) #Create a ridge grid search object
Grid1.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)  #Fit the model
BestRR = Grid1.best_estimator_   #The object finds the best parameter values on the validation data. We can obtain the estimator with the best parameters and assign it to the variable BestRR
grid_score = BestRR.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)   #test our model on the test data
print(grid_score)

#Perform a grid search for the alpha parameter and the normalization parameter, then find the best values of the parameters:
parameters2 = [{'alpha': [0.001, 0.1, 1, 10, 100, 1000, 10000, 100000, 100000]}]
Grid2 = GridSearchCV(Ridge(), parameters2, cv=4)
Grid2.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
best_alpha = Grid2.best_params_['alpha']
best_ridge_model = Ridge(alpha=best_alpha)
best_ridge_model.fit(x_data[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_data)
grid_score2 = best_ridge_model.score(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_test)   #test our model on the test data
print(grid_score2)  # grid_score = grid_score2 i.e. this is doing the same as previous


