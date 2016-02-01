"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor

################################
### ADD EXTRA LIBRARIES HERE ###
################################
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
    median_absolute_error, make_scorer)
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import NearestNeighbors


#assuming prices are in $1,000's
HOUSE_PRICE_MULTIPLIER = 1000


def load_data():
    """Load the Boston dataset."""
    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    ###################################
    ### Step 1. YOUR CODE GOES HERE ###
    ###################################

    number_of_houses, number_of_features = city_data.data.shape
    min_house_price = np.min(housing_prices) * HOUSE_PRICE_MULTIPLIER
    max_house_price = np.max(housing_prices) * HOUSE_PRICE_MULTIPLIER
    mean_house_price = np.mean(housing_prices) * HOUSE_PRICE_MULTIPLIER
    median_house_price = np.median(housing_prices) * HOUSE_PRICE_MULTIPLIER
    house_prices_std = np.std(housing_prices)

    print("\n***** EXPLORING CITY DATA *****")
    #print(city_data.DESCR)
    print("Number of Houses: {}".format(number_of_houses))
    print("Number of features: {}".format(number_of_features))
    print("Minimum Housing Price: ${:,.2f}".format(min_house_price))
    print("Maximum Housing Price: ${:,.2f}".format(max_house_price))
    print("Mean Housing Price: ${:,.2f}".format(mean_house_price))
    print("Median Housing Price: ${:,.2f}".format(median_house_price))
    print("Standard Deviation: {:.3f}\n".format(house_prices_std))

    # Please calculate the following values using the Numpy library
    # Size of data (number of houses)?
    # Number of features?
    # Minimum price?
    # Maximum price?
    # Calculate mean price?
    # Calculate median price?
    # Calculate standard deviation?

    #further exploration of the dataset and it's features
    city_dataframe = pd.DataFrame(city_data.data)
    city_dataframe.columns = city_data.feature_names

    #plot each feature vs the house price
    print('***** FEATURES vs TARGET PLOTS *****')
    pl.figure()
    for index, feature in enumerate(city_dataframe.columns.values):
        ax = pl.subplot(3, 5, index + 1)
        ax.set_title(str(feature) + " vs Housing Price")
        ax.set_ylabel("Price in $10,000's")
        ax.set_xlabel(feature)
        pl.scatter(city_dataframe[str(feature)], city_data.target,
                   s=20, c='c', alpha=.5)
    pl.show()

    #further exploration of the model prediction
    #comparing the prediction's features to the averages in it's price range
    price_range = np.where(np.logical_and(city_data.target >= 21, city_data.target <= 22))
    price_range_dataframe = city_dataframe.loc[price_range].reset_index(drop=True)
    print('\n***** PRICE RANGE DATAFRAME ******')
    print('Feature averages between $21,000.00 & $22,000.00')
    print(price_range_dataframe.mean(), '\n')

def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################
    X_train, X_test, y_train, y_test = (
        train_test_split(X, y, train_size=0.70)
        )

    return X_train, y_train, X_test, y_test


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""

    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################
    acc = mean_squared_error(label, prediction)
    #acc = mean_absolute_error(label, prediction)
    #acc = median_absolute_error(label, prediction)

    # The following page has a table of scoring functions in sklearn:
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    return acc


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print ("Decision Tree with Max Depth: ", depth)

    for i, s in enumerate(sizes):
        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, color='r', label = 'test error')
    pl.plot(sizes, train_err, lw=2, color='c', label = 'training error')
    #pl.axvline(x=337, ymin=0, ymax=100, linewidth=2, color='b', label='3 k-fold')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print ("***** MODEL COMPLEXITY PLOT *****\n")

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, color='r', label = 'test error')
    pl.plot(max_depth, train_err, lw=2, color='c', label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()


def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################

    # 1. Find an appropriate performance metric. This should be the same as the
    # one used in your performance_metric procedure above:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    scorer = make_scorer(mean_squared_error, greater_is_better=False)


    # 2. We will use grid search to fine tune the Decision Tree Regressor and
    # obtain the parameters that generate the best training performance. Set up
    # the grid search object here.
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
    reg = GridSearchCV(regressor, parameters, scoring=scorer)
    reg.fit(X, y)

    # Fit the learner to the training data to obtain the best parameter set
    print ("***** FINAL MODEL *****")
    print (reg.fit(X, y), '\n')
    print ("Best Max Depth: {}".format(reg.best_params_))


    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = reg.predict(x)
    print ("House: " + str(x))
    print ("Prediction: ${:,.2f}".format(y[0] * HOUSE_PRICE_MULTIPLIER))

    #find the nearest neighbors
    indices = find_nearest_neighbors(x, X)
    print("Mean of nearest neighbors: ${:,.2f}".format(
            city_data.target[indices].mean() * HOUSE_PRICE_MULTIPLIER))


def find_nearest_neighbors(x, X):
    """Find the nearest neighbors of the vector (x) in the dataset (X)."""
    neighbors = NearestNeighbors(n_neighbors = 10).fit(X)
    distances, indices = neighbors.kneighbors(x)
    return indices.tolist()[0]


def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs
    print("***** LEARNING CURVE *****")
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)
    print()

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()
