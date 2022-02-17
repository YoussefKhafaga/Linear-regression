import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics


def read_data():
    return pd.read_csv("book1.csv")


if __name__ == '__main__':

    #data read
    data = read_data()
    x = data.iloc[:, :-1].values
    y = data.iloc[:, 1].values
    print(x,y)

    #splitting 20% for testing and 80% for training
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, train_size=0.8,
                                                                        random_state=0)
    #training the model
    model = linear_model.LinearRegression()
    model.fit(x_train, y_train)
    line = model.coef_ * x + model.intercept_

    #plotting hours against scores
    plt.scatter(x, y)
    plt.plot(x, line)
    plt.show()

    #testing our model
    y_pred = model.predict(x_test)
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))

    #testing for specific score
    predictedScore = model.predict([[9.25]])
    print("Predicted score for 9.25 hours is ", predictedScore)