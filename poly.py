import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# trainig set
x_train = [[129], [135], [147], [160], [171], [184], [198], [223], [240], [293]] # home sizes square meters 
y_train = [[1182], [1172], [1264], [1493], [1571], [1711], [1804], [1840], [1956], [1954]] # KW hrs/month

# testing set
x_test = [[118], [137], [150], [161], [173], [185], [199], [226], [245], [300]]
y_test = [[1000], [1198], [1260], [1458], [1555], [1750], [1830], [1880], [1980], [1940]]

# linear regression model
regressor = LinearRegression()
regressor.fit(x_train, y_train)
xx_test = np.linspace(110, 310, 1000)
y_pred = regressor.predict(xx_test.reshape(xx_test.shape[0], 1))

# polynomial regression model
polynomial_features= PolynomialFeatures(degree=3)
x_poly_train = polynomial_features.fit_transform(x_train)
x_poly_test = polynomial_features.fit_transform(xx_test.reshape(xx_test.shape[0], 1))

model = LinearRegression()
model.fit(x_poly_train, y_train)
y_poly_pred = model.predict(x_poly_test)

# plotting sets and models
plt.scatter(x_train, y_train)
plt.title('Electricity consumption in kilowatt-hours per month from ten houses')
plt.xlabel('Home sizes (m2)')
plt.ylabel('KW hrs/month')
plt.grid(True)
plt.plot(xx_test, y_pred, c = 'black') 
plt.plot(xx_test, y_poly_pred, c = 'red', linestyle = '--')
plt.show()
