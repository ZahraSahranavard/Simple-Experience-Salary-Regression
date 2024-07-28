import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

years_of_experience = np.array([0, 1, 2, 3, 4, 6, 8, 10]).reshape(-1, 1)
salaries = np.array([30000, 40000, 55000, 60000, 70000, 80000, 85000, 87000])

model = DecisionTreeRegressor(max_depth=4)
model.fit(years_of_experience, salaries)

y_predict = model.predict(years_of_experience)

plt.plot(years_of_experience, salaries, color="red", marker='*')
plt.plot(years_of_experience, y_predict)
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()