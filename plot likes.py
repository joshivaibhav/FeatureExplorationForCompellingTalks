import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np


df = pd.read_csv("likes.csv")

plt.hist([i for i in range(100)], bins=[10,20])
plt.show()



