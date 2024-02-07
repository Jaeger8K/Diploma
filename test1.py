import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Fairlearn.datasets import fetch_adult

data = fetch_adult(as_frame=True)

X = pd.get_dummies(data.data)

y_true = (data.target == '>50K') * 1

sex = data.data['sex']

sex.value_counts()