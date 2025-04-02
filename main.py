import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Recourses
# https://www.youtube.com/watch?v=Liv6eeb1VfE

# Path to dataset
path = "data/WineQT.csv"

# Reading Dataset
df = pd.read_csv(path)

df.info()

