# Data-Toolkit
Data Visualization and Analysis with Python

This repository contains various Python scripts demonstrating data visualization and analysis techniques using libraries such as NumPy, Pandas, Matplotlib, Seaborn, and Plotly.

Features

Creating and manipulating Pandas DataFrames

Filtering data based on conditions

Generating various plots using Matplotlib, Seaborn, and Plotly

Performing matrix operations using NumPy

Loading and processing CSV files using Pandas

Installation

Ensure you have Python installed, then install the required dependencies:

pip install numpy pandas matplotlib seaborn plotly

Usage

Clone this repository:

git clone https://github.com/your-username/repository-name.git
cd repository-name

Run the desired script:

python script_name.py

Code Examples

1. Creating a Pandas DataFrame and Filtering Rows

import pandas as pd

data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 40, 22]}

df = pd.DataFrame(data)
filtered_df = df[df['Age'] > 30]
print(filtered_df)

2. Performing Matrix Multiplication using NumPy

import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
result = A @ B  # or np.dot(A, B)
print(result)

3. Generating a Histogram using Seaborn

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = np.random.randn(500)
sns.histplot(data, bins=30, kde=True, color='blue')
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of Random Data")
plt.show()

4. Creating a 3D Scatter Plot using Plotly

import plotly.express as px
import pandas as pd

data = {
    'X': [1, 2, 3, 4, 5],
    'Y': [10, 20, 30, 40, 50],
    'Z': [5, 15, 25, 35, 45],
    'Category': ['A', 'B', 'A', 'B', 'A']
}

df = pd.DataFrame(data)
fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Category', title="3D Scatter Plot")
fig.show()

Example Outputs

The Pandas filtering script prints a filtered DataFrame.

The NumPy script outputs the matrix multiplication result.

The Seaborn script displays a histogram of a random dataset.

The Plotly script generates an interactive 3D scatter plot.
