import pandas as pd
import numpy as np

if __name__=='__main__':
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']

    raw_dataset = pd.read_csv(url, names=column_names,
                            na_values='?', comment='\t',
                            sep=' ', skipinitialspace=True)

    print("Mean: ",raw_dataset.mean(),"\n")
    
    filtered = raw_dataset[raw_dataset['Cylinders'] == 3]
    print("Filtered: ",filtered)