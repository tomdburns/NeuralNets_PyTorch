"""
Reads in the full dataset and assigns a pass (1) and fail (0) flag
for all of the process points tested
"""


import pandas as pd


infile = 'combined_ML_Names.csv'


def main():
    """main"""
    data = pd.read_csv(infile)


if __name__ in '__main__':
    main()
