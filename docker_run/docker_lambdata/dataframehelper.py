"""Helper utility functions for Pandas DataFrames."""

import pandas as pd
import scipy.stats as scs
import numpy as np
from typing import Union


class DataFrameHelper:
    """Class for pandas dataframe that contains several helper functions.

    Args:
        data_frame: pandas dataframe to use with helper utilities.
        out_path: path in container for output code to go.

    Attributes:
        data_frame: pandas dataframe to use with helper utilities.
        X_train: pandas dataframe of independent features for training.
        X_val: pandas dataframe of independent features for validation.
        X_test: pandas dataframe of independent features for testing.
        y_train: pandas dataframe of dependent features for training.
        y_val: pandas dataframe of dependent features for validation.
        y_test: pandas dataframe of dependent features for testing.
    """

    def __init__(self, data_frame: pd.DataFrame, out_path: str):
        self.data_frame = data_frame
        self.output = open(out_path, "w")

        self.X_train = None
        self.X_val = None
        self.X_test = None

        self.y_train = None
        self.y_val = None
        self.y_test = None

    def invalid_data_values(self, invalid_dict: dict) -> pd.DataFrame:
        """Replaces invalid values in a dictionary from a user-defined
        dictionary of column names corresponding to a list of invalid values.

        Args:
            invalid_dict: dictionary of column names to list of invalid values
        Returns:
            pandas dataframe
        """
        for key in invalid_dict:
            for val in invalid_dict[key]:
                self.data_frame.loc[lambda data_frame:
                                    self.data_frame[key] == val, key] = np.nan
        return self.data_frame

    def check_data(self) -> None:
        """Prints NaN values for a Pandas DataFrame to output file.

        Args:
            None
        Returns:
            None
        """
        self.output.write("NaN Values\n")
        format_string = "{:40s}: {: 10d} / {: 10d}\n"
        for col_name in self.data_frame:
            self.output.write(format_string.format(col_name,
                              self.data_frame[col_name].isna().sum(),
                              self.data_frame[col_name].count()))
        self.output.write("\n")

    def data_split(self, ycols: list, train_ratio: float = 0.5,
                   val_ratio: float = 0.25) -> (pd.DataFrame, pd.DataFrame,
                                                pd.DataFrame, pd.DataFrame,
                                                pd.DataFrame, pd.DataFrame):
        """Splits data into depednent and independent training, validation, and
        testing datasets.

        Args:
            ycols: list of column values or column value for df which represent
                the depedent data
            train_ratio: fraction of df for training (default 0.5)
            val_ratio: fraction of df for validation (default 0.25)
        Returns:
            (X_train, X_val, X_test,
             y_train, y_val, y_test)
                *** pandas dataframes/series
        """
        df_train = self.data_frame.sample(frac=train_ratio)
        data_frame_n = self.data_frame.drop(df_train.index)
        df_val = data_frame_n.sample(frac=val_ratio)
        df_test = data_frame_n.drop(df_val.index)

        xcols = [c for c in self.data_frame if c not in ycols]

        self.X_train = df_train[xcols]
        self.X_val = df_val[xcols]
        self.X_test = df_test[xcols]
        self.y_train = df_train[ycols]
        self.y_val = df_val[ycols]
        self.y_test = df_test[ycols]

        return (self.X_train, self.X_val, self.X_test,
                self.y_train, self.y_val, self.y_test)

    def generate_data(self, nrows: int, nsamples: int) -> pd.DataFrame:
        """Returns a dataframe with randomly copied rows copied a number of
        times.

        Args:
            nrows: number of rows to sample
            nsamples: number of times to sample dataframe
        Returns:
            pandas dataframe
        """
        for _ in range(nsamples):
            self.data_frame = pd.concat([self.data_frame,
                                         self.data_frame.sample(n=nrows)],
                                        axis=0)
        return self.data_frame

    def chi_viz(self, col1, col2) -> None:
        """Prints contingency table and chi-squared statistics for it to output
        file.

        Args:
            col1: column name in dataframe
            col2: column name in dataframe
        Returns:
            None
        """
        cross = pd.crosstab(self.data_frame[col1], self.data_frame[col2])
        self.output.write("{} : {}\n".format(col1, col2))
        self.output.write(cross.to_string())
        self.output.write("\n")
        chi_val, p_val, dof, expected = scs.chi2_contingency(cross)
        self.output.write("chi2 statistic: {}\n".format(chi_val))
        self.output.write("p-value statistic: {}\n".format(p_val))
        self.output.write("degrees of freedom: {}\n".format(dof))
        self.output.write("expected frequencies {}\n\n".format(expected))
