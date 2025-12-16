import pandas as pd
import numpy as np


#-----EXERCISE 1-----#
#-----PART 1-----#
class GroupEstimate: 
    """
    This function implements a simple group-based estimator for
    categorical data

    Model groups observations by categorical vareiables and
    estimates a continuouse target value using either the mean
    or median of each group
    """
    def __init__(self, estimate: str = "mean"):
        """
        session state initializer

        function inputs:
        self, estimate : str (pass in the type of estimate to compute for each group. It must be either mean or median)
        function returns:
        n/a
        """
        if estimate not in {"mean", "median"}:
            raise ValueError("estimate must be either 'mean' or 'median'")

        self.estimate = estimate

        #store group-level estimates after fitting
        self.group_estimates = None

        #stores the column names used during fitting
        self.columns = None
    
    #-----PART 2-----#
    def fit(self, X: pd.DataFrame, y):
        """
        Fit model using categorical predictors and a continuous target

        function inputs: 
        self, X, and Y (x is pandas.DataFrame containing the categorical vars, Y is array that stores continuous values)
        function returns:
        return self (the fitted model instance)
        """

        #save column names only not X or Y
        self.columns = list(X.columns)

        #combine X and y into a single DataFrame
        df = X.copy()
        df["_y"] = y

        #group by all categorical columns
        grouped = df.groupby(self.columns)["_y"]

        #gompute group-level estimate
        if self.estimate == "mean":
            self.group_estimates = grouped.mean()
        else:
            self.group_estimates = grouped.median()

        return self

    #-----PART 3-----#
    def predict(self, X):
        """
        prdicts estimated values for new observations

        function inputs:
        X (this is new categorical obersvations with the same columns used during fitting)
        function outputs:
        returns results list which is a list of estimated values of corresponding to each row in X. Missing group combinatione return NaN
        """

        #convert input to dataFrame to ensure consistent access
        X_ = pd.DataFrame(X, columns=self.columns)

        results = []
        missing_count = 0

        #iterate over each observation
        for _, row in X_.iterrows():
            #create a tuple key representing the group
            key = tuple(row[col] for col in self.columns)

            #check if the groupe exists
            if key in self.group_estimates.index:
                results.append(self.group_estimates.loc[key])
            else:
                results.append(np.nan)
                missing_count += 1

        #print message if any groups were missing
        if missing_count > 0:
            print(f"{missing_count} observations had missing groups.")

        return np.array(results)
