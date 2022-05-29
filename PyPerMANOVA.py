# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=C0325

"""
Conduct a permutational Multi-Variate Analysis of Variance and
its post-hoc testing.



To conduct the analysis, use the "permutational_analysis" function.
It takes the following parameters:
    data : pandas DataFrame
        a numerical dataframe of N x M size
    mapping : iterator, dictionary, or pd.DataFrame
        will map columns (or indices) of data to group
    column : if mapping is pd.DataFrame, column in
        mapping to map columns (or indices) of data

It takes the following key arguments:
    by : string, optional
        What orientation is taken to produce a distance matrix.
        Can be either via column, or via row. Default is column.
    norm : string, optional
        If the data is normalized before constructing a distance matrix.
        Can be normalized with respect to "row"s or "column"s, or
        doesn't need to be normalized (None). Default is row.
    metric : string, optional
        Valid distance metric with which to construct a distance matrix.
        Default is "euclidean".
    permutations : int, optional
        Number of permutations used to calculate P value.
        Default is 999.

It returns the following results:
    permanova_result : pd.DataFrame
        Result of perMANOVA in the form of dataframe
            Columns:
            "Pval" - the P value
            "eta-sqr" - identical to Pearson R square
            "F" - F statistic

    posthoc_result : pd.DataFrame
        Result of post hoc perMANOVAs in the form of dataframe
        Columns:
            "A","B - the test done between
            "Pval" - the unadjusted P value
            "bonf" - bonfferoni corrected P value
            "eta-sqr" - identical to Pearson R square
            "F" - F statistic
            "t" - t statistic
            
References
--------------
Anderson, Marti J. (2001). "A new method for non-parametric multivariate analysis of variance". Austral Ecology.

@author: ivanp
"""

import itertools
import numpy as np
import pandas as pd
from sklearn import preprocessing
from scipy.spatial.distance import pdist, squareform
from scipy.stats.stats import _unequal_var_ttest_denom

# %% FUNCTIONS FOR PREPROCESSING MATRICES
def normalize_matrix(matrix, by="column"):
    """
    Normalize the matrix

    Parameters
    ----------
    matrix :  pandas DataFrame
        Numerical dataframe of N x M shape that will
        be normalized.
    by : string, optional
        The axis to normalize along to.
        The default is "column".

    Raises
    ------
    ValueError
        When invalid ax is selected.

    Returns
    -------
    normalized_matrix : pandas DataFrame
        Numerical dataframe of N x M shape whose
        rows (or columns) have a mean of 0 and std of 1

    """
    if by in ["column", 1, "col", "c"]:
        result = preprocessing.scale(matrix)
        return(result)
    if by in ["row", "r", 0]:
        result = preprocessing.scale(matrix.T)
        result = result.T
        return(result)
    raise ValueError(f"Invalid value for {by}")


def convert_to_distance_matrix(matrix, metric="euclidean", norm="row", by="column"):
    """
    Parameters
    ----------
    matrix : pandas DataFrame
        Numerical dataframe of N x M shape that will be converted
        into a distance matrix.
    metric : string, optional
        The distance metric to use. The default is "euclidean".
        For more methods, see
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

    norm : string, optional
        Normalize the matrix before applying distance. The default is "row".
        Valid options are "row", "column" ("col"), None.
    by : string, optional
        Calculate the distance metric how. The default is "column".

    Returns
    -------
    distance_matrix : pandas DataFrame
        Numerical dataframe of N x N (or M x M) shape

    """
    input_matrix = matrix.copy()

    if norm in ["row", "r", 0,"rows"]:
        input_values = normalize_matrix(input_matrix, "row")
    elif norm in ["column", "col", 1, "c","columns"]:
        input_values = normalize_matrix(input_matrix, "column")
    else:
        input_values = input_matrix.values

    if by in ["column", "col", "c", 1]:
        input_matrix = input_matrix.T
        input_values = input_values.T

    vector = pdist(input_values, metric=metric)
    dis_matrix = squareform(vector)
    dis_df = pd.DataFrame(
        dis_matrix, columns=input_matrix.index, index=input_matrix.index)
    return(dis_df)


def preprocess_distance_matrix(distance_matrix, status_df, column):
    """
    DEPRECATED

    This function preprocesses distance matrix for subsequent downstream
    usage. For example, if its columns are :
        ["Sample1","Sample2","Sample3","SampleA","SampleB","SampleC"]
    And those values map to :
        ["Healthy", "Healthy", "Healthy", "Tumor","Tumor","Tumor"]
    This function will convert the columns and indices of distance matrix
    to the latter.


    Parameters
    ----------
    distance_matrix :  pandas DataFrame
        Numerical dataframe of N x N (or M x M) shape.
    status_df : pandas DataFrame
        Its indices are equal to indices (and columns) of distance_matrix
        Based on the value of "column",
        a mapping is done
    column : string
        a valid column of status_df that contains mapped values.

    Returns
    -------
    processed_distance_matrix : pandas DataFrame
        Numerical dataframe of N x N (or M x M) shape
        whose columns and indices contain many repeated values

    """
    processed = distance_matrix.copy()

    sample_grouping = dict(zip(status_df.index, status_df[column]))

    processed.columns = processed.columns.map(sample_grouping)
    processed.index = processed.index.map(sample_grouping)

    return(processed)

# %% FUNCTIONS FOR STATISTICAL TESTING AND THEIR SUPPLEMENTS
def _calculate_degrees_freedom(matrix,item):
    """
    Calculates degrees of freedom from a given
    distance matrix and a tuple of locators

    Formula used is from to scipy.stats.stats module

    Parameters
    ----------
    matrix : pd.DataFrame
        Symmetric distance matrix
    item : tuple / iterator
        Two items maximum

    Returns
    -------
    dof : float
    Calculated according to formula for unequal variances for t-test

    """
    first_matrix = matrix.loc[item[0]][item[0]]
    second_matrix = matrix.loc[item[1]][item[1]]

    var_x = sum_square_dist(first_matrix)
    var_y = sum_square_dist(second_matrix)
    n_x = len(first_matrix)
    n_y = len(second_matrix)

    dof = _unequal_var_ttest_denom(var_x,n_x,var_y,n_y)[0]
    return(dof)

def _calculate_cohend(F,dof):
    """
    Calculates Cohen D
    from F statistic and degrees of freedom

    The formula is 2 * t / sqrt(dof)
    and since t is sqrt of F,
    the formula is 2 * sqrt (F/dof)

    https://www.bwgriffin.com/gsu/courses/edur9131/content/Effect_Sizes_pdf5.pdf

    Parameters
    ----------
    F : float
        F statistic.
    dof : float
        degrees of freedom.

    Returns
    -------
    cohen-d

    """
    cohend = 2 * (F/dof)**0.5
    return(cohend)

def sum_square_dist(distance_matrix):
    """Returns sum of all values in dataframe over its length"""
    return(distance_matrix.values.sum()/len(distance_matrix))


def calculating_F_stat(grouping, valid_distance, effsize=False):
    """
    Calculates F stat from a given distance matrix
    Grouping is a list of groups equal in its length
    to size of matrix.

    Distance matrix can be calculated from any given matrix
    via "distance_matrix_transform"


    For more details, see
    Anderson, Marti J. “A new method for non-parametric multivariate analysis of variance.”
    Austral Ecology 26.1 (2001): 32-46.

    Parameters
    ----------
    grouping : list or any iterable
        list og groups.
    valid_distance : pandas DataFrame
        A symmetrical matrix containing
        distances. Its columns and indices are identical.
    effsize : Boolean,
        Whether to return eta squared as well

    Returns
    -------
    F : float
    F statistic (or pseudo-F statistic)

    """
    # group the distance matrix
    permuted_dismatrix = valid_distance.copy()
    permuted_dismatrix.columns = grouping
    permuted_dismatrix.index = grouping

    # grouping must contain redundant information
    # so to iterate through it efficiently
    # reduce groups
    all_groups = list(set(grouping))

    SST = sum_square_dist(permuted_dismatrix)

    SSW = 0
    # maybe refactor this - but shouldn't be
    # a problem when there's few groups
    for group in all_groups:
        sub_slice = permuted_dismatrix.loc[group][group]
        SSW = SSW + sum_square_dist(sub_slice)

    N = len(permuted_dismatrix)
    a = len(all_groups)

    SSA = SST - SSW
    F = (SSA/(a-1)) / (SSW/(N-a))

    # return both F and eff
    if effsize:
        etasq = SSA / SST
        return(F, etasq)

    return(F)


def perMANOVA(valid_distance, grouping=None, permutations=999):
    """
    The base permutational function
    to be used in conjuction with functools.partial.

    This is for conjuction with "calculating_t_stat" and "calculating_F_stat"
    since only those two functions share syntax similarities.

    The function takes a distance matrix and grouping (expected to be
    exact grouping) and calculates the statistic according to provided callable
    "func". Then it permutates the grouping and recalculates the statistic many
    times.
    It calculates the permutational P value (empirical P value?).


    Parameters
    ----------

    valid_distance : pandas DataFrame
        A symmetrical matrix containing
        distances. Its columns and indices are identical

    grouping : string
        An index (or column) to be applied to valid_distance
        If None, taken to be the columns of valid distance

    permutations : int, optional
        Number of permutations. The default is 999.

    Returns
    -------
    pvalue : float
        P Value of the provided grouping
    score : float
        Score of the provided grouping

    """
    # parse grouping
    if grouping is None:
        grouping = valid_distance.columns

    # lambda function for permutation, optimized for apply axis 1
    # def permutating_lambda(
    #    series, grouping): return np.random.permutation(grouping)

    # score every permutation based on t stat
    # the function takes as input groupings
    score = calculating_F_stat(
        grouping=grouping, valid_distance=valid_distance)

    # build a permutation dataframe
    perm_df = pd.DataFrame(index=range(permutations),
                           columns=range(len(grouping)))
    perm_df = perm_df.apply(lambda series, grouping: np.random.permutation(grouping), axis=1,
                            result_type="expand", grouping=grouping)

    # calculate outcomes
    outcomes = perm_df.apply(calculating_F_stat, axis=1,
                             valid_distance=valid_distance)
    # sns.histplot(outcomes)
    # calculate pvalue
    pvalue = ((outcomes >= score).sum() + 1) / (permutations + 1)
    return(pvalue, score)


def posthoc_perMANOVA(valid_distance, permutations=999):
    """
    Calculates post-hoc tests of permutational MANOVA.
    They are just ANOVA of all possible combinations of groups.
    The result is stored in a dataframe with columns:
        "A","B" - the test done between
        "Pval" - the unadjusted P value
        "bonf" - bonfferoni corrected P value
        "eta-sqr" - identical to Pearson R square
        "cohen-d" - Cohen's d
        "F" - F statistic
        "t" - t statistic
        "dof" - degrees of freedom


    Parameters
    ----------
    valid_distance : pandas DataFrame
        A symmetrical matrix containing
        distances. Its columns and indices are identical

    Returns
    -------
    result : pandas DataFrame

    """

    result = list()
    for item in itertools.combinations(valid_distance.columns.unique().tolist(), 2):
        sub_slice = valid_distance.loc[list(item)][list(item)]
        pvalue, F = perMANOVA(sub_slice, permutations=permutations)
        efsize = calculating_F_stat(sub_slice.columns, sub_slice, effsize=True)[1]
        dof = _calculate_degrees_freedom(sub_slice,item)
        cohend = _calculate_cohend(F,dof)
        result.append([item[0],item[1], pvalue, F, efsize,dof,cohend])
    result = pd.DataFrame(result, columns=["A","B", "Pval", "F", "eta-sqr","dof","cohen-d"])
    result["bonf"] = result["Pval"]*len(result)
    result["t"] = np.sqrt(result["F"])
    result = result[["A","B", "Pval", "bonf", "eta-sqr","cohen-d", "F", "t","dof"]]
    return(result)

# %%DEPRECATED CLASS - too slow
class _permanova_constructor():
    """
    This constructor is deprecated in
    favor of different functions
    which are faster (See calculations below).

    imeloop of constructor via %timeit:
        2.21 s ± 3.99 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

    timeloop of other function via %timeit:
        --------------------
        1.91 s ± 22 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
    """

    def __init__(self, matrix, **kwargs):

        self.matrix = matrix
        self.all_groups = pd.Series(matrix.columns.unique())
        self.a = len(self.all_groups)
        self.N = len(self.matrix)
        self.SST = sum_square_dist(self.matrix)
        self.permutations = kwargs.pop("permutations", 999)

    def calculate_SSW(self, permuted_matrix):
        """
        General function that calculates SSW
        from any matrix - including permuted matrix

        Parameters
        ----------
        permuted_matrix : pandas DataFrame
            Distance matrix (NxN) with identical
            columns and indices

        Returns
        -------
        The sum of squared differences between group means
        and overall sample mean (the Within-Group Sum of Squares)

        """

        SSW = self.all_groups.apply(lambda item, matrix: sum_square_dist(matrix.loc[item][item]),
                                    matrix=permuted_matrix).sum()
        return(SSW)

    @staticmethod
    def permute_matrix(grouping, original_matrix):
        """
        Takes a matrix of NxN
        and simply relabels columns and indices

        Parameters
        ----------
        grouping : iterator that can be accepted
        as a Pandas Index
            DESCRIPTION.
        original_matrix : pandas DataFrame
            Distance matrix (NxN) with identical
            columns and indices

        Returns
        -------
        permuted_matrix : pandas DataFrame
            Its columns and indices are equal to grouping now

        """

        permuted_matrix = original_matrix.copy()
        permuted_matrix.index = grouping
        permuted_matrix.columns = grouping
        return(permuted_matrix)

    @property
    def p_value(self):
        """
        P value for PermANOVA
        """
        return((self.F_dist >= self.F).sum() + 1) / (self.permutations + 1)

    @property
    def eta_sq(self):
        """
        Eta squared - defined as ratio of SSA and SST,
        also considered a sample size
        """
        return(self.SSA/self.SST)

    @property
    def SSA(self):
        """
        Sum of squared differences
        between group means and the overall sample mean

        """
        return(self.SST-self.SSW)

    @property
    def SSW(self):
        """
        The sum of squared differences between group means
        and overall sample mean (the Within-Group Sum of Squares)
        """
        return(self.calculate_SSW(self.matrix))

    @property
    def F(self):
        """
        F statistic corrected for the
        degrees of freedom
        """
        return(self.SSA)/(self.SSW) * (self.N-self.a)/(self.a-1)

    @property
    def F_dist(self):
        """
        Distribution of F statistic
        via permutation of a given matrix
        """
        # building permutational dataframe
        grouping = self.matrix.columns
        permutations = self.permutations

        # building permutational dataframe
        perm_df = pd.DataFrame(index=range(permutations),
                               columns=range(len(grouping)))
        perm_df = perm_df.apply(lambda row, grouping: np.random.permutation(grouping),
                                axis=1, result_type="expand", grouping=grouping)
        # building series of permuted matrix
        # and quickly calculating SSW for these matrices
        SSW = perm_df.apply(self.permute_matrix, axis=1,
                            original_matrix=self.matrix,
                            ).apply(self.calculate_SSW)

        # calculate F values according to equation
        F_vals = (self.SST-SSW)/(SSW) * (self.N-self.a)/(self.a-1)
        return(F_vals)

def perMANOVA_via_constructor(matrix, permutations=999):
    """DEPRECATED"""
    pc = _permanova_constructor(matrix, permutations=permutations)
    return(pc)

# %% WORKFLOW FUNCTION
def permutational_analysis(data, mapping, column=None, **kwargs):
    """
    Main level function that incorporates every step needed to
    get permutational analyses from a numerical matrix of uneven size.

    The steps include:
        1) (optional) normalizing data matrix:
            using "norm" kwarg to specify "row", "column", or "none"
        2) (optional) selecting which axis to construct a distance matrix
            from using the "by" kwarg"
        3) Mapping columns (by = "column") or rows (by = "row") to
           a group using "mapping" parameter
        4) Constructing a distance matrix based on the
            provided "metric" kwarg
        5) Executing Permutational Analysis (perMANOVA and posthoc_perMANOVA)
            and returning results in the form of a DataFrame

    Parameters
    ----------
    data : pandas DataFrame
        A numerical dataframe of N x M size.
    mapping : iterator (list,pd.Series), pd.DataFrame, or None
        Will map every "sample" in indices or columns of data
        to a "group"

        When it's iterator - mapping should be in the same
        order as the columns or indices of data

        When it's None, assumption is made that data is already
        grouped

        When it's pd.DataFrame, specify column that will be
        contains groups with "column" parameter, and make sure
        that indices of mapping are the same as columns or indices of data

    column : string, optional
        Value of column when mapping is of pd.DataFrame type. The default is None.
    **kwargs :
        by : string, optional
            What orientation is taken to produce a distance matrix.
            Can be either via column, or via row. Default is column.
        norm : string, optional
            If the data is normalized before constructing a distance matrix.
            Can be normalized with respect to "row"s or "column"s, or
            doesn't need to be normalized (None). Default is row.
        metric : string, optional
            Valid distance metric with which to construct a distance matrix.
            Default is "euclidean".
        permutations : int, optional
            Number of permutations used to calculate P value.
            Default is 999.
    Raises
    ------
    ValueError
        When invalid value for "by" key argument is passed.

    AttributeError
        -When there is a mismatch of length between provided mapping
         and columns (or indices) of data.
        -When mapping is of pd.DataFrame type and no
         column is provided, or its indices do not match
         columns (or indices) of data
        -When mapping contains only one value (no ANOVA),
         or when there are only unique values


    Returns
    -------
    permanova_result : pd.DataFrame
        Result of perMANOVA in the form of dataframe
            Columns:
            "Pval" - the P value
            "eta-sqr" - identical to Pearson R square
            "F" - F statistic

    posthoc_result : pd.DataFrame
        Result of post hoc perMANOVAs in the form of dataframe
        Columns:
            "Source" - the test done between
            "Pval" - the unadjusted P value
            "bonf" - bonfferoni corrected P value
            "eta-sqr" - identical to Pearson R square
            "F" - F statistic
            "t" - t statistic

    """
    input_matrix = data.copy()

    # process kwargs
    by = kwargs.pop("by", "column")
    norm = kwargs.pop("norm", "row")
    metric = kwargs.pop("metric", "euclidean")
    permutations = kwargs.pop("permutations", 999)

    # check what orientation of matrix will be taken
    if by in ["column", 1, "col", "c","columns"]:
        by = "column"
        samples = data.columns.tolist()
    elif by in ["row", "r", 0,"rows"]:
        by = "row"
        samples = data.index.tolist()
    else:
        raise ValueError("Invalid value for 'by' - use either row or column")
    #set checker for valid mapping
    check_for_valid_groups = True

    # check for proper mapping
    if isinstance(mapping, (list, pd.Series, pd.Index)):
        if len(mapping) != len(samples):
            raise AttributeError(
                f"Mismatch of length between provided mapping and {by}s of data")
        sample_group_mapping = dict(zip(samples, mapping))
    elif isinstance(mapping, dict):
        if len(mapping) != len(samples):
            raise AttributeError(
                f"Mismatch of length between provided dictionary and {by}s of data")
        sample_group_mapping = mapping
    elif isinstance(mapping, pd.DataFrame):
        if len(mapping) != len(samples):
            raise AttributeError(
                f"Mismatch of length between provided dataframe and {by}s of data")
        if column is None or column not in mapping.columns:
            raise AttributeError(
                "If mapping is a DataFrame, a valid column name must be provided")
        # check if indices of provided dataframe are the same as samples
        if set(mapping.index) != set(samples):
            raise AttributeError(
                "If mapping is a DataFrame, its indices must be equal to samples,('.set_index')")
        sample_group_mapping = dict(zip(mapping.index, mapping[column]))
    elif not mapping:
        # mapping is None, data will be passed as they are
        sample_group_mapping = dict(zip(samples, samples))
        check_for_valid_groups = False
    else:
        raise ValueError("Invalid type of mapping")

    # Check if mapping gives values that have redundant values
    mapped_values_length = len(set(sample_group_mapping.values()))
    if mapped_values_length == 1:
        raise AttributeError("The mapping contains one group")
    if mapped_values_length == len(sample_group_mapping.values()) and check_for_valid_groups:
        raise AttributeError("The mapping contains no duplicates")

    # map the matrix to get proper groups
    if by == "column":
        input_matrix.columns = input_matrix.columns.map(sample_group_mapping)
    if by == "row":
        input_matrix.index = input_matrix.index.map(sample_group_mapping)

    # get the distance matrix
    distance_matrix = convert_to_distance_matrix(
        input_matrix, metric=metric, norm=norm, by=by)

    # calculate PerMANOVA results
    pvalue, F = perMANOVA(distance_matrix, permutations=permutations)
    etasqr = calculating_F_stat(
        distance_matrix.columns, distance_matrix, effsize=True)[1]
    permanova_result = pd.DataFrame([pvalue, etasqr, F], index=[
                                    "Pval", "eta-sqr", "F"]).T

    # calculate Posthoc results
    posthoc_result = posthoc_perMANOVA(distance_matrix, permutations)

    return(permanova_result, posthoc_result)
