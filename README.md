# PyPerMANOVA
Implementation of permutational multivariate analysis of variance in Python
# Intro
Permutational multivariate analysis of variance (PerMANOVA) is a semi-parametric statistical method.
It is described as "a geometric partitioning of multivariate variation in the space of a chosen **dissimilarity measure**",
with p-values obtained using "appropriate distribution-free **permutation techniques**".

The script implements the simplest version - one-way PerMANOVA, and doesn't require much additional packaging.

The full tutorial is given in `Tutorial` Jupyter Notebook.

# How-to

To conduct the analysis, use the `permutational_analysis` unction from PyPerMANOVA.
It takes the following parameters:
* `data` : pandas DataFrame
		a numerical dataframe of N x M size
* `mapping` : iterator, dictionary, or pd.DataFrame
		will map columns (or indices) of data to group
* `column` : if mapping is pd.DataFrame, column in
		mapping to map columns (or indices) of data

It takes the following key arguments:
* `by` : string, optional
		What orientation is taken to produce a distance matrix.
		Can be either via column, or via row. Default is column.
* `norm` : string, optional
		If the data is normalized before constructing a distance matrix.
		Can be normalized with respect to "row"s or "column"s, or
		doesn't need to be normalized (None). Default is row.
* `metric` : string, optional
		Valid distance metric with which to construct a distance matrix.
		Default is "euclidean".
* `permutations` : int, optional
		Number of permutations used to calculate P value.
		Default is 999.

It returns the following results:
* `permanova_result` : pd.DataFrame, result of perMANOVA in the form of dataframe, with the following columns
  * "Pval" - the P value
  * "eta-sqr" - eta squared
  * "F" - F statistic
* `posthoc_result` : pd.DataFrame, result of post hoc perMANOVAs in the form of dataframe, with the following columns
  * "A","B - the test done between
  * "Pval" - the unadjusted P value
  * "bonf" - bonfferoni corrected P value
  * "eta-sqr" - identical to Pearson R square
  * "F" - F statistic
  * "t" - t statistic

## Notes:
The possible values for `metric` parameter are found on https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
And they include the following metrics : ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’, 
‘jaccard’, ‘jensenshannon’, ‘kulsinski’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’, ‘russellrao’, 
‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’

Use whatever metric you deem appropriate.
# References
 https://doi.org/10.1002/9781118445112.stat07841
