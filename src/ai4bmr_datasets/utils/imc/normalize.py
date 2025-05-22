import pandas as pd
import numpy as np

def normalize(data: pd.DataFrame, scale: str = 'minmax', exclude_zeros: bool = False):
    """
    Normalize a DataFrame of numerical values using arcsinh transformation
    followed by either min-max or standard scaling.

    Parameters:
    ----------
    data : pd.DataFrame
        Input DataFrame where each column is a feature to be normalized.
    scale : str, default='minmax'
        Scaling method to apply after arcsinh transformation. Must be one of:
        - 'minmax': scales features to the [0, 1] range.
        - 'standard': standardizes features by removing the mean and scaling to unit variance.
    exclude_zeros : bool, default=False
        If True, excludes zeros when computing the upper quantile clipping threshold.

    Returns:
    -------
    pd.DataFrame
        Normalized DataFrame with the same shape, index, and column names as the input.

    Raises:
    ------
    NotImplementedError
        If an unsupported scaling method is specified.
    """

    index = data.index
    columns = data.columns
    x = data.values

    censoring = 0.999
    cofactor = 1
    x = np.arcsinh(x / cofactor)

    if exclude_zeros:
        masked_x = np.where(x == 0, np.nan, x)
        thres = np.nanquantile(masked_x, censoring, axis=0)
    else:
        thres = np.nanquantile(x, censoring, axis=0)

    x = np.minimum(x, thres)
    assert (x.max(axis=0) <= thres).all()

    if scale == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        x = MinMaxScaler().fit_transform(x)
    elif scale == "standard":
        from sklearn.preprocessing import StandardScaler
        x = StandardScaler().fit_transform(x)
    else:
        raise NotImplementedError()

    return pd.DataFrame(x, index=index, columns=columns)
