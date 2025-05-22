import pandas as pd

def normalize(data: pd.DataFrame, scale: str = 'minmax', exclude_zeros: bool = False):
    import numpy as np

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
        x = MinMaxScaler().fit_transform(x)
    elif scale == "standard":
        x = StandardScaler().fit_transform(x)
    else:
        raise NotImplementedError()

    return pd.DataFrame(x, index=index, columns=columns)