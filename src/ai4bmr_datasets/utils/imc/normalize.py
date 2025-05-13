import pandas as pd

def normalize(data: pd.DataFrame, scale: str = 'minmax'):
    import numpy as np

    index = data.index
    columns = data.columns
    data = data.values

    censoring = 0.999
    cofactor = 1
    x = np.arcsinh(data / cofactor)
    thres = np.quantile(x, censoring, axis=0)

    data = np.minimum(data, thres, out=data)
    assert (data.max(axis=0) <= thres).all()

    if scale == "minmax":
        from sklearn.preprocessing import MinMaxScaler
        data = MinMaxScaler().fit_transform(x)
    elif scale == "standard":
        from sklearn.preprocessing import StandardScaler
        data = StandardScaler().fit_transform(x)
    elif scale is None:
        data = x
    else:
        raise NotImplementedError()

    return pd.DataFrame(data, index=index, columns=columns)