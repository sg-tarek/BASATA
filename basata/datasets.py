def get_data(
    dataset="index",
    address="https://raw.githubusercontent.com/sg-tarek/BASATA/main/Datasets/",
):

    """
    This function loads sample datasets from git repository. List of available
    datasets can be checked using ``get_data('index')``.
    
    Example
    -------
    >>> from basata.datasets import get_data
    >>> all_datasets = get_data('index')
    >>> titanic = get_data('titanic')

    dataset: str, default = 'index'
        Index value of dataset.
    address: string, default = "https://raw.githubusercontent.com/sg-tarek/BASATA/main/Datasets/"
        Download url of dataset.
    Returns:
        pandas.DataFrame
    Warnings
    --------
    - Use of ``get_data`` requires internet connection.
    """

    import pandas as pd
    import os.path

    extension = ".csv"
    filename = str(dataset) + extension

    complete_address = address + filename

    if os.path.isfile(filename):
        data = pd.read_csv(filename)
    else:
        data = pd.read_csv(complete_address)

    return data