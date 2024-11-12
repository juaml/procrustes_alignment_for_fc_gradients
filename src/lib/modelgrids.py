"""Define the model grids used in this project."""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

C_VALS = np.geomspace(1e-4, 1e4, 50)
LAMBDS = np.geomspace(0.001, 10000, 50)

C_VALS_RBF = np.geomspace(1e-4, 1e4, 10)
GAMMA_VALS = np.geomspace(1e-9, 1e4, 10)

model_dict_classification = {
    "rf": ("rf", {"max_depth": [5, 10, 20, None]}),
    "ridge": (RidgeClassifier(), {"alpha": LAMBDS}),
    "linear_svm": (
        "svm",
        {"kernel": ["linear"], "C": C_VALS, "probability": [True]},
    ),
    "rbf_svm": (
        "svm",
        {
            "kernel": ["rbf"],
            "C": C_VALS_RBF,
            "probability": [True],
            "gamma": GAMMA_VALS,
        },
    ),
    "knn": (
        KNeighborsClassifier(algorithm="auto"),
        {
            "n_neighbors": [x for x in range(5, 50, 2)],
            "weights": ["uniform", "distance"],
            "p": [x for x in range(1, 5)],
        },
    ),
}

model_dict_regression = {
    "rf": ("rf", {"max_depth": [5, 10, 20, None]}),
    "ridge": (Ridge(), {"alpha": LAMBDS}),
    "linear_svm": (
        "svm",
        {"kernel": ["linear"], "C": C_VALS},
    ),
    "rbf_svm": (
        "svm",
        {
            "kernel": ["rbf"],
            "C": C_VALS_RBF,
            "gamma": GAMMA_VALS,
        },
    ),
    "kernelridge": (
        KernelRidge(),
        {"alpha": np.geomspace(0.00001, 1000000, 40)},
    ),
}


def get_model_grid(model_name, problem_type):
    """Get the model and the corresponding grid.

    Parameters
    ----------
    model_name : string
        For which model to retrieve gridsearch parameters.
    problem_type : String
        "regression" or "classification".
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    model : String
        model to pass to run_cross_validation().
    param_dict : dict
        params to pass over to grid search.
    """
    if problem_type == "classification":
        return model_dict_classification[model_name]
    else:
        return model_dict_regression[model_name]
