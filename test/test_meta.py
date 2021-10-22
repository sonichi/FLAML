from sklearn.datasets import fetch_california_housing, load_wine
from flaml.default import meta_feature, suggest_config
from flaml.data import DataTransformer


def test_meta_feature():
    X, y = load_wine(return_X_y=True, as_frame=True)
    print(meta_feature("classification", X, y))
    dt = DataTransformer()
    X_train, y_train = dt.fit_transform(X, y, "classification")
    print(meta_feature("classification", X, y))

    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    print(meta_feature("regressions", X, y))
    X_train, y_train = dt.fit_transform(X, y, "regressions")
    print(meta_feature("regressions", X, y))


def test_suggest_config():
    portfolio = {
        "version": "default",
        "preprocessing": {
            "center": [1000, 10, 2, 0.5],
            "scale": [1000, 10, 1, 1],
        },
        "neighbors": [
            {
                "features": [0.1, 0.2, 0.8, 0.3],
                "choice": 0,
            },
            {
                "features": [0.5, 1.0, 0.0, -2],
                "choice": 1,
            },
        ],
        "portfolio": [
            {
                "class": "lgbm",
                "hyperparameters": {},
            },
            {
                "class": "xgboost",
                "hyperparameters": {},
            },
        ],
    }
    X, y = load_wine(return_X_y=True, as_frame=True)
    print(suggest_config("classification", X, y, portfolio))
    X, y = fetch_california_housing(return_X_y=True, as_frame=True)
    print(suggest_config("regression", X, y, portfolio))
    

if __name__ == "__main__":
    test_suggest_config()
    test_meta_feature()
