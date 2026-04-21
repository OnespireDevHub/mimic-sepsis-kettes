import numpy as np
import cloudpickle
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import (
    StandardScaler,
    PolynomialFeatures,
    FunctionTransformer,
)
from sklearn.pipeline import Pipeline


def get_model():
    def medical_feature_engineering(X):
        # Per FEATURE_COLUMNS in dataset_loader.py:
        # HR=0, SBP=3, MAP=4, DBP=5, BUN=15, Creatinine=19
        # Always keep these transformations under the get_model() function!

        hr = X[:, [0]]
        sbp = X[:, [3]]
        map_ = X[:, [4]]
        dbp = X[:, [5]]
        bun = X[:, [15]]
        creat = X[:, [19]]
        lactate = X[:, [VALAMILYEN_INDEX]]

        shock_index = hr / (sbp + 1e-6)
        bun_creat_ratio = bun / (creat + 1e-6)
        pulse_pressure = sbp - dbp
        hr_map_ratio = hr / (map_ + 1e-6)
        map_sbp_ratio = map_ / (sbp + 1e-6)
        lactate_hr = lactate * hr

        return np.hstack(
            [X, shock_index, bun_creat_ratio, pulse_pressure, hr_map_ratio, map_sbp_ratio, lactate_hr]
        )

    model = Pipeline(
        [
            ("engineering", FunctionTransformer(medical_feature_engineering)),
            # ("poly", PolynomialFeatures(degree=2, interaction_only=True)),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(warm_start=True, max_iter=10, class_weight={0:1, 1:3})),
        ]
    )

    # Pre-fit with dummy data to initialize pipeline shapes (40 input features)
    model.fit(np.zeros((5, 40)), np.array([0, 1, 0, 1, 0]))
    return model


def get_model_parameters(model):
    return [model.named_steps["clf"].coef_, model.named_steps["clf"].intercept_]


def set_model_parameters(model, parameters):
    model.named_steps["clf"].coef_ = parameters[0]
    model.named_steps["clf"].intercept_ = parameters[1]


def save_model(model, path="final_model.pkl"):
    """Serialize the full pipeline to disk (cloudpickle preserves custom transforms)."""
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)


def load_model(path="final_model.pkl"):
    """Load a previously saved pipeline from disk."""
    with open(path, "rb") as f:
        return cloudpickle.load(f)