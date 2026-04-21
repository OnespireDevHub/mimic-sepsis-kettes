import cloudpickle
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


# -----------------------------------------------------------------------------
# Tunables
# -----------------------------------------------------------------------------
# The competition metric penalizes FN 5x more than FP, so the model should lean
# toward higher recall. Two simple, strong levers are:
#   1) positive class weighting during training
#   2) a lower decision threshold during predict()
DECISION_THRESHOLD = 0.25
POSITIVE_CLASS_WEIGHT = 4.0
REGULARIZATION_C = 0.40
MAX_ITER = 300
N_RAW_FEATURES = 40
EPS = 1e-6


# -----------------------------------------------------------------------------
# Feature indices from DATA_DICTIONARY.md / dataset_loader.py
# -----------------------------------------------------------------------------
HR = 0
O2SAT = 1
TEMP = 2
SBP = 3
MAP = 4
DBP = 5
RESP = 6
ETCO2 = 7
BASE_EXCESS = 8
HCO3 = 9
FIO2 = 10
PH = 11
PACO2 = 12
SAO2 = 13
AST = 14
BUN = 15
ALKALINEPHOS = 16
CALCIUM = 17
CHLORIDE = 18
CREATININE = 19
BILIRUBIN_DIRECT = 20
GLUCOSE = 21
LACTATE = 22
MAGNESIUM = 23
PHOSPHATE = 24
POTASSIUM = 25
BILIRUBIN_TOTAL = 26
TROPONINI = 27
HCT = 28
HGB = 29
PTT = 30
WBC = 31
FIBRINOGEN = 32
PLATELETS = 33
AGE = 34
GENDER = 35
UNIT1 = 36
UNIT2 = 37
HOSP_ADM_TIME = 38
ICULOS = 39


def _col(X: np.ndarray, idx: int) -> np.ndarray:
    return X[:, [idx]]


def _safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    return num / (den + EPS)


def _signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def medical_feature_engineering(X: np.ndarray) -> np.ndarray:
    """
    Stateless feature engineering only.

    Why stateless? In this starter kit, only clf.coef_ and clf.intercept_ are
    aggregated/saved by the server. That makes deterministic transforms safer
    than client-fitted preprocessors like StandardScaler/PCA.
    """
    X = np.asarray(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    hr = _col(X, HR)
    o2sat = _col(X, O2SAT)
    temp = _col(X, TEMP)
    sbp = _col(X, SBP)
    map_ = _col(X, MAP)
    dbp = _col(X, DBP)
    resp = _col(X, RESP)
    etco2 = _col(X, ETCO2)
    base_excess = _col(X, BASE_EXCESS)
    hco3 = _col(X, HCO3)
    fio2 = _col(X, FIO2)
    ph = _col(X, PH)
    ast = _col(X, AST)
    bun = _col(X, BUN)
    creat = _col(X, CREATININE)
    lactate = _col(X, LACTATE)
    bilirubin_total = _col(X, BILIRUBIN_TOTAL)
    ptt = _col(X, PTT)
    wbc = _col(X, WBC)
    platelets = _col(X, PLATELETS)
    age = _col(X, AGE)
    hosp_adm_time = _col(X, HOSP_ADM_TIME)
    iculos = _col(X, ICULOS)

    # Compress raw scale differences without learning any client-specific state.
    base_features = _signed_log1p(X)

    # Clinically motivated derived features.
    shock_index = _safe_div(hr, sbp)
    bun_creat_ratio = _safe_div(bun, creat)
    pulse_pressure = sbp - dbp
    hr_map_ratio = _safe_div(hr, map_)
    resp_o2_ratio = _safe_div(resp, np.maximum(o2sat, 1.0))
    resp_etco2_ratio = _safe_div(resp, np.maximum(etco2, 1.0))

    map_deficit = np.maximum(65.0 - map_, 0.0)
    sbp_deficit = np.maximum(100.0 - sbp, 0.0)
    hypoxemia = np.maximum(92.0 - o2sat, 0.0)
    temp_dev = np.abs(temp - 37.0)
    fever_or_hypothermia = ((temp >= 38.0) | (temp <= 36.0)).astype(np.float64)

    acid_base_stress = (
        np.maximum(7.40 - ph, 0.0)
        + np.maximum(22.0 - hco3, 0.0)
        + np.maximum(-base_excess, 0.0)
    )
    lactate_log = np.log1p(np.clip(lactate, 0.0, None))
    renal_stress = np.log1p(np.clip(bun, 0.0, None)) + np.log1p(np.clip(creat, 0.0, None))
    hepatic_stress = np.log1p(np.clip(ast, 0.0, None)) + np.log1p(
        np.clip(bilirubin_total, 0.0, None)
    )
    inflam_load = np.abs(wbc - 11.0)
    coag_risk = np.log1p(np.clip(ptt, 0.0, None)) - np.log1p(np.clip(platelets, 0.0, None))
    oxygen_support_gap = np.maximum(fio2 - o2sat, 0.0)

    qsofa_proxy = ((sbp <= 100.0).astype(np.float64) + (resp >= 22.0).astype(np.float64))
    age_shock = (age / 100.0) * shock_index
    iculos_log = np.log1p(np.clip(iculos, 0.0, None))
    hosp_adm_time_log = _signed_log1p(hosp_adm_time)

    # Targeted non-linear interactions.
    septic_stress = shock_index * lactate_log
    renal_hypotension = map_deficit * renal_stress
    inflammation_temp = inflam_load * temp_dev
    oxygen_work_of_breathing = hypoxemia * resp_o2_ratio
    qsofa_age = qsofa_proxy * (age / 100.0)

    continuous_engineered = np.hstack(
        [
            shock_index,
            bun_creat_ratio,
            pulse_pressure,
            hr_map_ratio,
            resp_o2_ratio,
            resp_etco2_ratio,
            map_deficit,
            sbp_deficit,
            hypoxemia,
            temp_dev,
            acid_base_stress,
            lactate_log,
            renal_stress,
            hepatic_stress,
            inflam_load,
            coag_risk,
            oxygen_support_gap,
            age_shock,
            iculos_log,
            hosp_adm_time_log,
            septic_stress,
            renal_hypotension,
            inflammation_temp,
            oxygen_work_of_breathing,
            qsofa_age,
        ]
    )

    flags = np.hstack([fever_or_hypothermia, qsofa_proxy])
    engineered = np.hstack([_signed_log1p(continuous_engineered), flags])

    return np.hstack([base_features, engineered])


class ThresholdedLogisticRegression(BaseEstimator, ClassifierMixin):
    """LogisticRegression with a custom prediction threshold.

    This is useful because the competition objective is asymmetric:
    Cost = FP + 5 * FN
    """

    def __init__(
        self,
        threshold: float = DECISION_THRESHOLD,
        positive_class_weight: float = POSITIVE_CLASS_WEIGHT,
        C: float = REGULARIZATION_C,
        max_iter: int = MAX_ITER,
    ):
        self.threshold = threshold
        self.positive_class_weight = positive_class_weight
        self.C = C
        self.max_iter = max_iter
        self._clf = None

    def _build_clf(self) -> LogisticRegression:
        return LogisticRegression(
            solver="lbfgs",
            C=self.C,
            max_iter=self.max_iter,
            warm_start=True,
            class_weight={0: 1.0, 1: self.positive_class_weight},
        )

    def _ensure_clf(self) -> None:
        if self._clf is None:
            self._clf = self._build_clf()

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._ensure_clf()
        self._clf.set_params(
            C=self.C,
            max_iter=self.max_iter,
            class_weight={0: 1.0, 1: self.positive_class_weight},
        )
        self._clf.fit(X, y)
        self.classes_ = self._clf.classes_
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self._ensure_clf()
        return self._clf.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._ensure_clf()
        return self._clf.decision_function(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(np.int64)

    @property
    def coef_(self) -> np.ndarray:
        self._ensure_clf()
        return self._clf.coef_

    @coef_.setter
    def coef_(self, value: np.ndarray) -> None:
        self._ensure_clf()
        self._clf.coef_ = value

    @property
    def intercept_(self) -> np.ndarray:
        self._ensure_clf()
        return self._clf.intercept_

    @intercept_.setter
    def intercept_(self, value: np.ndarray) -> None:
        self._ensure_clf()
        self._clf.intercept_ = value


def get_model() -> Pipeline:
    model = Pipeline(
        steps=[
            (
                "engineering",
                FunctionTransformer(
                    medical_feature_engineering,
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            ),
            (
                "clf",
                ThresholdedLogisticRegression(
                    threshold=DECISION_THRESHOLD,
                    positive_class_weight=POSITIVE_CLASS_WEIGHT,
                    C=REGULARIZATION_C,
                    max_iter=MAX_ITER,
                ),
            ),
        ]
    )

    # Pre-fit once so coef_/intercept_ exist before FL starts.
    dummy_X = np.zeros((8, N_RAW_FEATURES), dtype=np.float64)
    dummy_X[:, HR] = np.array([70, 72, 75, 78, 95, 100, 105, 110], dtype=np.float64)
    dummy_X[:, SBP] = np.array([120, 118, 115, 112, 100, 98, 95, 90], dtype=np.float64)
    dummy_X[:, MAP] = np.array([85, 84, 82, 80, 72, 70, 68, 65], dtype=np.float64)
    dummy_X[:, TEMP] = np.array([36.8, 36.9, 37.0, 37.1, 37.8, 38.1, 38.5, 39.0], dtype=np.float64)
    dummy_X[:, RESP] = np.array([16, 16, 17, 18, 22, 24, 26, 28], dtype=np.float64)
    dummy_X[:, O2SAT] = np.array([98, 98, 97, 97, 95, 94, 92, 90], dtype=np.float64)
    dummy_X[:, LACTATE] = np.array([0.8, 0.9, 1.0, 1.1, 1.8, 2.2, 2.8, 3.5], dtype=np.float64)
    dummy_y = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    model.fit(dummy_X, dummy_y)
    return model


def get_model_parameters(model: Pipeline):
    clf = model.named_steps["clf"]
    return [clf.coef_, clf.intercept_]


def set_model_parameters(model: Pipeline, parameters):
    clf = model.named_steps["clf"]
    clf.coef_ = parameters[0]
    clf.intercept_ = parameters[1]


def save_model(model, path: str = "final_model.pkl"):
    with open(path, "wb") as f:
        cloudpickle.dump(model, f)


def load_model(path: str = "final_model.pkl"):
    with open(path, "rb") as f:
        return cloudpickle.load(f)
