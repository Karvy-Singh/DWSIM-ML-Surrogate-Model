#!/usr/bin/env python3

"""Keep the model's purity predictions within the material balance."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted


FEED_MOLAR_FLOW = 100.0

FEATURES = [
    "feed_temperature_K",
    "feed_pressure_Pa",
    "benzene_feed_fraction",
    "number_of_stages",
    "feed_stage_fraction",
    "reflux_ratio",
    "bottoms_flow_mol_s",
]

TARGETS = [
    "distillate_benzene_purity",
    "bottoms_toluene_purity",
    "condenser_duty",
    "reboiler_duty",
]

INPUT_LIMITS = {
    "feed_temperature_K": (345.0, 385.0),
    "feed_pressure_Pa": (101325.0, 200000.0),
    "benzene_feed_fraction": (0.25, 0.75),
    "number_of_stages": (18.0, 35.0),
    # Rounding the sampled 0.30-0.70 relative position produces these limits.
    "feed_stage_fraction": (5.0 / 18.0, 13.0 / 18.0),
    "reflux_ratio": (1.5, 4.5),
    "bottoms_flow_mol_s": (25.0, 75.0),
}


def _column(values, name):
    if isinstance(values, pd.DataFrame):
        return values[name].to_numpy(dtype=float)
    return np.asarray(values, dtype=float)[:, FEATURES.index(name)]


def feasible_benzene_flow_bounds(features):
    """Return physical bounds for benzene molar flow in the distillate."""

    benzene_feed = FEED_MOLAR_FLOW * _column(features, "benzene_feed_fraction")
    bottoms_flow = _column(features, "bottoms_flow_mol_s")
    distillate_flow = FEED_MOLAR_FLOW - bottoms_flow
    lower = np.maximum(0.0, benzene_feed - bottoms_flow)
    upper = np.minimum(benzene_feed, distillate_flow)
    return lower, upper


def purities_to_bounded_split(features, targets, epsilon=1e-6):
    """Express both DWSIM purities as one valid benzene split.

    DWSIM accepts a finite material-balance tolerance, so the two reported
    purities imply slightly different component flows. The weighted projection
    minimizes the sum of squared changes to the two reported purities.
    """

    values = np.asarray(targets, dtype=float)
    benzene_feed = FEED_MOLAR_FLOW * _column(features, "benzene_feed_fraction")
    bottoms_flow = _column(features, "bottoms_flow_mol_s")
    distillate_flow = FEED_MOLAR_FLOW - bottoms_flow

    flow_from_distillate = distillate_flow * values[:, 0]
    flow_from_bottoms = benzene_feed - bottoms_flow * (1.0 - values[:, 1])
    distillate_weight = 1.0 / np.square(distillate_flow)
    bottoms_weight = 1.0 / np.square(bottoms_flow)
    projected_flow = (
        distillate_weight * flow_from_distillate
        + bottoms_weight * flow_from_bottoms
    ) / (distillate_weight + bottoms_weight)

    lower, upper = feasible_benzene_flow_bounds(features)
    width = upper - lower
    if np.any(width <= 0):
        raise ValueError("Inputs produce an empty feasible component-flow interval")

    split = (projected_flow - lower) / width
    return np.clip(split, epsilon, 1.0 - epsilon)


def split_to_purities(features, split):
    """Convert a bounded split to purities that satisfy component balance."""

    split = np.asarray(split, dtype=float)
    benzene_feed = FEED_MOLAR_FLOW * _column(features, "benzene_feed_fraction")
    bottoms_flow = _column(features, "bottoms_flow_mol_s")
    distillate_flow = FEED_MOLAR_FLOW - bottoms_flow
    lower, upper = feasible_benzene_flow_bounds(features)
    distillate_benzene_flow = lower + split * (upper - lower)

    distillate_purity = distillate_benzene_flow / distillate_flow
    bottoms_toluene_purity = 1.0 - (
        benzene_feed - distillate_benzene_flow
    ) / bottoms_flow
    return np.column_stack([distillate_purity, bottoms_toluene_purity])


def component_balance_residual(features, predictions):
    """Return benzene feed minus benzene in the two product streams, mol/s."""

    values = np.asarray(predictions, dtype=float)
    benzene_feed = FEED_MOLAR_FLOW * _column(features, "benzene_feed_fraction")
    bottoms_flow = _column(features, "bottoms_flow_mol_s")
    distillate_flow = FEED_MOLAR_FLOW - bottoms_flow
    return benzene_feed - (
        distillate_flow * values[:, 0]
        + bottoms_flow * (1.0 - values[:, 1])
    )


def project_predictions(features, predictions):
    """Make the two predicted purities satisfy the benzene balance."""

    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Expected four predicted outputs")

    benzene_feed = FEED_MOLAR_FLOW * _column(features, "benzene_feed_fraction")
    bottoms_flow = _column(features, "bottoms_flow_mol_s")
    distillate_flow = FEED_MOLAR_FLOW - bottoms_flow
    flow_from_distillate = distillate_flow * values[:, 0]
    flow_from_bottoms = benzene_feed - bottoms_flow * (1.0 - values[:, 1])

    distillate_weight = 1.0 / np.square(distillate_flow)
    bottoms_weight = 1.0 / np.square(bottoms_flow)
    projected_flow = (
        distillate_weight * flow_from_distillate
        + bottoms_weight * flow_from_bottoms
    ) / (distillate_weight + bottoms_weight)
    lower, upper = feasible_benzene_flow_bounds(features)
    projected_flow = np.clip(projected_flow, lower, upper)

    projected = values.copy()
    projected[:, 0] = projected_flow / distillate_flow
    projected[:, 1] = 1.0 - (benzene_feed - projected_flow) / bottoms_flow
    return projected


class PhysicsProjectedSurrogate(BaseEstimator, RegressorMixin):
    """Fit all outputs, then make the purity predictions physically valid."""

    def __init__(self, regressor):
        self.regressor = regressor

    def fit(self, features, targets):
        values = np.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("Expected four targets: two purities and two duties")
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(features, values)
        self.n_features_in_ = np.asarray(features).shape[1]
        return self

    def predict_raw(self, features):
        check_is_fitted(self, "regressor_")
        return np.asarray(self.regressor_.predict(features), dtype=float)

    def predict(self, features):
        return project_predictions(features, self.predict_raw(features))


class PhysicsConstrainedSurrogate(BaseEstimator, RegressorMixin):
    """Predict one bounded benzene split and calculate both purities.

    The wrapped regressor predicts logit(split), condenser duty, and reboiler
    duty. Applying a logistic inverse and the benzene balance guarantees both
    purity bounds and component balance without clipping predicted purities.
    """

    def __init__(self, regressor, epsilon=1e-6):
        self.regressor = regressor
        self.epsilon = epsilon

    def fit(self, features, targets):
        values = np.asarray(targets, dtype=float)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("Expected four targets: two purities and two duties")

        split = purities_to_bounded_split(features, values, self.epsilon)
        latent_targets = np.column_stack(
            [logit(split), values[:, 2], values[:, 3]]
        )
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(features, latent_targets)
        self.n_features_in_ = np.asarray(features).shape[1]
        return self

    def predict(self, features):
        check_is_fitted(self, "regressor_")
        latent_predictions = np.asarray(self.regressor_.predict(features))
        if latent_predictions.ndim == 1:
            latent_predictions = latent_predictions.reshape(-1, 3)
        purities = split_to_purities(features, expit(latent_predictions[:, 0]))
        return np.column_stack([purities, latent_predictions[:, 1:]])
