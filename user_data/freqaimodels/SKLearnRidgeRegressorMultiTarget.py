from __future__ import annotations

from typing import Any

from sklearn.linear_model import Ridge

from freqtrade.freqai.base_models.BaseRegressionModel import BaseRegressionModel
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen


class SKLearnRidgeRegressorMultiTarget(BaseRegressionModel):
    """
    Minimal multi-target regression model for context-feature research.

    Uses scikit-learn, which is already available in the controller venv, so
    the research run does not require installing LightGBM/XGBoost extras.
    """

    def fit(self, data_dictionary: dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        model = Ridge(**self.model_training_parameters)
        model.fit(
            data_dictionary["train_features"],
            data_dictionary["train_labels"],
            sample_weight=data_dictionary["train_weights"],
        )
        return model
