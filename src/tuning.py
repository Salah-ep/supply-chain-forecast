import numpy as np
import optuna
import warnings
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES = ["year", "month", "day", "dayofweek",
            "store_nbr", "family_encoded", "is_holiday",
            "city_encoded", "state_encoded", "type_encoded",
            "cluster", "onpromotion",
            "lag_1", "lag_7", "lag_30", "rolling_7", "rolling_30",
            "family_avg_sales", "store_avg_sales", "trend_7",
            "promo_ratio", "is_month_start", "is_month_end", "is_weekend"]

TARGET = "sales"


def objective(trial, X_train, y_train):
    """
    Fonction objectif qu'Optuna va minimiser.
    À chaque trial, Optuna propose des paramètres différents
    et on retourne le MAE — Optuna cherche à le minimiser.
    """
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 100, 500),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3),
        "max_depth":       trial.suggest_int("max_depth", 4, 12),
        "num_leaves":      trial.suggest_int("num_leaves", 20, 150),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1
    }

    model = LGBMRegressor(**params)

    # Cross-validation sur 3 folds — plus fiable qu'un simple train/test
    scores = cross_val_score(
        model, X_train, y_train,
        cv=3,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )

    return -scores.mean()  # On retourne le MAE positif


def run_tuning(df, n_trials=20):
    """
    Lance la recherche des meilleurs paramètres.
    n_trials = nombre d'essais — plus c'est grand, meilleur le résultat
    mais plus c'est long. 20 est un bon compromis.
    """
    train = df[df["year"] < 2017]
    train_sample = train.sample(frac=0.3, random_state=42)

    X_train = train_sample[FEATURES]
    y_train = train_sample[TARGET]

    print(f"Tuning LightGBM sur {len(train_sample)} lignes ({n_trials} trials)...")

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=True
    )

    print(f"\n✅ Meilleur MAE trouvé : {study.best_value:.2f}")
    print(f"Meilleurs paramètres :")
    for key, val in study.best_params.items():
        print(f"  {key}: {val}")

    return study.best_params