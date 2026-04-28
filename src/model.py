import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
FEATURES = ["year", "month", "day", "dayofweek",
            "store_nbr", "family_encoded", "is_holiday",
            "city_encoded", "state_encoded", "type_encoded",
            "cluster", "onpromotion",
            "lag_1", "lag_7", "lag_30", "rolling_7", "rolling_30",
            "family_avg_sales", "store_avg_sales", "trend_7",
            "promo_ratio", "is_month_start", "is_month_end", "is_weekend"]

TARGET = "sales"

def split_train_test(df, test_year=2017):
    train = df[df["year"] < test_year]
    test  = df[df["year"] >= test_year]
    print(f"Train : {len(train)} lignes | Test : {len(test)} lignes")
    return train, test

def train_model(train, params=None):
    """
    Entraîne LightGBM avec les paramètres donnés.
    Si params=None, utilise les paramètres par défaut.
    """
    from lightgbm import LGBMRegressor

    train_sample = train.sample(frac=0.55, random_state=42)
    print(f"Échantillon d'entraînement : {len(train_sample)} lignes")

    X_train = train_sample[FEATURES]
    y_train = train_sample[TARGET]

    if params is None:
        params = {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "max_depth": 8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        }

    # On ajoute les paramètres fixes
    params["random_state"] = 42
    params["n_jobs"]       = -1
    params["verbose"]      = -1

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    print("Modèle LightGBM entraîné !")
    return model


def evaluate_model(model, test):
    X_test = test[FEATURES]
    y_test = test[TARGET]

    predictions = model.predict(X_test)

    mae  = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print(f"\n--- Performance du modèle ---")
    print(f"MAE  (erreur absolue moyenne) : {mae:.2f}")
    print(f"RMSE (erreur quadratique)     : {rmse:.2f}")

    return predictions, y_test

def plot_predictions(test, predictions, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    # Trouve le bon code pour BEVERAGES
    beverages_code = test[test["family"] == "BEVERAGES"]["family_encoded"].iloc[0]
    mask = (test["store_nbr"] == 1) & (test["family_encoded"] == beverages_code)

    # Clip les prédictions négatives à 0
    preds_clipped = np.clip(predictions, 0, None)

    plt.figure(figsize=(14, 5))
    plt.plot(test["date"].values[mask], test[TARGET].values[mask],
            label="Réel", linewidth=1.5)
    plt.plot(test["date"].values[mask], preds_clipped[mask],
            label="Prédit", linewidth=1.5, linestyle="--", color="orange")
    plt.title("Réel vs Prédit — Magasin 1 | Famille BEVERAGES")
    plt.xlabel("Date")
    plt.ylabel("Ventes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/predictions.png")
    plt.show()
    print("Graphique sauvegardé : predictions.png")


def plot_feature_importance(model, output_dir="outputs"):
    """
    Affiche l'importance de chaque feature pour le modèle.
    Permet de savoir quelles colonnes influencent le plus les prédictions.
    """
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importance.plot(kind="bar")
    plt.title("Importance des features — LightGBM")
    plt.ylabel("Score d'importance")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.show()
    print("Graphique sauvegardé : feature_importance.png")


def run_model(df, params=None):
    train, test         = split_train_test(df)
    model               = train_model(train, params)
    predictions, y_test = evaluate_model(model, test)
    plot_predictions(test, predictions)
    plot_feature_importance(model)
    return model
