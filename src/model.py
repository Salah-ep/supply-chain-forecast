import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

FEATURES = ["year", "month", "day", "dayofweek",
            "store_nbr", "family_encoded", "is_holiday",
            "city_encoded", "state_encoded", "type_encoded", 
            "cluster", "onpromotion"] 

TARGET = "sales"

def split_train_test(df, test_year=2017):
    train = df[df["year"] < test_year]
    test  = df[df["year"] >= test_year]
    print(f"Train : {len(train)} lignes | Test : {len(test)} lignes")
    return train, test

def train_model(train):

    # On échantillonne 40% pour ne pas saturer la RAM
    train_sample = train.sample(frac=0.4, random_state=42)
    print(f"Échantillon d'entraînement : {len(train_sample)} lignes")

    X_train = train_sample[FEATURES]
    y_train = train_sample[TARGET]

    model = XGBRegressor(
    n_estimators=300,       # réduit de 500 à 300
    learning_rate=0.1,      # augmenté de 0.05 à 0.1
    max_depth=8,            # augmenté de 6 à 8
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    min_child_weight=5,     #  évite le surapprentissage
    gamma=0.1,
    n_jobs=-1,
    verbosity=0
)

    model.fit(X_train, y_train)
    print("Modèle XGBoost entraîné !")
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

    store = test["store_nbr"].iloc[0]
    mask  = test["store_nbr"] == store

    plt.figure(figsize=(14, 5))
    plt.plot(test["date"].values[mask], test[TARGET].values[mask],
            label="Réel", linewidth=1)
    plt.plot(test["date"].values[mask], predictions[mask],
            label="Prédit", linewidth=1, linestyle="--")
    plt.title(f"Réel vs Prédit — Magasin {store}")
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
    plt.title("Importance des features — XGBoost")
    plt.ylabel("Score d'importance")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.show()
    print("Graphique sauvegardé : feature_importance.png")


def run_model(df):
    train, test         = split_train_test(df)
    model               = train_model(train)
    predictions, y_test = evaluate_model(model, test)
    plot_predictions(test, predictions)
    plot_feature_importance(model)
    return model
