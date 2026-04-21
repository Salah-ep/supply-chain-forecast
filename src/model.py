import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os


# Les colonnes qu'on donne au modèle comme entrée
FEATURES = ["year", "month", "day","dayofweek", "weekofyear", 
            "store_nbr", "family_encoded", "is_holiday","is_workday",
            "city_encoded", "state_encoded", "type_encoded", "cluster"]

# La colonne qu'on veut prédire
TARGET = "sales"


def split_train_test(df, test_year=2017):
    """
    Sépare les données en train et test selon l'année.
    On entraîne sur les années passées, on teste sur la dernière année.
    C'est comme séparer des données d'entraînement et de validation.
    """
    train = df[df["year"] < test_year]
    test  = df[df["year"] >= test_year]

    print(f"Train : {len(train)} lignes | Test : {len(test)} lignes")
    return train, test


def train_model(train):
    """
    Entraîne un modèle Random Forest sur les données d'entraînement.
    
    Random Forest = un ensemble d'arbres de décision.
    """
    X_train = train[FEATURES]
    y_train = train[TARGET]

    model = RandomForestRegressor(
        n_estimators=80, # 100 arbres de décision
        max_depth=15,
        random_state=42,    # Pour avoir des résultats reproductibles
        n_jobs=-1           # Utilise tous les coeurs du CPU
    )

    model.fit(X_train, y_train)
    print("Modèle entraîné !")
    return model


def evaluate_model(model, test):
    """
    Évalue la performance du modèle sur les données de test.
    On compare les prédictions avec les vraies valeurs.
    """
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
    """
    Trace les vraies ventes vs les prédictions sur un graphique.
    Permet de voir visuellement si le modèle est bon.
    """
    os.makedirs(output_dir, exist_ok=True)

    # On prend un seul magasin pour que le graphique soit lisible
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


def run_model(df):
    """
    Fonction principale qui enchaîne toutes les étapes.
    """
    train, test      = split_train_test(df)
    model            = train_model(train)
    predictions, y_test = evaluate_model(model, test)
    plot_predictions(test, predictions)
    return model