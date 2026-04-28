import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import time

FEATURES = ["year", "month", "day", "dayofweek",
            "store_nbr", "family_encoded", "is_holiday",
            "city_encoded", "state_encoded", "type_encoded",
            "cluster", "onpromotion",
            "lag_1", "lag_7", "lag_30", "rolling_7", "rolling_30",
            "family_avg_sales", "store_avg_sales", "trend_7",
            "promo_ratio", "is_month_start", "is_month_end", "is_weekend"]

TARGET = "sales"


def get_models():
    """
    Retourne les 3 modèles à comparer avec leurs paramètres.
    """
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=50,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            gamma=0.1,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        "LightGBM": LGBMRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
    }


def compare_models(df):
    """
    Entraîne et évalue chaque modèle, puis trace un graphique comparatif.
    """
    # Split train/test
    train = df[df["year"] < 2017]
    test  = df[df["year"] >= 2017]

    # Échantillon d'entraînement
    train_sample = train.sample(frac=0.55, random_state=42)
    print(f"Échantillon : {len(train_sample)} lignes\n")

    X_train = train_sample[FEATURES]
    y_train = train_sample[TARGET]
    X_test  = test[FEATURES]
    y_test  = test[TARGET]

    results = {}

    for name, model in get_models().items():
        print(f"Entraînement {name}...")

        # On mesure le temps d'entraînement
        start = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start

        # Prédictions
        preds = np.clip(model.predict(X_test), 0, None)

        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        results[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "Temps (s)": round(duration, 1)
        }

        print(f"  MAE={mae:.2f} | RMSE={rmse:.2f} | Temps={duration:.1f}s\n")

    # Affiche le tableau comparatif
    results_df = pd.DataFrame(results).T
    print("--- Comparaison finale ---")
    print(results_df)

    # Graphique comparatif
    plot_comparison(results_df)

    return results_df


def plot_comparison(results_df, output_dir="outputs"):
    """
    Trace un graphique comparatif MAE et RMSE pour les 3 modèles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # MAE
    axes[0].bar(results_df.index, results_df["MAE"], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[0].set_title("MAE par modèle (plus bas = meilleur)")
    axes[0].set_ylabel("MAE")

    # RMSE
    axes[1].bar(results_df.index, results_df["RMSE"], color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_title("RMSE par modèle (plus bas = meilleur)")
    axes[1].set_ylabel("RMSE")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png")
    plt.show()
    print("Graphique sauvegardé : model_comparison.png")