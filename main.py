
from src.preprocessing import run_preprocessing
from src.eda import run_eda
from src.tuning import run_tuning
from src.model import run_model

DATA_PATH = "data/train.csv"

def main():
    print("=" * 50)
    print("   SUPPLY CHAIN DEMAND FORECASTING")
    print("=" * 50)

    print("\n[1/3] Preprocessing...")
    df = run_preprocessing(DATA_PATH)

    print("\n[2/3] Hyperparameter tuning LightGBM...")
    best_params = run_tuning(df, n_trials=20)

    print("\n[3/3] Modélisation LightGBM avec meilleurs paramètres...")
    run_model(df, best_params)

    print("\n✅ Pipeline terminé ! Résultats dans outputs/")

if __name__ == "__main__":
    main()