from src.preprocessing import run_preprocessing
from src.eda import run_eda
from src.model import run_model

# Chemin vers le dataset
DATA_PATH = "data/train.csv"

def main():
    print("=" * 50)
    print("   SUPPLY CHAIN DEMAND FORECASTING")
    print("=" * 50)

    # Etape 1 : Chargement et nettoyage des données
    print("\n[1/3] Preprocessing...")
    df = run_preprocessing(DATA_PATH)

    # Etape 2 : Exploration et visualisation
    print("\n[2/3] Analyse exploratoire...")
    run_eda(df)

    # Etape 3 : Modélisation et prédiction
    print("\n[3/3] Modélisation...")
    model = run_model(df)

    print("\n✅ Pipeline terminé ! Résultats dans outputs/")

if __name__ == "__main__":
    main()