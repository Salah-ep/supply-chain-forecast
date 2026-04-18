import pandas as pd
import numpy as np

def load_data(filepath):
    """
    Charge le dataset CSV depuis le chemin donné.
    Retourne un DataFrame pandas.
    """
    df = pd.read_csv(filepath)
    print(f"Dataset chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
    return df


def convert_dates(df, date_column="date"):
    """
    Convertit la colonne de dates en type datetime.
    Sans ça, Python traite les dates comme du texte — on ne peut pas
    faire d'opérations temporelles dessus.
    """
    df[date_column] = pd.to_datetime(df[date_column])
    print(f"Colonne '{date_column}' convertie en datetime")
    return df


def check_missing_values(df):
    """
    Affiche le nombre de valeurs manquantes par colonne.
    Equivalent d'un tableau de bord de qualité des données.
    """
    missing = df.isnull().sum()
    print("\n--- Valeurs manquantes ---")
    print(missing[missing > 0] if missing.sum() > 0 else "Aucune valeur manquante")
    return missing


def extract_date_features(df, date_column="date"):
    """
    Extrait des features utiles depuis la date :
    jour, mois, année, jour de la semaine.
    Ces infos sont très importantes pour la prévision de demande
    (saisonnalité, weekends, fin de mois...).
    """
    df["year"]       = df[date_column].dt.year
    df["month"]      = df[date_column].dt.month
    df["day"]        = df[date_column].dt.day
    df["dayofweek"]  = df[date_column].dt.dayofweek  # 0=lundi, 6=dimanche
    df["weekofyear"] = df[date_column].dt.isocalendar().week.astype(int)
    print("Features temporelles extraites : year, month, day, dayofweek, weekofyear")
    return df


def run_preprocessing(filepath):
    """
    Fonction principale qui enchaîne toutes les étapes.
    C'est le 'main' de ce module.
    """
    df = load_data(filepath)
    df = convert_dates(df)
    check_missing_values(df)
    df = extract_date_features(df)
    return df