import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder




def load_data(filepath):
    """
    Charge le dataset CSV depuis le chemin donné et Retourne un DataFrame pandas.
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


def encode_categorical(df):
    """
    Convertit les colonnes texte en nombres.
    Le modèle ML ne comprend que des chiffres — pas du texte.
    LabelEncoder remplace chaque catégorie unique par un entier.
    """
    le = LabelEncoder()
    df["family_encoded"] = le.fit_transform(df["family"])
    print(f"Colonne 'family' encodée : {df['family'].nunique()} catégories")
    return df

def merge_holidays(df, holidays_filepath):
    """
    Fusionne le dataset principal avec les jours fériés.
    C'est un LEFT JOIN sur la date — toutes les lignes de df
    sont gardées, on ajoute juste l'info 'est-ce un jour férié ?'
    """
    holidays=pd.read_csv(holidays_filepath)
    holidays["date"]=pd.to_datetime(holidays["date"])
    # On garde seulement les colonnes utiles
    holidays=holidays[["date","type"]].drop_duplicates(subset="date")
    # On crée les deux colonnes binaires
    holidays["is_holiday"] = holidays["type"].isin(
        ["Holiday", "Transfer", "Additional", "Bridge"]
    ).astype(int)

    holidays["is_workday"] = (holidays["type"] == "Work Day").astype(int)

    # On supprime la colonne type — on n'en a plus besoin
    holidays = holidays.drop(columns=["type"])

    # LEFT JOIN sur la date
    df = df.merge(holidays, on="date", how="left")

    # Les jours sans info = pas de jour férié → on remplace NaN par 0
    df["is_holiday"] = df["is_holiday"].fillna(0).astype(int)
    df["is_workday"] = df["is_workday"].fillna(0).astype(int)

    print(f"Jours fériés intégrés : {df['is_holiday'].sum()} lignes concernées")
    return df

def merge_stores(df, stores_filepath):
    """
    Fusionne le dataset principal avec les infos des magasins.
    JOIN sur store_nbr — on enrichit chaque ligne avec
    le type, la ville, l'état et le cluster du magasin.
    """
    stores = pd.read_csv(stores_filepath)

    # Encode les colonnes texte en nombres
    le = LabelEncoder()
    stores["city_encoded"]  = le.fit_transform(stores["city"])
    stores["state_encoded"] = le.fit_transform(stores["state"])
    stores["type_encoded"]  = le.fit_transform(stores["type"])

    # On garde seulement les colonnes utiles
    stores = stores[["store_nbr", "city_encoded", "state_encoded", "type_encoded", "cluster"]]

    # LEFT JOIN sur store_nbr
    df = df.merge(stores, on="store_nbr", how="left")

    print(f"Infos magasins intégrées : {stores.shape[0]} magasins")
    return df
    
def run_preprocessing(filepath):
    
    df = load_data(filepath)
    df = convert_dates(df)
    check_missing_values(df)
    df = extract_date_features(df)
    df = encode_categorical(df) 
    df = merge_holidays(df, "data/holidays_events.csv") 
    df = merge_stores(df, "data/stores.csv")
    
    return df