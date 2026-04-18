import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Style global des graphiques
sns.set_theme(style="darkgrid")


def plot_sales_over_time(df, output_dir="outputs"):
    """
    Trace l'évolution globale des ventes dans le temps.
    On regroupe toutes les ventes par date (somme) pour voir la tendance générale.
    """
    os.makedirs(output_dir, exist_ok=True)

    # On additionne les ventes de tous les magasins par date
    # C'est un GROUP BY date
    daily_sales = df.groupby("date")["sales"].sum().reset_index()

    plt.figure(figsize=(14, 5))
    plt.plot(daily_sales["date"], daily_sales["sales"], linewidth=1)
    plt.title("Ventes totales par jour")
    plt.xlabel("Date")
    plt.ylabel("Ventes")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ventes_par_jour.png")
    plt.show()
    print("Graphique sauvegardé : ventes_par_jour.png")


def plot_sales_by_month(df, output_dir="outputs"):
    """
    Moyenne des ventes par mois — permet de voir la saisonnalité.
    Ex : est-ce que décembre est toujours plus fort ?
    """
    monthly = df.groupby("month")["sales"].mean().reset_index()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=monthly, x="month", y="sales")
    plt.title("Ventes moyennes par mois")
    plt.xlabel("Mois")
    plt.ylabel("Ventes moyennes")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ventes_par_mois.png")
    plt.show()
    print("Graphique sauvegardé : ventes_par_mois.png")


def plot_sales_by_store(df, top_n=10, output_dir="outputs"):
    """
    Compare les ventes totales des top N magasins.
    Utile pour détecter si certains magasins dominent les données.
    """
    store_sales = df.groupby("store_nbr")["sales"].sum().reset_index()
    store_sales = store_sales.sort_values("sales", ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=store_sales, x="store_nbr", y="sales")
    plt.title(f"Top {top_n} magasins par ventes totales")
    plt.xlabel("Numéro de magasin")
    plt.ylabel("Ventes totales")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ventes_par_magasin.png")
    plt.show()
    print("Graphique sauvegardé : ventes_par_magasin.png")


def print_statistics(df):
    """
    Affiche les statistiques de base du dataset.
    Equivalent d'un résumé rapide pour comprendre l'échelle des données.
    """
    print("\n--- Aperçu des données ---")
    print(df.head())                  # Les 5 premières lignes
    print("\n--- Types des colonnes ---")
    print(df.dtypes)                  # Type de chaque colonne
    print("\n--- Statistiques descriptives ---")
    print(df.describe())              # Min, max, moyenne, écart-type...


def run_eda(df):
    """
    Fonction principale qui enchaîne toutes les analyses.
    """
    print_statistics(df)
    plot_sales_over_time(df)
    plot_sales_by_month(df)
    plot_sales_by_store(df)