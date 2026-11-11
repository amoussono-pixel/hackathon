# ============================================================
# DATABASE.PY
# Gestion de la base de données SQLite du chatbot d'arômes
# ============================================================

import sqlite3
from pathlib import Path


# ------------------------------------------------------------
# Chemin vers la base de données
# ------------------------------------------------------------
# __file__ correspond au fichier database.py.
# On crée la base de données dans le même dossier que le projet.
DATABASE_PATH = Path(__file__).parent / "aroma_chatbot.db"


# ------------------------------------------------------------
# Fonction de connexion à la base de données
# ------------------------------------------------------------
def get_connection():
    """
    Crée une connexion à la base de données SQLite.

    La connexion est utilisée par les différentes fonctions
    pour créer, enregistrer ou récupérer des informations.
    """

    connection = sqlite3.connect(DATABASE_PATH)

    # Permet de récupérer les résultats sous forme de dictionnaire.
    # Exemple : row["content"] au lieu de row[2].
    connection.row_factory = sqlite3.Row

    return connection


# ------------------------------------------------------------
# Création des tables de la base de données
# ------------------------------------------------------------
def init_database():
    """
    Initialise la base de données et crée les tables
    si elles n'existent pas encore.
    """

    # Ouverture de la connexion à SQLite
    connection = get_connection()

    # Création du curseur permettant d'exécuter les requêtes SQL
    cursor = connection.cursor()


    # ========================================================
    # TABLE 1 : conversations
    # ========================================================
    # Cette table permet d'enregistrer chaque conversation
    # entre un utilisateur et le chatbot.

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            status TEXT DEFAULT 'incomplete'
        )
    """)


    # ========================================================
    # TABLE 2 : messages
    # ========================================================
    # Cette table contient tous les messages échangés.
    #
    # role peut être :
    # - user      → message du client
    # - assistant → réponse du chatbot

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
        )
    """)


    # ========================================================
    # TABLE 3 : proposals
    # ========================================================
    # Cette table contient la proposition d'arôme construite
    # progressivement par le chatbot.

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS proposals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,

            type TEXT,
            application TEXT,
            famille TEXT,
            saveur TEXT,
            profil TEXT,
            intensite TEXT,

            status TEXT DEFAULT 'incomplete',

            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
        )
    """)


    # ========================================================
    # TABLE 4 : missing_information
    # ========================================================
    # Cette table permet d'enregistrer les informations
    # qui manquent encore pour finaliser la proposition.

    # Exemple :
    # - saveur
    # - intensite
    # - profil

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS missing_information (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            field TEXT NOT NULL,

            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
        )
    """)


    # --------------------------------------------------------
    # Validation des modifications
    # --------------------------------------------------------
    # commit() enregistre définitivement les changements
    # dans la base de données.

    connection.commit()


    # --------------------------------------------------------
    # Fermeture de la connexion
    # --------------------------------------------------------
    # On ferme la connexion une fois le travail terminé.
    connection.close()


# ------------------------------------------------------------
# Point d'entrée du programme
# ------------------------------------------------------------
# Cette partie est exécutée uniquement lorsque nous lançons :
#
#     python database.py
#
# Elle permet donc de créer automatiquement les tables.

if __name__ == "__main__":

    # Création de la base et des tables
    init_database()

    # Message permettant de vérifier que tout s'est bien passé
    print("Base de données créée avec succès !")