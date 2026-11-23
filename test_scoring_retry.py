"""Tests de correction des réponses invalides du modèle."""

import unittest
from unittest.mock import patch

import scoring


class ScoringRetryTests(unittest.TestCase):
    def setUp(self):
        self.project = {
            "market": "agro",
            "quantity": "small",
            "positioning": "mid",
            "description": "Une boisson au citron.",
            "worked_with_givaudan": False,
            "sales_zone": "europe",
            "role": "marketing",
        }

    def test_corrige_une_citation_inventee_avant_de_calculer_le_score(self):
        invalid = '{"idee":"citron","usage":"boisson","cible":"enfants","contraintes":null}'
        valid = '{"idee":"citron","usage":"boisson","cible":null,"contraintes":null}'
        with patch.object(scoring.rag, "chat", side_effect=[invalid, valid]) as chat:
            result = scoring.score_project(**self.project)
        self.assertEqual(result["description_score"], 50)
        self.assertEqual(chat.call_count, 2)
        correction = chat.call_args.args[0][-1]["content"]
        self.assertIn("Citation invalide", correction)
        self.assertIn("null", correction)

    def test_reponses_invalides_epuisees_retournent_une_erreur(self):
        with patch.object(scoring.rag, "chat", return_value="invalide") as chat:
            with self.assertRaises(scoring.ScoringError):
                scoring.score_project(**self.project)
        self.assertEqual(chat.call_count, 3)

    def test_erreur_ollama_n_est_pas_masquee(self):
        with patch.object(scoring.rag, "chat", side_effect=ConnectionError("indisponible")) as chat:
            with self.assertRaises(ConnectionError):
                scoring.score_project(**self.project)
        self.assertEqual(chat.call_count, 1)


if __name__ == "__main__":
    unittest.main()
