"""Tests de /score et /ask avec un LLM simulé (sans Ollama).

Lancement : python -m unittest test_api -v
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import database

# La base SQLite des tests est jetable : on la redirige avant d'importer l'API.
_tmp = tempfile.TemporaryDirectory()
database.DATABASE_PATH = Path(_tmp.name) / "test.db"

from fastapi.testclient import TestClient  # noqa: E402

import api  # noqa: E402
import scoring  # noqa: E402

PROJECT = {
    "role": "marketing",
    "salesZone": "europe",
    "market": "perfume",
    "quantity": "small",
    "positioning": "premium",
    "workedWithGivaudan": False,
    "description": "Un parfum d'intérieur inspiré des jardins méditerranéens.",
}
SOURCE = api.Source(chapter="Doc", text="Extrait", score=0.9)


class ScoreTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_score_combine_criteres_et_description(self):
        # critères : 87 ; une rubrique explicite : 25 -> 50
        evidence = '{"idee":"jardins méditerranéens","usage":null,"cible":null,"contraintes":null}'
        with patch.object(api.rag, "chat", return_value=evidence):
            response = self.client.post("/score", json=PROJECT)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["score"], 50)
        self.assertEqual(response.json()["criteria_score"], 87)
        self.assertEqual(response.json()["description_score"], 25)

    def test_le_llm_recoit_la_description_mais_pas_d_identite(self):
        with patch.object(api.rag, "chat", return_value='{"idee":null,"usage":null,"cible":null,"contraintes":null}') as chat:
            self.client.post("/score", json=PROJECT)
        prompt = " ".join(m["content"] for m in chat.call_args.args[0])
        self.assertIn("jardins méditerranéens", prompt)
        self.assertTrue(chat.call_args.kwargs["json_output"])

    def test_une_description_vague_ne_peut_pas_atteindre_65(self):
        with patch.object(api.rag, "chat", return_value='{"idee":null,"usage":null,"cible":null,"contraintes":null}'):
            response = self.client.post("/score", json={**PROJECT, "description": "Je veux vendre quelque chose de bon."})
        self.assertLess(response.json()["score"], 65)

    def test_plusieurs_citations_separees_par_des_virgules(self):
        raw = '{"idee":"citron, peche","usage":"boisson","cible":null,"contraintes":null}'
        score, fields = scoring.description_score(raw, "Une boisson au citron et a la peche.")
        self.assertEqual(score, 50)
        self.assertEqual(fields, ["idee", "usage"])

    def test_un_projet_detaille_peut_atteindre_65(self):
        description = "Boisson citronnée pour les jeunes adultes, sans sucre ajouté."
        evidence = '{"idee":"citronnée","usage":"Boisson","cible":"jeunes adultes","contraintes":"sans sucre ajouté"}'
        with patch.object(api.rag, "chat", return_value=evidence):
            response = self.client.post("/score", json={**PROJECT, "description": description})
        self.assertGreaterEqual(response.json()["score"], 65)

    def test_reponses_llm_invalides_retournent_502(self):
        for raw in ["pas de json", '{"score": 120}', '{"idee":true,"usage":null,"cible":null,"contraintes":null}', '{"idee":"inventé","usage":null,"cible":null,"contraintes":null}', '{"idee":"parfum","usage":"parfum","cible":null,"contraintes":null}']:
            with patch.object(api.rag, "chat", return_value=raw):
                self.assertEqual(self.client.post("/score", json=PROJECT).status_code, 502, raw)

    def test_ollama_indisponible_retourne_502(self):
        with patch.object(api.rag, "chat", side_effect=ConnectionError("down")):
            self.assertEqual(self.client.post("/score", json=PROJECT).status_code, 502)

    def test_requete_invalide_retourne_422(self):
        bad = {k: v for k, v in PROJECT.items() if k != "market"}
        self.assertEqual(self.client.post("/score", json=bad).status_code, 422)


class AskTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        patcher = patch.object(api, "retrieve", return_value=[SOURCE])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_la_conversation_est_poursuivie_avec_son_historique(self):
        with patch.object(api.rag, "chat", return_value="Bonjour, enchanté.") as chat:
            first = self.client.post("/ask", json={"question": "Bonjour"}).json()
            self.assertIsInstance(first["conversation_id"], int)
            self.client.post(
                "/ask",
                json={"question": "Un arôme fruité ?", "conversation_id": first["conversation_id"], "context": "Boisson premium"},
            )
        messages = chat.call_args.args[0]
        self.assertIn("GivBot", messages[0]["content"])
        self.assertIn("Boisson premium", messages[0]["content"])
        self.assertEqual([m["role"] for m in messages[1:-1]], ["user", "assistant"])
        self.assertEqual(messages[1]["content"], "Bonjour")

    def test_conversation_inconnue_retourne_404(self):
        response = self.client.post("/ask", json={"question": "Bonjour", "conversation_id": 99999})
        self.assertEqual(response.status_code, 404)

    def test_un_echec_du_llm_n_enregistre_rien(self):
        conversation_id = database.create_conversation()
        with patch.object(api.rag, "chat", side_effect=ConnectionError("down")):
            response = self.client.post("/ask", json={"question": "Bonjour", "conversation_id": conversation_id})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(database.get_messages(conversation_id), [])


class ApiKeyTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)
        patcher = patch.object(config, "API_KEY", "secret")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cle_requise_sur_score_mais_pas_sur_health(self):
        with patch.object(api.rag, "chat", return_value='{"idee":null,"usage":null,"cible":null,"contraintes":null}'):
            self.assertEqual(self.client.post("/score", json=PROJECT).status_code, 401)
            self.assertEqual(
                self.client.post("/score", json=PROJECT, headers={"Authorization": "Bearer faux"}).status_code, 401
            )
            self.assertEqual(
                self.client.post("/score", json=PROJECT, headers={"Authorization": "Bearer secret"}).status_code, 200
            )


if __name__ == "__main__":
    unittest.main()
