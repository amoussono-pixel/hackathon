"""Score de viabilité d'un projet (0 à 100).

Le score combine :
- une note déterministe sur les critères structurés (marché, quantité, positionnement) ;
- des éléments explicites extraits par le LLM de la description du projet.
Les pondérations et barèmes sont provisoires : ils se règlent ici, sans toucher à l'API.
"""

from __future__ import annotations

import json
import rag

# Grille de démonstration : chaque rubrique présente vaut 25 points sur la
# description. Une description sous 50 ne peut pas atteindre le seuil de 65.
WEIGHT_CRITERIA = 0.4
WEIGHT_DESCRIPTION = 0.6
MAX_EXTRACTION_ATTEMPTS = 3

QUANTITY_POINTS = {"sample": 40, "small": 60, "medium": 80, "large": 100}
POSITIONING_POINTS = {"entry": 60, "mid": 80, "premium": 100}
MARKET_POINTS = {"agro": 100, "perfume": 100, "beauty": 100, "home": 80, "other": 50}

SCORING_PROMPT = """Tu es un outil d'extraction STRICTE. Cherche exactement ces informations dans le texte fourni. Chaque citation doit être copiée mot pour mot, sans paraphrase. Si l'information n'apparaît PAS, retourne null.

Règles :
1. Ne jamais inventer, inférer ou déduire.
2. Chaque valeur doit être un extrait textuel direct du texte, ou null.
3. Si tu hésites, c'est null.
4. "sans" n'est une contrainte que s'il est écrit avec la contrainte (ex: "sans sucre").

Réponds UNIQUEMENT avec ce JSON valide :
{"idee": null ou texte, "usage": null ou texte, "cible": null ou texte, "contraintes": null ou texte}

Référence : le brief peut contenir "Type de produit", "Secteur", "Zone de vente", etc. Accepte uniquement :
- "Type de produit" ou "usage" → usage
- Rien pour idée sauf si "odeur", "saveur", "inspiration", "arôme" apparaît
- Rien pour cible sauf si "public", "clients", "enfants", "adultes", etc. apparaît
- Rien pour contraintes sauf si "sans", "allergies", "restrictions" apparaît

Exemple : "Type de produit : Boisson\nSecteur : Agroalimentaire"
→ {"idee":null,"usage":"Boisson","cible":null,"contraintes":null}

JSON uniquement, pas de texte supplémentaire."""


class ScoringError(Exception):
    pass


def criteria_score(market: str, quantity: str, positioning: str) -> int:
    points = [
        MARKET_POINTS.get(market, 50),
        QUANTITY_POINTS.get(quantity, 40),
        POSITIONING_POINTS.get(positioning, 60),
    ]
    return round(sum(points) / len(points))


def description_score(raw: str, description: str) -> tuple[int, list[str]]:
    """Ne compte que les citations présentes dans la description du client."""
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ScoringError("Réponse du LLM illisible.") from exc
    fields = ("idee", "usage", "cible", "contraintes")
    if not isinstance(data, dict) or set(data) != set(fields):
        raise ScoringError("Rubriques du LLM invalides.")
    present = []
    used_citations = set()
    desc_lower = description.casefold()
    for field in fields:
        value = data[field]
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ScoringError(f"Citation invalide pour la rubrique {field}.")
        parts = [part.strip().casefold() for part in value.split(",")]
        if any(not part or part not in desc_lower or part in used_citations for part in parts):
            raise ScoringError(f"Citation invalide pour la rubrique {field}.")
        used_citations.update(parts)
        present.append(field)
    return 25 * len(present), present


def score_project(
    *,
    market: str,
    quantity: str,
    positioning: str,
    description: str,
    worked_with_givaudan: bool,
    sales_zone: str,
    role: str,
) -> dict:
    messages = [
        {"role": "system", "content": SCORING_PROMPT},
        {
            "role": "user",
            "content": description,
        },
    ]
    for attempt in range(MAX_EXTRACTION_ATTEMPTS):
        response = rag.chat(messages, json_output=True)
        try:
            detail_score, present = description_score(response, description)
            break
        except ScoringError as exc:
            if attempt == MAX_EXTRACTION_ATTEMPTS - 1:
                raise
            messages.extend([
                {"role": "assistant", "content": response},
                {
                    "role": "user",
                    "content": (
                        f"Ta réponse est invalide : {exc} "
                        "Corrige le JSON. Recopie uniquement des extraits de la "
                        "description initiale, ou null si aucun extrait ne convient."
                    ),
                },
            ])
    by_criteria = criteria_score(market, quantity, positioning)
    final = round(WEIGHT_CRITERIA * by_criteria + WEIGHT_DESCRIPTION * detail_score)
    if detail_score < 50:
        final = min(final, 64)
    reason = (
        f"Rubriques explicites détectées : {', '.join(present) if present else 'aucune'}. "
        "Grille de démonstration ; ne constitue pas une décision métier."
    )
    return {
        "score": final,
        "criteria_score": by_criteria,
        "description_score": detail_score,
        "reason": reason,
    }
