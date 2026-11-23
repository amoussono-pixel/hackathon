"""API de questions/réponses sur l'histoire (RAG + Ollama).

Lancement :
    uvicorn api:app --reload
Swagger : http://localhost:8000/docs
"""

import hmac

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

import config
import rag
import database
import scoring

SYSTEM_PROMPT = """Tu es GivBot, l'assistant courtois de Givaudan. Tu aides un client à préciser son projet d'arôme ou de parfum.
Règles :
- Vouvoie le client, sois chaleureux et poli, et pose au plus une question à la fois.
- Appuie-toi sur les extraits de documents fournis ; s'ils ne contiennent pas la réponse, dis-le honnêtement sans inventer.
- Ne promets ni prix, ni délai, ni disponibilité.
- Il s'agit d'une démonstration : les références et données des documents sont fictives et ne correspondent pas au catalogue réel.
- Ne prétends pas avoir commandé, transmis une demande ou généré une fiche : ces actions ne sont pas disponibles.
- Les réglages sensoriels sont un aperçu statique : ne prétends pas les avoir modifiés. Formule tes suggestions comme des pistes à étudier.
- Réponds en français, en quelques phrases maximum."""

app = FastAPI(
    title="API IA Givaudan",
    description="Chat et évaluation de projets, à partir des documents chargés dans la base vectorielle. "
                "Les réponses sont générées par un LLM Ollama.",
    version="1.1.0",
)
# Initialise la base de données SQLite au démarrage de l'API
database.init_database()


def require_api_key(authorization: str | None = Header(default=None)):
    """Si API_KEY est définie, exige `Authorization: Bearer <API_KEY>`."""
    if not config.API_KEY:
        return
    expected = f"Bearer {config.API_KEY}"
    if not hmac.compare_digest((authorization or "").encode(), expected.encode()):
        raise HTTPException(401, "Clé d'API absente ou invalide.")


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        examples=["Je cherche un arôme fruité pour une boisson."]
    )

    top_k: int = Field(
        config.TOP_K, 
        ge=1,
        le=20,
        description="Nombre de passages à retrouver"
    )
    # Identifiant permettant de retrouver la même conversation
    conversation_id: int | None = None
    # Résumé du projet du client, pour que GivBot en tienne compte
    context: str | None = Field(default=None, max_length=4000)


class Source(BaseModel):
    chapter: str
    text: str
    score: float = Field(description="Similarité cosinus (1 = identique)")


class AskResponse(BaseModel):
    conversation_id: int
    question: str
    answer: str
    model: str
    sources: list[Source]


class ScoreRequest(BaseModel):
    role: str
    salesZone: str
    market: str
    quantity: str
    positioning: str
    workedWithGivaudan: bool
    description: str = Field(..., min_length=1, max_length=4000)


class ScoreResponse(BaseModel):
    score: int = Field(ge=0, le=100)
    criteria_score: int
    description_score: int
    reason: str

# ============================================================
# Recherche des passages les plus pertinents
# ============================================================
def retrieve(question: str, top_k: int) -> list[Source]:
    collection = rag.get_collection()
    if collection.count() == 0:
        raise HTTPException(503, "La base est vide : lance d'abord `python ingest.py`.")
    result = collection.query(
        query_embeddings=[rag.embed_query(question)],
        n_results=min(top_k, collection.count()),
    )
    return [
        Source(chapter=meta["chapter"], text=doc, score=round(1 - dist, 4))
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0])
    ]

# ============================================================
# Vérification de l'état de l'API
# ============================================================

@app.get("/health", tags=["système"])
def health():
    """Vérifie que la base est chargée et indique les modèles utilisés."""
    return {
        "chunks": rag.get_collection().count(),
        "llm_model": config.LLM_MODEL,
        "embed_model": config.EMBED_MODEL,
    }

# ============================================================
# Recherche de passages sans génération de réponse
# ===========================================================
@app.post("/search", response_model=list[Source], tags=["histoire"], dependencies=[Depends(require_api_key)])
def search(req: AskRequest):
    """Retourne seulement les passages pertinents (sans appeler le LLM)."""
    return retrieve(req.question, req.top_k)

# ============================================================
# Génération d'une réponse à partir des passages retrouvés
# ============================================================
@app.post("/ask", response_model=AskResponse, tags=["histoire"], dependencies=[Depends(require_api_key)])
def ask(req: AskRequest):
    """Répond à une question ; avec `conversation_id`, poursuit la conversation existante."""
    if req.conversation_id is None:
        conversation_id = database.create_conversation()
    elif database.conversation_exists(req.conversation_id):
        conversation_id = req.conversation_id
    else:
        raise HTTPException(404, "Conversation introuvable.")
    history = database.get_messages(conversation_id)
    sources = retrieve(req.question, req.top_k)
    context = "\n\n---\n\n".join(f"[{s.chapter}]\n{s.text}" for s in sources)
    system = SYSTEM_PROMPT
    if req.context:
        system += f"\n\nProjet du client :\n{req.context}"
    messages = [
        {"role": "system", "content": system},
        *history,
        {"role": "user", "content": f"Extraits de documents :\n\n{context}\n\nQuestion : {req.question}"},
    ]
    try:
        answer = rag.chat(messages)
    except Exception as exc:  # Ollama injoignable, modèle absent...
        raise HTTPException(502, f"Erreur Ollama : {exc}") from exc
    database.add_message(conversation_id, "user", req.question)
    database.add_message(conversation_id, "assistant", answer.strip())
    return AskResponse(
        conversation_id=conversation_id,
        question=req.question,
        answer=answer.strip(),
        model=config.LLM_MODEL,
        sources=sources,
    )

# ============================================================
# Score de viabilité d'un projet
# ============================================================
@app.post("/score", response_model=ScoreResponse, tags=["projet"], dependencies=[Depends(require_api_key)])
def score(req: ScoreRequest):
    """Retourne un score de 0 à 100 pour un projet (sans nom, email ni entreprise)."""
    try:
        return scoring.score_project(
            market=req.market,
            quantity=req.quantity,
            positioning=req.positioning,
            description=req.description,
            worked_with_givaudan=req.workedWithGivaudan,
            sales_zone=req.salesZone,
            role=req.role,
        )
    except scoring.ScoringError as exc:
        raise HTTPException(502, str(exc)) from exc
    except Exception as exc:  # Ollama injoignable, modèle absent...
        raise HTTPException(502, f"Erreur Ollama : {exc}") from exc

# ============================================================
# Lancement de l'API
# ===========================================================
if __name__ == "__main__":
    import uvicorn
    
# Lance le serveur FastAPI sur le port 8000
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
