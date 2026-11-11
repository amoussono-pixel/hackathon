"""API de questions/réponses sur l'histoire (RAG + Ollama).

Lancement :
    uvicorn api:app --reload
Swagger : http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
import rag
import database

SYSTEM_PROMPT = """Tu es un conteur bienveillant qui répond aux questions d'enfants sur une histoire.
Règles :
- Réponds UNIQUEMENT à partir des extraits de l'histoire fournis.
- Si la réponse ne se trouve pas dans les extraits, dis gentiment que l'histoire ne le précise pas.
- Utilise des phrases simples et claires, adaptées à un enfant.
- Réponds en français, en quelques phrases maximum."""

app = FastAPI(
    title="Story RAG API",
    description="Pose des questions sur l'histoire chargée dans la base vectorielle. "
                "Les réponses sont générées par un LLM Ollama à partir des passages retrouvés.",
    version="1.0.0",
)
# Initialise la base de données SQLite au démarrage de l'API
database.init_database()

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
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

   

class Source(BaseModel):
    chapter: str
    text: str
    score: float = Field(description="Similarité cosinus (1 = identique)")


class AskResponse(BaseModel):
    question: str
    answer: str
    model: str
    sources: list[Source]

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
@app.post("/search", response_model=list[Source], tags=["histoire"])
def search(req: AskRequest):
    """Retourne seulement les passages pertinents (sans appeler le LLM)."""
    return retrieve(req.question, req.top_k)

# ============================================================
# Génération d'une réponse à partir des passages retrouvés
# ============================================================
@app.post("/ask", response_model=AskResponse, tags=["histoire"])
def ask(req: AskRequest):
    """Répond à une question sur l'histoire à partir des passages retrouvés."""
    conversation_id = database.create_conversation()
    database.add_message(conversation_id, "user", req.question)
    sources = retrieve(req.question, req.top_k)
    context = "\n\n---\n\n".join(f"[{s.chapter}]\n{s.text}" for s in sources)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Extraits de l'histoire :\n\n{context}\n\nQuestion : {req.question}"},
    ]
    try:
        answer = rag.chat(messages)
        database.add_message(conversation_id, "assistant", answer)
    except Exception as exc:  # Ollama injoignable, modèle absent...
        raise HTTPException(502, f"Erreur Ollama : {exc}") from exc
    return AskResponse(question=req.question, answer=answer.strip(), model=config.LLM_MODEL, sources=sources)

# ============================================================
# Lancement de l'API
# ===========================================================
if __name__ == "__main__":
    import uvicorn
    
# Lance le serveur FastAPI sur le port 8000
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
