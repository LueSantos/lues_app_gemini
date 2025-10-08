# memory.py
import os
from typing import List, Dict, Optional
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"

def _get_embeddings():
    """Inicializa embeddings do Gemini (usa GOOGLE_API_KEY do ambiente)."""
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-004")
    return GoogleGenerativeAIEmbeddings(model=embedding_model_name)

def init_or_load_index() -> FAISS:
    """Carrega o índice FAISS se existir; caso contrário, cria um vazio."""
    embeddings = _get_embeddings()
    if os.path.isdir(INDEX_DIR) and os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        try:
            db = FAISS.load_local(
                INDEX_DIR,
                embeddings,
                allow_dangerous_deserialization=True
            )
            return db
        except Exception as e:
            print(f"[Memória] Falha ao carregar índice existente: {e}")
    # retorna índice vazio
    return FAISS(embedding_function=embeddings, index=None, docstore={}, index_to_docstore_id={})

def add_interaction_to_memory(
    question: str,
    answer: str,
    summary: str = "",
    artifacts: Optional[List[str]] = None,
    extra_meta: Dict = None
):
    """Adiciona uma interação (pergunta/resposta) à memória FAISS."""
    artifacts = artifacts or []
    extra_meta = extra_meta or {}

    db = init_or_load_index()

    content = f"Q: {question}\nA: {answer}\nSummary: {summary}\nArtifacts: {','.join(artifacts)}"
    doc = Document(
        page_content=content,
        metadata={
            "question": question,
            "answer": answer,
            "summary": summary,
            "artifacts": artifacts,
            **extra_meta,
        },
    )

    db.add_documents([doc])
    os.makedirs(INDEX_DIR, exist_ok=True)
    db.save_local(INDEX_DIR)

def query_memory(query: str, k: int = 4) -> List[Dict]:
    """Recupera até k documentos mais similares à consulta."""
    db = init_or_load_index()
    try:
        results = db.similarity_search_with_score(query, k=k)
    except Exception:
        return []

    out = []
    for doc, score in results:
        out.append(
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            }
        )
    return out
