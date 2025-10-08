import os
from typing import Tuple, List
import pandas as pd



from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import ChatPromptTemplate
clear 
from tools import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap,
    cluster_kmeans,
    run_python_code,
)
from memory import add_interaction_to_memory, query_memory

# Configuração do modelo
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-pro-latest")  # Ou outro modelo do Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Certifique-se de ter a chave de API no seu ambiente
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0)

# Prompt para decidir qual ferramenta usar
DECISION_PROMPT = """
Você é um agente de EDA que decide qual ferramenta executar.
Contexto da memória:
{memory}

Pergunta: {question}

Escolha uma das opções:
- plot_histogram (se pedir histograma)
- plot_boxplot (se pedir boxplot)
- plot_correlation_heatmap (se pedir correlação)
- cluster_kmeans (se pedir agrupamento/clustering)
- python_repl (para QUALQUER outra análise, cálculo, resumo estatístico, intervalos etc.)

Responda apenas com o nome da ferramenta.
"""

def _get_memory_context(question: str, k: int = 3) -> str:
    mems = query_memory(question, k=k)
    if not mems:
        return "Nenhuma memória relevante encontrada."
    return "\n".join([f"- {m['metadata'].get('summary')}" for m in mems])

def _sanitize_code(code: str) -> str:
    """Remove wrappers markdown (```python ... ```) e retorna apenas o código."""
    code = code.strip()
    if code.startswith("```"):
        lines = [ln for ln in code.splitlines() if not ln.strip().startswith("```")]
        code = "\n".join(lines)
    return code.strip()

def _explain_output(question: str, raw_output: str) -> str:
    """Usa a LLM para gerar uma explicação em português baseada na saída do código."""
    explain_prompt = f"""
    Você é um analista de dados.
    Pergunta original: {question}

    Aqui está a saída bruta de um cálculo em pandas:
    {raw_output}

    Escreva uma explicação em português clara, objetiva e concisa
    descrevendo os resultados para um leitor leigo em estatística.
    """
    res = llm.invoke(explain_prompt)
    return res.content.strip() if hasattr(res, "content") else str(res).strip()

def ask_agent(question: str, df: pd.DataFrame) -> Tuple[str, List[str]]:
    artifacts: List[str] = []
    memory_context = _get_memory_context(question)

    # 1. Decisão de ferramenta
    prompt = ChatPromptTemplate.from_template(DECISION_PROMPT)
    chain = prompt | llm
    decision = chain.invoke({"memory": memory_context, "question": question})
    decision_text = decision.content.strip().lower() if hasattr(decision, "content") else str(decision).strip().lower()

    response_text = ""

    try:
        if "plot_histogram" in decision_text:
            col = df.select_dtypes(include="number").columns[0]
            path = plot_histogram(df, col)
            artifacts.append(path)
            response_text = f"Histograma gerado para {col}"

        elif "plot_boxplot" in decision_text:
            col = df.select_dtypes(include="number").columns[0]
            path = plot_boxplot(df, col)
            artifacts.append(path)
            response_text = f"Boxplot gerado para {col}"

        elif "plot_correlation_heatmap" in decision_text:
            path = plot_correlation_heatmap(df)
            artifacts.append(path)
            response_text = "Mapa de correlação gerado."

        elif "cluster_kmeans" in decision_text:
            cols = df.select_dtypes(include="number").columns[:2]
            path = cluster_kmeans(df, cols)
            artifacts.append(path)
            response_text = f"KMeans executado em {cols.tolist()}"

        elif "python_repl" in decision_text or "none" in decision_text:
            python_prompt = f"""
            Você é um analista de dados.
            Pergunta: {question}

            Você tem acesso ao DataFrame `df` (pandas).
            Colunas disponíveis: {list(df.columns)}

            Regras importantes:
            - Use pandas diretamente para cálculos estatísticos (ex: df.mean(), df.std(), df.var()).
            - Não use índices inexistentes em df.describe().
            - Responda SOMENTE com código Python válido.
            - Não escreva texto, nem comentários, nem explicações.
            - O código deve terminar imprimindo a resposta
              ou atribuindo a uma variável chamada result.
            """
            res = llm.invoke(python_prompt)
            code = res.content.strip() if hasattr(res, "content") else str(res).strip()
            code = _sanitize_code(code)

            raw_output = run_python_code(df, code)
            response_text = _explain_output(question, raw_output)

        else:
            response_text = "Não sei qual análise aplicar."

    except Exception as e:
        response_text = f"Erro ao executar a ferramenta: {e}"

    # 2. Salvar memória da interação
    summary = response_text if len(response_text) < 500 else response_text[:497] + "..."
    try:
        add_interaction_to_memory(
            question=question,
            answer=response_text,
            summary=summary,
            artifacts=artifacts,
            extra_meta={"cols": ','.join(map(str, df.columns[:5]))},
        )
    except Exception:
        pass

    return response_text, artifacts
