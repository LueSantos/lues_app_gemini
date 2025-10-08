import os
from typing import Tuple, List
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from tools import (
    plot_histogram,
    plot_boxplot,
    plot_correlation_heatmap,
    cluster_kmeans,
    run_python_code,
)
from memory import add_interaction_to_memory, query_memory

# Configuração do modelo
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=LLM_MODEL, temperature=0)

# Prompt para decidir qual ferramenta usar
DECISION_PROMPT = """
Você é um agente de EDA (Análise Exploratória de Dados) que decide qual ferramenta executar para responder à pergunta do usuário.
Contexto da memória:
{memory}

Pergunta: {question}

Com base na pergunta, escolha UMA das seguintes ferramentas:
- Se a pergunta solicitar a visualização da distribuição de uma única variável numérica, use: plot_histogram
- Se a pergunta solicitar a identificação de outliers em uma única variável numérica, use: plot_boxplot
- Se a pergunta solicitar a análise da relação entre múltiplas variáveis numéricas, use: plot_correlation_heatmap
- Se a pergunta solicitar o agrupamento de dados com base em variáveis numéricas, use: cluster_kmeans
- Se a pergunta exigir qualquer outro tipo de análise, cálculo estatístico, resumo ou manipulação de dados, use: python_repl

Responda APENAS com o nome da ferramenta escolhida.
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
    Você é um analista de dados experiente.
    Com base na pergunta original do usuário e na saída do código Python, gere uma explicação concisa e clara em português para um público não técnico.

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
            Você é um analista de dados experiente, especializado em responder perguntas complexas sobre DataFrames do pandas.
            Com base na pergunta do usuário, gere um código Python conciso e eficiente para responder à pergunta.

            Pergunta: {question}

            Informações importantes sobre o DataFrame:
            - O DataFrame está armazenado na variável `df`.
            - As colunas disponíveis no DataFrame são: {list(df.columns)}

            Regras importantes:
            - Use pandas diretamente para cálculos estatísticos (ex: df.mean(), df.std(), df.var()).
            - Não use índices inexistentes em df.describe().
            - Responda SOMENTE com código Python válido.
            - Não escreva texto, nem comentários, nem explicações.
            - O código deve terminar imprimindo a resposta
              ou atribuindo a uma variável chamada result.
            
             Restrições importantes:
            - Forneça APENAS o código Python. Não inclua explicações, comentários ou qualquer outro texto.
            - O código deve ser completo e auto-suficiente, incluindo todas as importações necessárias.
            - O código deve imprimir o resultado final ou atribuí-lo a uma variável chamada `result`.
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
