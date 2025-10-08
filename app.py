import streamlit as st
import pandas as pd
from agent import ask_agent
from memory import query_memory

# Configuracao da pagina
st.set_page_config(page_title="Agente Autonomo de EDA", layout="wide")

st.title("Agente Autônomo de EDA")
st.markdown("Carregue um CSV e faça perguntas em linguagem natural sobre os dados.")

# Inicializar sessão
if "history" not in st.session_state:
    st.session_state["history"] = []
if "df" not in st.session_state:
    st.session_state["df"] = None

# Upload de CSV
uploaded_file = st.file_uploader("Carregue um arquivo CSV", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(
            uploaded_file,
            sep=",",
            quotechar='"',
            engine="python",
            encoding="utf-8",
        )
    except Exception:
        df = pd.read_csv(
            uploaded_file,
            sep=",",
            quotechar='"',
            engine="python",
            encoding="latin-1",
        )

    st.session_state["df"] = df

    st.subheader("Prévia dos dados")
    st.dataframe(df.head())

    # Caixa de perguntas
    question = st.text_input("Faça uma pergunta sobre o dataset:")

    if st.button("Perguntar") and question:
        response, artifacts = ask_agent(question, df)
        st.session_state["history"].append(
            {"question": question, "response": response, "artifacts": artifacts}
        )

# Histórico de interações
if st.session_state["history"]:
    st.subheader("Histórico de análises")
    for i, entry in enumerate(st.session_state["history"], 1):
        st.markdown(f"**Pergunta {i}:** {entry['question']}")
        st.markdown(f"**Resposta:** {entry['response']}")
        if entry["artifacts"]:
            for art in entry["artifacts"]:
                st.image(art, caption=f"Artefato gerado ({art})", width="stretch")
        st.markdown("---")

# 🔍 Painel lateral para debug de memória
st.sidebar.header("Memória do Agente (Debug)")
query = st.sidebar.text_input("Consultar memória (digite uma palavra-chave):")

if query:
    results = query_memory(query, k=5)
    if not results:
        st.sidebar.write("Nenhuma memória encontrada.")
    else:
        for r in results:
            st.sidebar.markdown(f"**Score:** {r['score']:.3f}")
            st.sidebar.write(r["metadata"].get("summary") or r["content"][:200])
            st.sidebar.write("---")
