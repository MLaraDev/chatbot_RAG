import os
import tempfile
from datetime import datetime, timedelta

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA

# =====================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================

st.set_page_config(
    page_title="Assistente Técnico Industrial",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# INTERFACE: PURE DARK & NEON YELLOW GLOW
# =====================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        background-color: #0A0B0D !important;
        background-image: radial-gradient(circle at 50% -20%, rgba(234, 179, 8, 0.09) 0%, transparent 55%);
        font-family: 'Inter', sans-serif;
        color: #E2E8F0;
    }

    [data-testid="stSidebar"] {
        background: rgba(13, 15, 18, 0.85) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.04) !important;
    }

    /* Brand do topo da sidebar */
    .sb-brand {
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 4px 0 18px 0;
    }
    .sb-brand-icon {
        width: 22px;
        height: 22px;
        border-radius: 6px;
        background: #EAB308;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        flex-shrink: 0;
    }
    .sb-brand-name {
        font-size: 13px;
        font-weight: 600;
        color: #FFFFFF;
    }

    .sidebar-section-title {
        font-size: 10px !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
        color: #EAB308 !important;
        margin: 18px 0 8px 0 !important;
        opacity: 0.9;
    }

    /* Itens de histórico clicáveis (botões do streamlit) na sidebar */
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(20, 22, 26, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.04) !important;
        color: #D4D4D8 !important;
        border-radius: 10px !important;
        padding: 8px 10px !important;
        font-size: 12px !important;
        font-weight: 400 !important;
        text-align: left !important;
        justify-content: flex-start !important;
        width: 100% !important;
        margin-bottom: 6px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        border-color: rgba(234, 179, 8, 0.35) !important;
        color: #FFFFFF !important;
        background: rgba(234, 179, 8, 0.06) !important;
    }
    [data-testid="stSidebar"] .stButton > button:focus:not(:active) {
        color: #FFFFFF !important;
    }

    /* Caixa de upgrade no rodapé da sidebar */
    .sb-upgrade {
        background: linear-gradient(135deg, rgba(234,179,8,0.12), rgba(234,179,8,0.04));
        border: 1px solid rgba(234, 179, 8, 0.18);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 11px;
        color: #FAEEDA;
        line-height: 1.5;
        margin-top: 18px;
    }

    /* Header do painel principal (logo + avatar) */
    .main-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 4px 0 18px 0;
    }
    .main-header-brand {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .main-header-icon {
        width: 24px;
        height: 24px;
        border-radius: 7px;
        background: #EAB308;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 13px;
    }
    .main-header-name {
        font-size: 14px;
        font-weight: 600;
        color: #FFFFFF;
    }
    /* Bolha central */
    .glow-orb {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        margin: 8px auto 18px auto;
        background: radial-gradient(circle at 35% 30%, rgba(234,179,8,0.55), rgba(234,179,8,0.05) 70%);
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    /* Cabeçalho "Welcome back" */
    .grok-header {
        text-align: center;
        padding: 10px 0 22px 0;
    }
    .grok-header h1 {
        color: #FFFFFF !important;
        font-size: 34px !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em;
        margin-bottom: 8px !important;
    }
    .grok-header h1 span {
        color: #EAB308;
    }
    .grok-header p {
        color: #71717A !important;
        font-size: 13px !important;
        max-width: 480px;
        margin: 0 auto !important;
    }

    /* Grid de cards inferiores */
    .grok-grid-container {
        display: flex;
        gap: 16px;
        margin-top: 8px;
        margin-bottom: 26px;
    }
    .grok-feature-card {
        flex: 1;
        background: rgba(18, 20, 24, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.03);
        border-radius: 14px;
        padding: 20px;
        text-align: left;
        transition: all 0.2s ease;
    }
    .grok-feature-card:hover {
        border-color: rgba(234, 179, 8, 0.2);
        background: rgba(22, 26, 32, 0.6);
    }
    .grok-feature-card .gf-icon {
        font-size: 18px;
        color: #EAB308;
        margin-bottom: 10px;
        display: block;
    }
    .grok-feature-card h3 {
        font-size: 13px !important;
        color: #FFFFFF !important;
        margin: 0 0 4px 0 !important;
        font-weight: 600 !important;
    }
    .grok-feature-card p {
        font-size: 11px !important;
        color: #52525B !important;
        margin: 0 !important;
        line-height: 1.5;
    }

    /* Painel de início de conversa (upload + resumo) */
    .start-panel {
        background: rgba(18, 20, 24, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 22px;
    }
    .start-panel-label {
        font-size: 11px;
        font-weight: 600;
        color: #A1A1AA;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Botões gerais (fora da sidebar) como pílulas escuras */
    .stButton > button {
        background: rgba(20, 22, 26, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        color: #D4D4D8 !important;
        border-radius: 24px !important;
        padding: 8px 18px !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        border-color: #EAB308 !important;
        color: #FFFFFF !important;
        background: rgba(234, 179, 8, 0.07) !important;
    }

    /* Uploader compacto */
    [data-testid="stFileUploader"] {
        background: rgba(10, 11, 13, 0.4);
        border-radius: 12px;
        padding: 4px;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: rgba(10, 11, 13, 0.5) !important;
        border: 1px dashed rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }

    /* Caixa das fontes de resposta */
    .source-card {
        background: rgba(18, 20, 24, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.02);
        border-left: 2px solid #EAB308;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
    }

    /* Barra de chat */
    [data-testid="stChatInput"] {
        background-color: rgba(18, 20, 24, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 14px !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.01) !important;
        border: 1px solid rgba(255, 255, 255, 0.03) !important;
        border-radius: 10px;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(234, 179, 8, 0.08);
        border: 1px solid rgba(234, 179, 8, 0.2);
        color: #EAB308;
        font-size: 11px;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 20px;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# CREDENCIAIS DO BANCO COGNITIVO
# =====================================================

nvidia_api_key = os.getenv("NVIDIA_API_KEY")
try:
    if "NVIDIA_API_KEY" in st.secrets:
        nvidia_api_key = st.secrets["NVIDIA_API_KEY"]
except Exception:
    pass

MODELO_LLM = "meta/llama-3.1-8b-instruct"
MODELO_EMBEDDING = "nvidia/nv-embedqa-e5-v5"

SUGESTOES = [
    "🔍 Busca Profunda",
    "⚡ Análise Rápida",
    "📋 Check de Segurança",
    "🛠️ Códigos de Falha"
]

# =====================================================
# RAG CORE PIPELINE
# =====================================================

@st.cache_resource(show_spinner=False)
def inicializar_rag(pdf_bytes, pdf_name):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        paginas = loader.load()
        num_paginas = len(paginas)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        docs = text_splitter.split_documents(paginas)
        num_chunks = len(docs)

        embeddings = NVIDIAEmbeddings(
            model=MODELO_EMBEDDING,
            nvidia_api_key=nvidia_api_key,
            model_type="passage"
        )

        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        return retriever, num_paginas, num_chunks
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# =====================================================
# ESTADO DA SESSÃO
# =====================================================

if "conversas" not in st.session_state:
    # Cada conversa: {id, titulo, timestamp, messages: [...]}
    st.session_state.conversas = []
if "conversa_atual_id" not in st.session_state:
    st.session_state.conversa_atual_id = None
if "resumo" not in st.session_state:
    st.session_state.resumo = None
if "sugestao_click" not in st.session_state:
    st.session_state.sugestao_click = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "confirmar_limpeza" not in st.session_state:
    st.session_state.confirmar_limpeza = False


def nova_conversa_id():
    return f"conv_{len(st.session_state.conversas)}_{datetime.now().timestamp()}"


def obter_mensagens_atuais():
    for c in st.session_state.conversas:
        if c["id"] == st.session_state.conversa_atual_id:
            return c["messages"]
    return []


def garantir_conversa_ativa(titulo_inicial):
    """Cria uma conversa nova se não houver uma ativa, e devolve a lista de mensagens."""
    if st.session_state.conversa_atual_id is None:
        nova = {
            "id": nova_conversa_id(),
            "titulo": titulo_inicial[:40],
            "timestamp": datetime.now(),
            "messages": []
        }
        st.session_state.conversas.insert(0, nova)
        st.session_state.conversa_atual_id = nova["id"]
    return obter_mensagens_atuais()


def agrupar_conversas_por_data():
    agora = datetime.now()
    grupos = {"Hoje": [], "Ontem": [], "7 dias atrás": [], "30 dias atrás": [], "Mais antigas": []}
    for c in st.session_state.conversas:
        delta = (agora.date() - c["timestamp"].date()).days
        if delta == 0:
            grupos["Hoje"].append(c)
        elif delta == 1:
            grupos["Ontem"].append(c)
        elif delta <= 7:
            grupos["7 dias atrás"].append(c)
        elif delta <= 30:
            grupos["30 dias atrás"].append(c)
        else:
            grupos["Mais antigas"].append(c)
    return grupos

# =====================================================
# MENU LATERAL — HISTÓRICO DE CONVERSAS
# =====================================================

with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <div class="sb-brand-icon">⚡</div>
        <span class="sb-brand-name">Assistente Técnico</span>
    </div>
    """, unsafe_allow_html=True)

    if st.button("＋ Nova conversa", use_container_width=True):
        st.session_state.conversa_atual_id = None
        st.session_state.resumo = None
        st.rerun()

    st.text_input("Buscar conversa", label_visibility="collapsed", placeholder="🔍 Buscar conversa", key="busca_historico")

    grupos = agrupar_conversas_por_data()
    termo_busca = st.session_state.get("busca_historico", "").strip().lower()

    algum_resultado = False
    for nome_grupo, lista in grupos.items():
        if not lista:
            continue
        lista_filtrada = [c for c in lista if termo_busca in c["titulo"].lower()] if termo_busca else lista
        if not lista_filtrada:
            continue
        algum_resultado = True
        st.markdown(f'<p class="sidebar-section-title">{nome_grupo}</p>', unsafe_allow_html=True)
        for c in lista_filtrada:
            rotulo = c["titulo"] if c["titulo"] else "Nova conversa"
            if st.button(rotulo, key=f"hist_{c['id']}", use_container_width=True):
                st.session_state.conversa_atual_id = c["id"]
                st.session_state.resumo = None
                st.rerun()

    if not st.session_state.conversas:
        st.caption("Suas conversas aparecerão aqui.")
    elif not algum_resultado and termo_busca:
        st.caption("Nenhuma conversa encontrada.")

    if st.session_state.conversas:
        st.markdown('<p class="sidebar-section-title">&nbsp;</p>', unsafe_allow_html=True)

        if not st.session_state.confirmar_limpeza:
            if st.button("🗑️ Limpar histórico", use_container_width=True, key="btn_limpar_historico"):
                st.session_state.confirmar_limpeza = True
                st.rerun()
        else:
            st.caption("Apagar todas as conversas? Essa ação não pode ser desfeita.")
            col_conf, col_canc = st.columns(2)
            with col_conf:
                if st.button("Confirmar", use_container_width=True, key="btn_confirmar_limpeza"):
                    st.session_state.conversas = []
                    st.session_state.conversa_atual_id = None
                    st.session_state.resumo = None
                    st.session_state.confirmar_limpeza = False
                    st.rerun()
            with col_canc:
                if st.button("Cancelar", use_container_width=True, key="btn_cancelar_limpeza"):
                    st.session_state.confirmar_limpeza = False
                    st.rerun()

    st.markdown("""
    <div class="sb-upgrade">
        ⚡ Assine para liberar análises ilimitadas e novos recursos
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# HEADER DO PAINEL PRINCIPAL
# =====================================================

st.markdown("""
<div class="main-header">
    <div class="main-header-brand">
        <div class="main-header-icon">⚡</div>
        <span class="main-header-name">Assistente Técnico Industrial</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not nvidia_api_key:
    st.error("Chave NVIDIA_API_KEY ausente.")
    st.stop()

# =====================================================
# BOLHA + TÍTULO DE BOAS-VINDAS
# =====================================================

mensagens_atuais = obter_mensagens_atuais()

if not mensagens_atuais:
    st.markdown("""
    <div class="glow-orb"></div>
    <div class="grok-header">
        <h1>Bem-vindo! <span>Como posso ajudar?</span></h1>
        <p>Envie um manual técnico em PDF ou selecione um modo abaixo para começar.</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# PAINEL DE INÍCIO: UPLOAD + RESUMO TÉCNICO (dentro do chat)
# =====================================================

if not mensagens_atuais:
    st.markdown('<div class="start-panel">', unsafe_allow_html=True)
    st.markdown('<p class="start-panel-label">📎 Anexar manual técnico (PDF)</p>', unsafe_allow_html=True)

    col_upload, col_resumo = st.columns([3, 1])
    with col_upload:
        pdf_arquivado = st.file_uploader(
            "Carregar Manual PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key="uploader_principal"
        )
        if pdf_arquivado is not None:
            st.session_state.pdf_bytes = pdf_arquivado.read()
            st.session_state.pdf_name = pdf_arquivado.name

    with col_resumo:
        gerar_resumo = st.button(
            "📄 Resumo Técnico",
            use_container_width=True,
            disabled=st.session_state.pdf_bytes is None
        )

    if st.session_state.pdf_name:
        st.markdown(
            f'<span class="badge-pill">✓ {st.session_state.pdf_name}</span>',
            unsafe_allow_html=True
        )

    st.markdown('</div>', unsafe_allow_html=True)
else:
    gerar_resumo = False

if st.session_state.pdf_bytes is None:
    st.markdown("""
    <div style="background: rgba(18, 20, 24, 0.4); border: 1px dashed rgba(255,255,255,0.05); border-radius:14px; padding: 36px; text-align:center; color:#52525B; font-size: 13px; margin-bottom: 24px;">
        Insira o manual técnico em formato PDF acima para ativar o painel de análise.
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# CARDS DE CAPACIDADES (contexto industrial)
# =====================================================

if not mensagens_atuais:
    st.markdown("""
    <div class="grok-grid-container">
        <div class="grok-feature-card">
            <i class="gf-icon">⚠️</i>
            <h3>Códigos de Falha</h3>
            <p>Diagnóstico rápido com base nos códigos e tabelas do manual carregado.</p>
        </div>
        <div class="grok-feature-card">
            <i class="gf-icon">🛡️</i>
            <h3>Check de Segurança</h3>
            <p>Procedimentos, EPIs e travas de segurança exigidos por etapa do processo.</p>
        </div>
        <div class="grok-feature-card">
            <i class="gf-icon">📊</i>
            <h3>Resumo Técnico</h3>
            <p>Síntese estruturada do documento, pronta para consulta rápida em campo.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if st.session_state.pdf_bytes is None:
    st.stop()

# =====================================================
# INICIALIZAÇÃO DO RAG (uma vez por PDF)
# =====================================================

with st.spinner("Mapeando manual..."):
    try:
        retriever, num_paginas, num_chunks = inicializar_rag(
            st.session_state.pdf_bytes, st.session_state.pdf_name
        )
    except Exception as e:
        st.error(f"Erro na matriz: {e}")
        st.stop()

try:
    llm = ChatNVIDIA(model=MODELO_LLM, nvidia_api_key=nvidia_api_key, temperature=0.1, max_tokens=1024)
except Exception as e:
    st.error(f"Erro LLM: {e}")
    st.stop()

template_prompt = "Você é o Assistente Técnico RAG. Responda em PT-BR direto ao ponto com base estritamente no manual:\nContexto:\n{context}\n\nPergunta:\n{question}"
prompt_template = ChatPromptTemplate.from_template(template_prompt)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_com_fontes(pergunta: str):
    docs_relevantes = retriever.invoke(pergunta)
    contexto = format_docs(docs_relevantes)
    chain = prompt_template | llm | StrOutputParser()
    resposta = chain.invoke({"context": contexto, "question": pergunta})
    return resposta, docs_relevantes


if gerar_resumo:
    with st.spinner("Sintetizando..."):
        docs_resumo = retriever.invoke("Resumo geral estruturado.")
        contexto_resumo = format_docs(docs_resumo)
        prompt_resumo = ChatPromptTemplate.from_template("Gere um sumário técnico conciso em PT-BR: {context}")
        st.session_state.resumo = (prompt_resumo | llm | StrOutputParser()).invoke({"context": contexto_resumo})

if st.session_state.resumo:
    with st.expander("📄 Sumário Executivo do Documento", expanded=True):
        st.markdown(st.session_state.resumo)

# =====================================================
# SUGESTÕES (apenas quando não há conversa ativa)
# =====================================================

if not mensagens_atuais:
    cols = st.columns(4)
    for i, sugestao in enumerate(SUGESTOES):
        with cols[i]:
            if st.button(sugestao, key=f"sug_{i}", use_container_width=True):
                st.session_state.sugestao_click = sugestao
                st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================
# RENDERIZAÇÃO DA CONVERSA ATUAL
# =====================================================

mensagens_atuais = obter_mensagens_atuais()

for message in mensagens_atuais:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("fontes"):
            with st.expander("🔍 Rastreabilidade das Fontes"):
                for idx, doc in enumerate(message["fontes"], 1):
                    page = doc.metadata.get("page", 0) + 1
                    st.markdown(f"""
                    <div class="source-card">
                        <span style="color:#EAB308; font-size:11px; font-weight:600;">Seção {idx} — Página {page}</span>
                        <p style="font-family:'JetBrains Mono', monospace; font-size:11px; margin-top:4px; color:#71717A;">{doc.page_content[:200]}...</p>
                    </div>
                    """, unsafe_allow_html=True)

# =====================================================
# ENTRADA DO PROMPT
# =====================================================

pergunta_usuario = st.session_state.sugestao_click if st.session_state.sugestao_click else st.chat_input("Fale conosco sobre o que deseja analisar...")

if pergunta_usuario:
    st.session_state.sugestao_click = None

    mensagens = garantir_conversa_ativa(pergunta_usuario)
    mensagens.append({"role": "user", "content": pergunta_usuario})

    # Atualiza o título da conversa se ainda for a primeira mensagem
    for c in st.session_state.conversas:
        if c["id"] == st.session_state.conversa_atual_id and len(c["messages"]) == 1:
            c["titulo"] = pergunta_usuario[:40]

    with st.spinner("Processando requisição..."):
        try:
            resposta, docs_fonte = rag_com_fontes(pergunta_usuario)
            mensagens.append({
                "role": "assistant",
                "content": resposta,
                "fontes": docs_fonte
            })
        except Exception as e:
            mensagens.append({
                "role": "assistant",
                "content": f"Erro: {str(e)}",
                "fontes": []
            })
    st.rerun()