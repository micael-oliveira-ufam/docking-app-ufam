import streamlit as st
import os
import subprocess
import sys
import requests
import re
import glob
import time
import io
import zipfile
from datetime import datetime
import numpy as np
import multiprocessing

# ==========================================
# CONFIGURAÇÃO DA PÁGINA E TEMA MODERNO
# ==========================================
st.set_page_config(
    page_title="Laboratório Virtual de Docking | FCF/UFAM", 
    layout="wide", 
    initial_sidebar_state="expanded",
    page_icon="🧬"
)

# Injeção de CSS para UX moderno e fundo branco puro
st.markdown("""
<style>
    /* Estilo Global e Fundo Branco */
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Configuração de Fontes e Espaçamento */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        color: #202124;
    }
    
    /* Títulos e Subtítulos Modernos */
    h1 {
        color: #1E88E5;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: #3C4043;
        font-weight: 600;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    
    /* Estilização da Sidebar (Painel Lateral) */
    [data-testid="stSidebar"] {
        background-color: #F8F9FA;
        border-right: 1px solid #E0E0E0;
    }
    [data-testid="stSidebar"] h3 {
        color: #1E88E5;
        font-size: 1.1rem;
        margin-top: 1.5rem;
    }
    [data-testid="stSidebar"] .stMarkdown {
        font-size: 0.9rem;
        color: #5F6368;
    }
    
    /* Modernização de Cards (Containers) */
    div[data-testid="stVerticalBlock"] > div:has(div[class*="stAlert"]),
    div[data-testid="stVerticalBlock"] > div:has(div[class*="stBlock"]) {
        # background-color: #FFFFFF;
        # border-radius: 12px;
        # padding: 1.5rem;
        # border: 1px solid #E0E0E0;
        # box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        # margin-bottom: 1rem;
    }
    
    /* Botões Modernos */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        text-transform: none;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton > button[kind="primary"] {
        background-color: #1E88E5;
        border: none;
    }
    .stButton > button[kind="secondary"] {
        background-color: #FFFFFF;
        color: #1E88E5;
        border: 1px solid #E0E0E0;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #F8F9FA;
        border: 1px solid #1E88E5;
    }

    /* Inputs Modernos */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > div {
        border-radius: 8px;
        border: 1px solid #E0E0E0;
        background-color: #FFFFFF;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #1E88E5;
        box-shadow: 0 0 0 1px #1E88E5;
    }
    
    /* Estilo das Abas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #F8F9FA;
        padding: 6px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre;
        background-color: transparent;
        border-radius: 8px;
        color: #5F6368;
        font-weight: 500;
        border: none;
        padding: 0px 16px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1E88E5;
        background-color: rgba(30, 136, 229, 0.05);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        color: #1E88E5;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Gráficos e Tabelas */
    [data-testid="stDataFrame"] {
        border: 1px solid #E0E0E0;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Tenta importar as bibliotecas principais
try:
    import py3Dmol
    from streamlit.components.v1 import html
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    import pubchempy as pcp
    from rdkit import Chem
    from rdkit.Chem import Draw
    from rdkit.Chem import AllChem
    import pandas as pd
    from Bio.PDB import PDBParser
    import plotly.express as px
    LIBS_INSTALADAS = True
except ImportError:
    LIBS_INSTALADAS = False

# ==========================================
# FUNÇÕES DE CONTROLE DE SERVIDOR (TRAVA GLOBAL)
# ==========================================
LOCK_FILE = "vina_execution.lock"

def is_server_busy():
    """Verifica se o servidor está rodando Vina para outro usuário."""
    if os.path.exists(LOCK_FILE):
        file_age = time.time() - os.path.getmtime(LOCK_FILE)
        if file_age > 1800: # Proteção Anti-Zumbi: 30 minutos
            os.remove(LOCK_FILE)
            return False
        return True
    return False

def lock_server():
    """Bloqueia o servidor para outros usuários."""
    with open(LOCK_FILE, 'w') as f:
        f.write(f"Running Vina. Locked at {datetime.now()}")

def unlock_server():
    """Libera o servidor para o próximo usuário."""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)

# ==========================================
# FUNÇÕES AUXILIARES DE QUIMIOINFORMÁTICA
# ==========================================
def get_ligands_from_pdb(pdb_file):
    """Extrai os nomes dos fármacos co-cristalizados no PDB."""
    ligands = set()
    filtros = ["HOH", "WAT", "DOD", "NA", "CL", "MG", "K", "SO4", "PO4", "EDO", "GOL", "FMT", "ACT"]
    if os.path.exists(pdb_file):
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("HETATM"):
                    res_name = line[17:20].strip()
                    if res_name not in filtros:
                        ligands.add(res_name)
    return list(ligands)

def extract_ligand_from_pdb(pdb_file, res_name, output_file):
    """Copia as coordenadas exatas do ligante do cristal."""
    with open(pdb_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.startswith("HETATM") and line[17:20].strip() == res_name:
                f_out.write(line)
        f_out.write("END\n")

def sanitize_filename(name):
    """Evita que caracteres químicos quebrem o nome do arquivo."""
    sanitized = re.sub(r'[\\/*?:"<>| ,()\[\]{}]', "_", str(name))
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized.lower()

def get_vina_affinity(file_path):
    """Lê a afinidade termodinâmica da melhor pose (MODEL 1) diretamente do arquivo PDBQT."""
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    match = re.search(r'REMARK VINA RESULT:\s+([-\d\.]+)', line)
                    if match:
                        return float(match.group(1))
    return np.nan

# ==========================================
# BARRA LATERAL (AUTORIA E LOGO)
# ==========================================
with st.sidebar:
    # Centraliza a imagem na sidebar
    col_logo_1, col_logo_2, col_logo_3 = st.columns([1, 4, 1])
    with col_logo_2:
        try:
            st.image("fundacao-universidade-do-amazonas.png", use_container_width=True)
        except Exception:
            st.warning("Logo da UFAM não encontrado.")
    
    st.markdown("---")
    with st.container(border=True): # Card de autoria
        st.markdown("### 👨‍🔬 Autoria do Projeto")
        st.markdown("**Micael Davi Lima de Oliveira**")
        st.markdown("*Iniciação Científica (PIBIC)*")
        st.markdown("---")
        st.markdown("**Faculdade de Ciências Farmacêuticas**")
        st.markdown("Universidade Federal do Amazonas (UFAM)")
    
    st.markdown("---")
    st.caption("Desenvolvido para ensino e pesquisa em Química Medicinal Computacional.")

# Cabeçalho Principal Estilizado
st.markdown("""
<div style="background-color: #FFFFFF; padding: 1.5rem 0rem; border-bottom: 1px solid #E0E0E0; margin-bottom: 2rem;">
    <h1 style="margin: 0; font-size: 2.2rem;">🧬 Laboratório Virtual: Docking Molecular e Triagem</h1>
    <p style="color: #5F6368; font-size: 1.1rem; margin-top: 0.5rem; margin-bottom: 0;">Plataforma acadêmica para ensino de <b>Química Medicinal Computacional</b> e <b>Planejamento de Fármacos</b>.</p>
</div>
""", unsafe_allow_html=True)

# Inicialização das variáveis de memória do Streamlit
for key, default in [
    ('cx', 0.0), ('cy', 0.0), ('cz', 0.0),
    ('sx', 20.0), ('sy', 20.0), ('sz', 20.0),
    ('smiles', ""), ('nome_ligante_salvar', "ligante"),
    ('rec_pdb_final', "receptor_prep.pdb"), ('rec_final', "receptor.pdbqt"),
    ('lig_final', "ligante.pdbqt"), ('original_pdb', "2XV7.pdb"),
    ('redocking_mode', False), ('extracted_lig_pdb', ""),
    ('vs_mode', False), ('vs_results_dir', ""),
    ('vina_log_output', "")
]:
    if key not in st.session_state: st.session_state[key] = default

# Organização das Abas com Títulos Amigáveis
tab_ambiente, tab_receptor, tab_ligante, tab_gridbox, tab_vina_conf, tab_executar, tab_visualizar, tab_referencias = st.tabs([
    "🛠️ 1. Ambiente", "🧬 2. Receptor", "💊 3. Ligante", "📦 4. Grid Box", "⚙️ 5. Vina Config", "🚀 6. Docking", "👁️ 7. Resultados", "📚 8. Referências"
])

# ==========================================
# ABA 1: Instalação de Dependências
# ==========================================
with tab_ambiente:
    st.header("1. Verificação do Ambiente Computacional")
    
    with st.container(border=True):
        st.subheader("📚 O que é o Docking Molecular?")
        st.markdown("""
        O **Docking Molecular** é uma técnica fundamental no Planejamento de Fármacos Baseado em Estrutura (SBDD). O objetivo computacional é prever a conformação tridimensional preferencial (a **Pose**) de uma molécula pequena (fármaco/ligante) quando ligada a uma macromolécula (receptor/proteína), formando um complexo estável.
        """)

    with st.container(border=True):
        st.subheader("☁️ Status de Nuvem (Streamlit Cloud)")
        st.markdown("Se você estiver rodando este software localmente, certifique-se de instalar as dependências. Na nuvem do Streamlit, as bibliotecas são carregadas via `requirements.txt` e o OpenBabel via `packages.txt`.")
        
        if LIBS_INSTALADAS:
            st.success("✅ **Sistema Operante:** Todas as bibliotecas de quimioinformática e bioinformática foram detectadas. A plataforma está pronta!")
        else:
            st.error("🚨 **Atenção:** Módulos fundamentais ausentes. Verifique os arquivos requirements.txt no repositório.")

# ==========================================
# ABA 2: Preparação do Receptor
# ==========================================
with tab_receptor:
    st.header("2. O Alvo Farmacológico (Receptor)")
    
    with st.container(border=True):
        st.subheader("Buscando Alvo no Protein Data Bank (RCSB PDB)")
        col_pdb1, col_pdb2 = st.columns([1, 2])
        with col_pdb1:
            pdb_id = st.text_input("Código PDB ID:", value="2XV7", help="Insira o código de 4 caracteres do PDB.")
            btn_baixar = st.button("Baixar e Visualizar Alvo", type="primary", use_container_width=True)
            
            if btn_baixar:
                r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb")
                if r.status_code == 200:
                    with open(f"{pdb_id.upper()}.pdb", "w") as f: f.write(r.text)
                    st.session_state.original_pdb = f"{pdb_id.upper()}.pdb"
                    st.success(f"PDB {pdb_id.upper()} salvo com sucesso!")
                else:
                    st.error("Falha ao buscar PDB no RCSB. Verifique o ID.")

        with col_pdb2:
            if os.path.exists(st.session_state.original_pdb):
                st.markdown("**Visualização Nativa do Cristal:**")
                with st.spinner("Carregando estrutura 3D..."):
                    with open(st.session_state.original_pdb, 'r') as f:
                        viewer = py3Dmol.view(width=700, height=400)
                        viewer.addModel(f.read(), "pdb")
                        viewer.setStyle({"cartoon": {'color':'spectrum'}})
                        viewer.zoomTo()
                        html(viewer._make_html(), width=700, height=400)

    st.divider()

    col_prep1, col_prep2 = st.columns(2)
    with col_prep1:
        with st.container(border=True):
            st.subheader("A. Limpeza e Protonação (PDBFixer)")
            st.markdown("Prepara a proteína removendo solvente, heterogêneos e adicionando hidrogênios no pH fisiológico (7.4) conservando a topologia original.")
            btn_fixer = st.button("Executar Limpeza Conservadora", use_container_width=True)
            
            if btn_fixer:
                try:
                    with st.spinner("Limpando estrutura sem distorcer topologia nativa..."):
                        fixer = PDBFixer(filename=st.session_state.original_pdb)
                        fixer.findNonstandardResidues()
                        fixer.replaceNonstandardResidues()
                        fixer.removeHeterogens(False)
                        fixer.addMissingHydrogens(7.4) 
                        out_pdb = f"{pdb_id.upper()}_prep.pdb"
                        PDBFile.writeFile(fixer.topology, fixer.positions, open(out_pdb, 'w'))
                        st.session_state.rec_pdb_final = out_pdb 
                        st.success(f"Receptor isolado e protonado salvo como '{out_pdb}'.")
                except Exception as e: st.error(f"Erro no PDBFixer: {e}")

    with col_prep2:
        with st.container(border=True):
            st.subheader("B. Atribuição de Cargas e Crate PDBQT")
            st.markdown("Converte o PDB para PDBQT adicionando cargas parciais iterativas de Gasteiger-Marsili usando OpenBabel, vital para o cálculo eletrostático do Vina.")
            btn_obabel_rec = st.button("Calcular Cargas e Rígidez (PDBQT)", use_container_width=True)
            
            if btn_obabel_rec:
                try:
                    with st.spinner("Calculando cargas parciais iterativas..."):
                        out_pdbqt = st.session_state.rec_pdb_final.replace(".pdb", ".pdbqt")
                        comando = ["obabel", "-i", "pdb", st.session_state.rec_pdb_final, "-o", "pdbqt", "-O", out_pdbqt, "-xr", "--partialcharge", "gasteiger"]
                        proc = subprocess.run(comando, capture_output=True, text=True)
                        if os.path.exists(out_pdbqt):
                            st.session_state.rec_final = out_pdbqt
                            st.success(f"Matriz de cargas gerada: '{out_pdbqt}'. Alvo pronto para Docking!")
                        else:
                            st.error(f"Erro ao gerar PDBQT: {proc.stderr}")
                except Exception as e: st.error(f"Erro: {e}")

# ==========================================
# ABA 3: Preparação do Ligante
# ==========================================
with tab_ligante:
    st.header("3. Preparação do(s) Fármaco(s)")
    
    with st.container(border=True):
        st.subheader("Selecione a Estratégia de Processamento")
        modo_preparacao = st.radio("Método de entrada do ligante:", [
            "🔬 Molécula Única (SMILES/Nome Comum)", 
            "♻️ Re-Docking (Extrair Fármaco do Cristal PDB)",
            "📝 Triagem Virtual Automática (Lista de SMILES Text/CSV)",
            "🚀 Triagem Virtual Automática (Upload SDF/Mol2/PDB)"
        ], horizontal=True)

    st.divider()

    if "Única" in modo_preparacao:
        st.session_state.redocking_mode = False
        st.session_state.vs_mode = False
        with st.container(border=True):
            col_input, col_2d, col_3d = st.columns([1.2, 1, 1])
            
            with col_input:
                st.subheader("Entrada Química")
                tipo_entrada = st.radio("Formato:", ("Nome Comum", "Código SMILES"))
                entrada_ligante = st.text_input("Valor químico:", placeholder="Ex: Aspirina ou CC(=O)OC1=CC=CC=C1C(=O)O")
                
                btn_topo = st.button("1. Gerar Topologia (2D)", type="secondary", use_container_width=True)
                
                if btn_topo and entrada_ligante:
                    try:
                        with st.spinner("Analisando estrutura..."):
                            smiles_obtido, nome_final = entrada_ligante, "mol_inedita"
                            if "Nome" in tipo_entrada:
                                comps = pcp.get_compounds(entrada_ligante, 'name')
                                if comps:
                                    smiles_obtido = comps[0].isomeric_smiles
                                    nome_final = sanitize_filename(entrada_ligante)
                                    st.success(f"Encontrado: {smiles_obtido}")
                                else: st.error("Molécula não encontrada no PubChem."); smiles_obtido = ""
                            else: nome_final = "smiles_inedito"
                                    
                            if smiles_obtido:
                                st.session_state.smiles = smiles_obtido
                                st.session_state.nome_ligante_salvar = nome_final
                                mol = Chem.MolFromSmiles(smiles_obtido)
                                if mol: st.session_state.img_2d = Draw.MolToImage(mol, size=(300, 300))
                                else: st.error("SMILES inválido.")
                    except Exception as e: st.error(f"Erro: {e}")

                if st.session_state.smiles:
                    st.info(f"Salvar como: **{st.session_state.nome_ligante_salvar}**")
                    btn_3d = st.button("2. Minimizar (3D) Rápido e Gerar PDBQT", type="primary", use_container_width=True)
                    
                    if btn_3d:
                        sdf_file = f"{st.session_state.nome_ligante_salvar}.sdf"
                        pdbqt_file = f"{st.session_state.nome_ligante_salvar}.pdbqt"
                        with st.spinner("Gerando 3D ultrarrápido (ETKDG + MMFF94)..."):
                            try:
                                mol_3d = Chem.AddHs(Chem.MolFromSmiles(st.session_state.smiles))
                                AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) 
                                AllChem.MMFFOptimizeMolecule(mol_3d) 
                                writer = Chem.SDWriter(sdf_file); writer.write(mol_3d); writer.close()
                                
                                subprocess.run(["obabel", "-isdf", sdf_file, "-opdbqt", "-O", pdbqt_file, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                                
                                if os.path.exists(sdf_file) and os.path.exists(pdbqt_file):
                                    st.session_state.mol2_file_path = sdf_file 
                                    st.session_state.lig_final = pdbqt_file 
                                    st.success(f"Ligante 3D otimizado salvo: '{pdbqt_file}'.")
                            except Exception as e: st.error(f"Falha na geração 3D: {e}")

            with col_2d:
                st.subheader("Topologia 2D")
                if 'img_2d' in st.session_state: st.image(st.session_state.img_2d, use_container_width=True)

            with col_3d:
                st.subheader("Geometria 3D")
                if 'mol2_file_path' in st.session_state and os.path.exists(st.session_state.mol2_file_path):
                    with open(st.session_state.mol2_file_path, 'r') as f:
                        viewer_lig = py3Dmol.view(width=300, height=300)
                        formato = "sdf" if st.session_state.mol2_file_path.endswith(".sdf") else "mol2"
                        viewer_lig.addModel(f.read(), formato)
                        viewer_lig.setStyle({"stick": {'colorscheme': 'greenCarbon'}})
                        viewer_lig.zoomTo()
                        html(viewer_lig._make_html(), width=300, height=300)

    elif "Re-Docking" in modo_preparacao:
        st.session_state.redocking_mode = True; st.session_state.vs_mode = False
        with st.container(border=True):
            st.subheader("Validação de Re-Docking")
            st.markdown("Extrai as coordenadas exatas do ligante nativo do cristal PDB carregado na Etapa 2 para validar o protocolo de docking.")
            if os.path.exists(st.session_state.original_pdb):
                ligantes = get_ligands_from_pdb(st.session_state.original_pdb)
                if ligantes:
                    lig_selecionado = st.selectbox("Fármaco co-cristalizado detectado:", ligantes)
                    btn_ext = st.button("Extrair e Gerar PDBQT Nativo", type="primary", use_container_width=True)
                    if btn_ext:
                        ext_pdb, ext_pdbqt = f"{lig_selecionado}_nativo.pdb", f"{lig_selecionado}_nativo.pdbqt"
                        extract_ligand_from_pdb(st.session_state.original_pdb, lig_selecionado, ext_pdb)
                        subprocess.run(["obabel", "-ipdb", ext_pdb, "-opdbqt", "-O", ext_pdbqt, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                        st.session_state.lig_final = ext_pdbqt
                        st.session_state.extracted_lig_pdb = ext_pdb
                        st.success(f"Coordenadas nativas extraídas: '{ext_pdbqt}'. Pronta para re-docking!")
                else: st.warning("Nenhum ligante orgânico detectado no PDB.")
            else: st.error("Carregue um receptor PDB na Etapa 2 primeiro.")

    elif "SMILES" in modo_preparacao:
        st.session_state.redocking_mode = False; st.session_state.vs_mode = True
        with st.container(border=True):
            st.subheader("Processamento em Lote de SMILES")
            texto_smiles = st.text_area("Insira SMILES (um por linha). Opcional: SMILES,Nome", placeholder="CC(=O)OC1=CC=CC=C1C(=O)O,Aspirina\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C,Cafeina", height=150)
            btn_csv = st.button("Processar Lista de SMILES (Lote)", type="primary", use_container_width=True)
            
            if btn_csv and texto_smiles.strip():
                os.makedirs("Ligantes", exist_ok=True); total_s, total_f = 0, 0
                linhas = [l for l in texto_smiles.split('\n') if l.strip()]
                my_bar = st.progress(0, text="Iniciando processamento em lote...")
                for idx, linha in enumerate(linhas):
                    partes = linha.split(','); smi = partes[0].strip(); nome = sanitize_filename(partes[1].strip()) if len(partes)>1 else f"lig_{idx+1}"
                    if smi:
                        try:
                            mol = Chem.AddHs(Chem.MolFromSmiles(smi))
                            if AllChem.EmbedMolecule(mol, AllChem.ETKDG()) == 0:
                                AllChem.MMFFOptimizeMolecule(mol)
                                sdf_temp = f"Ligantes/{nome}.sdf"; pdbqt_fin = f"Ligantes/{nome}.pdbqt"
                                writer = Chem.SDWriter(sdf_temp); writer.write(mol); writer.close()
                                subprocess.run(["obabel", "-isdf", sdf_temp, "-opdbqt", "-O", pdbqt_fin, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                                if os.path.exists(pdbqt_fin): total_s += 1; os.remove(sdf_temp)
                                else: total_f += 1
                            else: total_f += 1
                        except: total_f += 1
                    my_bar.progress(int(((idx+1)/len(linhas))*100), text=f"Lote: {idx+1}/{len(linhas)} (S: {total_s} | F: {total_f})")
                my_bar.empty()
                if total_s>0: st.success(f"Lote concluído: {total_s} moléculas prontas em 'Ligantes/'."); st.session_state.lig_final = "Múltiplos (Modo Lote)"
                else: st.error("Falha no processamento do lote.")

    else: # Upload Estrutural
        st.session_state.redocking_mode = False; st.session_state.vs_mode = True
        with st.container(border=True):
            st.subheader("Upload de Biblioteca Estrutural")
            uploaded_files = st.file_uploader("Arquivos de biblioteca (.sdf, .mol2, .pdb)", type=['sdf', 'mol2', 'pdb'], accept_multiple_files=True)
            if uploaded_files and st.button("Processar Arquivos Submetidos", type="primary", use_container_width=True):
                os.makedirs("Ligantes_temp", exist_ok=True); os.makedirs("Ligantes", exist_ok=True); total_s, total_f = 0, 0
                my_bar = st.progress(0, text="Processando arquivos submetidos...")
                for idx, uf in enumerate(uploaded_files):
                    t_path = os.path.join("Ligantes_temp", sanitize_filename(uf.name))
                    with open(t_path, "wb") as f: f.write(uf.getbuffer())
                    out_prefix = f"Ligantes_temp/{os.path.splitext(os.path.basename(t_path))[0]}_.sdf"
                    subprocess.run(["obabel", t_path, "-osdf", "-O", out_prefix, "-m", "-e"], capture_output=True)
                    sdf_files = glob.glob(f"Ligantes_temp/{os.path.splitext(os.path.basename(t_path))[0]}_*.sdf")
                    for sdf_split in sdf_files:
                        try:
                            suppl = Chem.SDMolSupplier(sdf_split); mol = next(suppl) if suppl else None
                            if mol:
                                mol_3d = Chem.AddHs(mol)
                                if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) == 0:
                                    AllChem.MMFFOptimizeMolecule(mol_3d)
                                    writer = Chem.SDWriter(sdf_split); writer.write(mol_3d); writer.close()
                                    final_pdbqt = f"Ligantes/{os.path.basename(sdf_split).replace('.sdf', '')}.pdbqt"
                                    subprocess.run(["obabel", "-isdf", sdf_split, "-opdbqt", "-O", final_pdbqt, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                                    if os.path.exists(final_pdbqt): total_s += 1
                                    else: total_f += 1
                                else: total_f += 1
                            else: total_f += 1
                        except: total_f += 1
                    my_bar.progress(int(((idx+1)/len(uploaded_files))*100), text=f"Arquivos: {idx+1}/{len(uploaded_files)}")
                my_bar.empty() 
                if total_s>0: st.success(f"Biblioteca processada: {total_s} moléculas prontas."); st.session_state.lig_final = "Múltiplos (Modo Lote)"
                else: st.error("Nenhuma molécula válida processada.")

# ==========================================
# ABA 4: Grid Box (LaBOX)
# ==========================================
with tab_gridbox:
    st.header("4. Mapeamento do Espaço de Busca (Grid Box)")
    
    with st.container(border=True):
        st.subheader("Estratégia de Docking e Definição da Caixa")
        tipo_docking = st.radio("Estratégia computacional:", ["🎯 Site-Directed (Focado no Sítio)", "🌍 Blind Docking (Busca Global)"], horizontal=True)
    
    st.divider()

    col_box1, col_box2 = st.columns([1.2, 1])
    
    with col_box1:
        if "Site-Directed" in tipo_docking:
            with st.container(border=True):
                st.subheader("Auto-Mapeamento do Sítio")
                st.markdown("Utiliza o algoritmo **LaBOX** para calcular o centro e dimensões baseados na posição nativa de um ligante co-cristalizado de referência.")
                default_ref_pdb = st.session_state.extracted_lig_pdb if (st.session_state.redocking_mode and os.path.exists(st.session_state.extracted_lig_pdb)) else "ligante_referencia.pdb"
                box_input_pdb = st.text_input("Ligante de Referência (PDB):", value=default_ref_pdb)
                
                btn_labox = st.button("Calcular Sítio Ativo (LaBOX)", type="primary", use_container_width=True)
                
                if btn_labox:
                    if not os.path.exists(box_input_pdb): st.error("PDB de referência não encontrado.")
                    else:
                        try:
                            with st.spinner("Mapeando topologia com LaBOX..."):
                                if not os.path.exists("LaBOX.py"):
                                    r_labox = requests.get("https://raw.githubusercontent.com/RyanZR/LaBOX/main/LaBOX.py")
                                    with open("LaBOX.py", "w") as f: f.write(r_labox.text)
                                res_l = subprocess.run([sys.executable, "LaBOX.py", "-l", box_input_pdb, "-c"], capture_output=True, text=True)
                                if res_l.returncode == 0:
                                    output = res_l.stdout
                                    match_c = re.search(r'X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z\s+([-\d.]+)', output)
                                    match_s = re.search(r'W\s+([-\d.]+)\s+H\s+([-\d.]+)\s+D\s+([-\d.]+)', output)
                                    if match_c and match_s:
                                        st.session_state.cx, st.session_state.cy, st.session_state.cz = map(float, match_c.groups())
                                        st.session_state.sx, st.session_state.sy, st.session_state.sz = map(float, match_s.groups())
                                        st.success("Coordenadas do sítio calculadas!"); time.sleep(1); st.rerun() 
                                else: st.error("Erro interno no LaBOX.")
                        except Exception as e: st.error(f"Erro: {e}")

        else: # Blind Docking
            with st.container(border=True):
                st.subheader("Auto-Mapeamento Global")
                st.markdown("Utiliza o algoritmo **LaBOX** para calcular uma 'Bounding Box' que envolve toda a estrutura da proteína alvo para busca global de sítios.")
                box_input_pdb_blind = st.text_input("Receptor PDB (Inteiro):", value=st.session_state.rec_pdb_final)
                btn_blind = st.button("Calcular Bounding Box Global", type="primary", use_container_width=True)
                
                if btn_blind:
                    if not os.path.exists(box_input_pdb_blind): st.error("PDB do receptor não encontrado.")
                    else:
                        try:
                            with st.spinner("Mapeando limites globais com LaBOX..."):
                                if not os.path.exists("LaBOX.py"):
                                    r_labox = requests.get("https://raw.githubusercontent.com/RyanZR/LaBOX/main/LaBOX.py")
                                    with open("LaBOX.py", "w") as f: f.write(r_labox.text)
                                res_l = subprocess.run([sys.executable, "LaBOX.py", "-l", box_input_pdb_blind, "-c"], capture_output=True, text=True)
                                if res_l.returncode == 0:
                                    output = res_l.stdout
                                    match_c = re.search(r'X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z\s+([-\d.]+)', output)
                                    match_s = re.search(r'W\s+([-\d.]+)\s+H\s+([-\d.]+)\s+D\s+([-\d.]+)', output)
                                    if match_c and match_s:
                                        st.session_state.cx, st.session_state.cy, st.session_state.cz = map(float, match_c.groups())
                                        # Blind docking geralmente precisa de uma caixa maior que os limites exatos
                                        st.session_state.sx, st.session_state.sy, st.session_state.sz = map(lambda x: float(x)+2.0, match_s.groups())
                                        st.success("Caixa global calculada!"); time.sleep(1); st.rerun() 
                                else: st.error("Erro interno no LaBOX.")
                        except Exception as e: st.error(f"Erro: {e}")

    with col_box2:
        with st.container(border=True):
            st.subheader("Coordenadas Finais e Visualização (Å)")
            st.markdown("**Centro da Caixa (Cartesiano):**")
            c_x, c_y, c_z = st.columns(3)
            cx = c_x.number_input("X Centro", key='cx', step=0.1, value=st.session_state.cx)
            cy = c_y.number_input("Y Centro", key='cy', step=0.1, value=st.session_state.cy)
            cz = c_z.number_input("Z Centro", key='cz', step=0.1, value=st.session_state.cz)
            
            st.markdown("**Dimensões da Caixa (Largura/Altura/Profundidade):**")
            s_x, s_y, s_z = st.columns(3)
            sx = s_x.number_input("W Largura", key='sx', step=0.1, value=st.session_state.sx)
            sy = s_y.number_input("H Altura", key='sy', step=0.1, value=st.session_state.sy)
            sz = s_z.number_input("D Profundidade", key='sz', step=0.1, value=st.session_state.sz)

            if os.path.exists(st.session_state.rec_pdb_final):
                st.markdown("---")
                st.markdown("**Caixa de Busca em Tempo Real:**")
                with st.spinner("Renderizando caixa no alvo..."):
                    with open(st.session_state.rec_pdb_final, 'r') as f:
                        viewer = py3Dmol.view(width=500, height=400)
                        viewer.addModel(f.read(), "pdb")
                        viewer.setStyle({"cartoon": {'color':'lightgray'}})
                        viewer.addBox({'center': {'x': cx, 'y': cy, 'z': cz}, 'dimensions': {'w': sx, 'h': sy, 'd': sz}, 'color': '#1E88E5', 'wireframe': True})
                        viewer.zoomTo()
                        html(viewer._make_html(), width=500, height=400)

# ==========================================
# ABA 5: Configuração Vina
# ==========================================
with tab_vina_conf:
    st.header("5. Geração de Protocolo de Docking (config.txt)")
    
    with st.container(border=True):
        col_conf1, col_conf2 = st.columns(2)
        with col_conf1:
            vina_receptor = st.text_input("Arquivo Receptor (PDBQT):", value=st.session_state.rec_final)
            if st.session_state.vs_mode: st.info("🔹 Modo Triagem Ativado: O config não especificará o ligante.")
            else: vina_ligante = st.text_input("Arquivo Ligante (PDBQT):", value=st.session_state.lig_final)
            vina_config_name = st.text_input("Nome do arquivo config:", value="config.txt")
        
        with col_conf2:
            vina_exhaustiveness = st.number_input("Poder Computacional (Exhaustiveness):", min_value=1, value=24, help="Padrão Vina é 8. Para IC/Publicação, 24-48 é recomendado.")
            max_cpus = multiprocessing.cpu_count()
            st.success(f"⚡ Autodetecção Vina: Alocação automática de {max_cpus} núcleos lógicos.")

        btn_gen_conf = st.button("Gerar Ordem de Cálculo 'config.txt'", type="primary", use_container_width=True)

        if btn_gen_conf:
            c_content = f"receptor = {vina_receptor}\n"
            if not st.session_state.vs_mode: c_content += f"ligand = {vina_ligante}\n"
            c_content += f"\ncenter_x = {st.session_state.cx}\ncenter_y = {st.session_state.cy}\ncenter_z = {st.session_state.cz}\n"
            c_content += f"size_x = {st.session_state.sx}\nsize_y = {st.session_state.sy}\nsize_z = {st.session_state.sz}\n"
            c_content += f"\nexhaustiveness = {vina_exhaustiveness}\ncpu = {max_cpus}\n"
            with open(vina_config_name, "w") as f: f.write(c_content)
            st.success(f"Arquivo config pronto."); st.code(c_content, language="ini")

# ==========================================
# ABA 6: Execução do Docking Molecular
# ==========================================
with tab_executar:
    st.header("6. Simulação Termodinâmica (Rigor Científico)")
    
    with st.container(border=True):
        st.subheader("Status do Servidor de Cálculo")
        if is_server_busy(): st.warning("⏳ **Fila de Espera Ativa:** O servidor está rodando uma simulação para outro usuário. Por favor, aguarde para não travar a nuvem.")
        else: st.success("✅ **Fila Livre:** O processador está disponível para seu cálculo multicore.")

    st.divider()
        
    vina_exe = "vina_1.2.7_linux_x86_64"
    config_file_exec = st.text_input("Arquivo de configuração a ler:", value="config.txt")
    
    if st.session_state.vs_mode:
        with st.container(border=True):
            st.subheader("Modo Triagem Virtual em Lote (Lote HTVS)")
            st.warning("⚠️ O Vina será executado 3 vezes independentes para TODAS as moléculas na pasta 'Ligantes/'.")
            
            data_atual = datetime.now().strftime("%Y%m%d_%H%M")
            output_dir_input = st.text_input("Diretório Base de Saída:", value=f"HTVS_results_{data_atual}")
            btn_vs = st.button("▶️ Iniciar Triagem HTVS em Triplicata", type="primary", use_container_width=True)
            
            if btn_vs:
                if not os.path.exists(config_file_exec): st.error("Configuração não encontrada.")
                elif is_server_busy(): st.error("Servidor ocupado. Aguarde.")
                else:
                    try:
                        lock_server(); st.session_state.vs_results_dir = output_dir_input
                        if not os.path.exists(vina_exe):
                            st.info("Baixando binários Vina (Linux)...")
                            with open(vina_exe, 'wb') as f: f.write(requests.get(f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{vina_exe}").content)
                            os.chmod(vina_exe, 0o755)
                        
                        my_bar = st.progress(0, text="Preparando Triagem em Triplicata multicore...")
                        l_placeholder = st.empty(); logs = ""
                        for rep in range(1, 4):
                            r_dir = os.path.join(output_dir_input, f"rep{rep}")
                            os.makedirs(r_dir, exist_ok=True)
                            cmd_b = f"./{vina_exe} --config {config_file_exec} --batch Ligantes/*.pdbqt --dir {r_dir}"
                            logs += f"\n=== INICIANDO HTVS LOTE: REPLICATA {rep} DE 3 ===\n"
                            l_placeholder.code(logs, language="text")
                            process = subprocess.Popen(cmd_b, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                            l_up = time.time()
                            for line in iter(process.stdout.readline, ''):
                                logs += line
                                if time.time() - l_up > 0.5: l_placeholder.code(logs, language="text"); l_up = time.time()
                            process.wait(); my_bar.progress(int((rep/3)*100), text=f"HTVS Lote: Rep {rep}/3")
                        st.session_state.vina_log_output = logs; my_bar.empty()
                        st.success(f"🎉 Triagem Virtual em Triplicata concluída! Resultados em '{output_dir_input}/'")
                    except Exception as e: st.error(f"Erro: {e}")
                    finally: unlock_server()
    
    else: # Modo Individual
        with st.container(border=True):
            st.subheader("Modo Ligante Único (Triplicata)")
            out_pdbqt_base = st.text_input("Nome base das Poses:", value=f"resultado_docking_{sanitize_filename(st.session_state.nome_ligante_salvar)}")
            btn_single = st.button("▶️ Iniciar Docking em Triplicata", type="primary", use_container_width=True)
            
            if btn_single:
                if not os.path.exists(config_file_exec): st.error("Configuração não encontrada.")
                elif is_server_busy(): st.error("Servidor ocupado. Aguarde.")
                else:
                    try:
                        lock_server()
                        if not os.path.exists(vina_exe):
                            with open(vina_exe, 'wb') as f: f.write(requests.get(f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{vina_exe}").content)
                            os.chmod(vina_exe, 0o755)
                        
                        my_bar = st.progress(0, text="Iniciando motor Vina (Multi-Replicatas)...")
                        l_placeholder = st.empty(); logs = ""
                        for rep in range(1, 4):
                            o_r = f"{out_pdbqt_base}_rep{rep}.pdbqt"
                            c = [f"./{vina_exe}", "--config", config_file_exec, "--out", o_r]
                            logs += f"\n=== INICIANDO DOCKING: REPLICATA {rep} DE 3 ===\n"
                            l_placeholder.code(logs, language="text")
                            process = subprocess.Popen(c, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
                            l_up = time.time()
                            for line in iter(process.stdout.readline, ''):
                                logs += line
                                if time.time() - l_up > 0.5: l_placeholder.code(logs, language="text"); l_up = time.time()
                            process.wait(); my_bar.progress(int((rep/3)*100), text=f"Replicata {rep}/3")
                        st.session_state.vina_log_output = logs; st.session_state.single_result_base = out_pdbqt_base; my_bar.empty()
                        st.success("✅ Simulação concluída! Vá para a Aba 7 para análise estocástica."); st.balloons()
                    except Exception as e: st.error(f"Erro: {e}")
                    finally: unlock_server()

    # --- LOG PERMANENTE ---
    if st.session_state.vina_log_output:
        st.divider()
        with st.container(border=True):
            st.subheader("Último Log de Execução (Terminal Vina)")
            st.text_area("Saída bruta (Multicore)", value=st.session_state.vina_log_output, height=300)

# ==========================================
# ABA 7: Análise Químico-Estrutural e Exportação
# ==========================================
with tab_visualizar:
    st.header("7. Análise de Resultados e Consolidação")
    
    st.divider()
    st.subheader("📈 Tabela Termodinâmica Global (Média Estocástica)")

    if st.session_state.vs_mode:
        if not st.session_state.get('vs_results_dir') or not os.path.exists(st.session_state.vs_results_dir):
            st.warning("Execute o HTVS (Aba 6) primeiro.")
        else:
            with st.spinner("Compilando dados das replicatas..."):
                with st.container(border=True):
                    lig_f = glob.glob(os.path.join(st.session_state.vs_results_dir, "rep1", "*.pdbqt"))
                    results = []
                    for f in lig_f:
                        base = os.path.basename(f)
                        v1, v2, v3 = get_vina_affinity(f), get_vina_affinity(os.path.join(st.session_state.vs_results_dir, "rep2", base)), get_vina_affinity(os.path.join(st.session_state.vs_results_dir, "rep3", base))
                        vals = [v for v in [v1, v2, v3] if not np.isnan(v)]
                        if vals:
                            mean_v = round(np.mean(vals), 2); std_v = round(np.std(vals), 2)
                            results.append({"Ligante": base.replace('_out.pdbqt', '').replace('.pdbqt', ''), "Média (kcal/mol)": mean_v, "DP": std_v, "Rep 1": v1, "Rep 2": v2, "Rep 3": v3})
                    
                    if results:
                        df = pd.DataFrame(results).sort_values(by="Média (kcal/mol)")
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        st.plotly_chart(px.bar(df, x="Ligante", y="Média (kcal/mol)", error_y="DP", title="Afinidade (ΔG) com Desvio Padrão", color="Média (kcal/mol)", color_continuous_scale="Viridis"), use_container_width=True)
                    else: st.error("Nenhum dado termodinâmico lido.")
            
            st.divider()
            with st.container(border=True):
                col_vs1, col_vis2 = st.columns([1, 2]); 
                all_p = glob.glob(os.path.join(st.session_state.vs_results_dir, "rep*", "*.pdbqt"))
                opcoes = [f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}" for p in all_p]
                
                with col_vs1:
                    st.subheader("Síntese Interativa")
                    if opcoes:
                        selected = st.selectbox("Escolha pose exata (HTVS):", opcoes)
                        if st.button("Sintetizar Complexo PDB", use_container_width=True):
                            f_path = os.path.join(st.session_state.vs_results_dir, selected)
                            with st.spinner("Fundindo matriz..."):
                                lines = []
                                with open(f_path, 'r') as f:
                                    for l in f:
                                        if l.startswith("MODEL 1"): lines.append(l)
                                        elif l.startswith("ENDMDL") and lines: lines.append(l); break
                                        elif lines: lines.append(l)
                                with open("temp_pose.pdbqt", "w") as f: f.writelines(lines)
                                subprocess.run(["obabel", "-ipdbqt", "temp_pose.pdbqt", "-opdb", "-O", "temp_pose.pdb"])
                                if os.path.exists("temp_pose.pdb"):
                                    with open(st.session_state.rec_pdb_final, 'r') as f: rec_l = "".join([l for l in f if not l.startswith("END")])
                                    with open("temp_pose.pdb", 'r') as f: lig_l = "".join(["HETATM" + l[6:17] + "UNL" + l[20:] for l in f if l.startswith("ATOM") or l.startswith("HETATM")])
                                    n_c = selected.replace('/', '_').replace('.pdbqt', '').replace('_out', '')
                                    with open(f"complex_{n_c}.pdb", "w") as f: f.write(rec_l + lig_l + "END\n")
                                    st.session_state.complex_generated = True; st.session_state.complex_file = f"complex_{n_c}.pdb"; st.session_state.rec_str = rec_l; st.session_state.lig_str = lig_l; st.success("Complexo pronto!")

                with col_vis2:
                    if st.session_state.get('complex_generated', False):
                        st.subheader("Visualização do Complexo Holo")
                        v_comp = py3Dmol.view(width=700, height=450)
                        v_comp.addModel(st.session_state.rec_str, "pdb"); v_comp.setStyle({'model': 0}, {"cartoon": {'color': 'spectrum'}})
                        v_comp.addModel(st.session_state.lig_str, "pdb"); v_comp.setStyle({'model': 1}, {"stick": {'colorscheme': 'magentaCarbon', 'radius': 0.15}})
                        v_comp.zoomTo({'model': 1}); html(v_comp._make_html(), width=700, height=450)

            st.divider()
            with st.container(border=True):
                st.subheader("📦 Exportação Completa")
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zip_f:
                    if os.path.exists(st.session_state.vs_results_dir):
                        for root, _, files in os.walk(st.session_state.vs_results_dir):
                            for file in files: zip_f.write(os.path.join(root, file), os.path.join("PDBQT_Poses", file))
                    if os.path.exists("config.txt"): zip_f.write("config.txt", "config_utilizado.txt")
                st.download_button("📥 Baixar ZIP com Todos os Resultados", data=buf.getvalue(), file_name=f"Resultados_HTVS_{st.session_state.vs_results_dir}.zip", mime="application/zip", type="primary", use_container_width=True)

    else: # Modo Único
        base_n = st.session_state.get('single_result_base', '')
        if not base_n or not os.path.exists(f"{base_n}_rep1.pdbqt"):
            st.warning("Execute o Docking (Aba 6) primeiro.")
        else:
            with st.container(border=True):
                v1, v2, v3 = get_vina_affinity(f"{base_n}_rep1.pdbqt"), get_vina_affinity(f"{base_n}_rep2.pdbqt"), get_vina_affinity(f"{base_n}_rep3.pdbqt")
                vals = [v for v in [v1, v2, v3] if not np.isnan(v)]
                if vals:
                    mean_v = round(np.mean(vals), 2); std_v = round(np.std(vals), 2)
                    df = pd.DataFrame([{"Sistema": os.path.basename(base_n), "Média (kcal/mol)": mean_v, "DP": std_v, "Rep 1": v1, "Rep 2": v2, "Rep 3": v3}])
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.plotly_chart(px.bar(df, x="Sistema", y="Média (kcal/mol)", error_y="DP", title="Afinidade Termodinâmica (ΔG)", color="Média (kcal/mol)", color_continuous_scale="Viridis"), use_container_width=True)

            st.divider()
            with st.container(border=True):
                col_vis1, col_vis2 = st.columns([1, 2])
                with col_vis1:
                    st.subheader("Síntese do Complexo PDB")
                    rep = st.selectbox("Escolha Replicata:", ["rep1", "rep2", "rep3"])
                    if st.button("Sintetizar Modelo Holo", type="primary", use_container_width=True):
                        f_p = f"{base_n}_{rep}.pdbqt"
                        with st.spinner("Fundindo matrizes..."):
                            lines = []
                            with open(f_p, 'r') as f:
                                for l in f:
                                    if l.startswith("MODEL 1"): lines.append(l)
                                    elif l.startswith("ENDMDL") and lines: lines.append(l); break
                                    elif lines: lines.append(l)
                            with open("temp_pose.pdbqt", "w") as f: f.writelines(lines)
                            subprocess.run(["obabel", "-ipdbqt", "temp_pose.pdbqt", "-opdb", "-O", "temp_pose.pdb"])
                            if os.path.exists(st.session_state.rec_pdb_final) and os.path.exists("temp_pose.pdb"):
                                with open(st.session_state.rec_pdb_final, 'r') as f: rec_l = "".join([l for l in f if not l.startswith("END")])
                                with open("temp_pose.pdb", 'r') as f: lig_l = "".join(["HETATM" + l[6:17] + "UNL" + l[20:] for l in f if l.startswith("ATOM") or l.startswith("HETATM")])
                                c_out = f"complex_{os.path.basename(base_n)}_{rep}.pdb"
                                with open(c_out, "w") as f: f.write(rec_l + lig_l + "END\n")
                                st.session_state.complex_generated = True; st.session_state.complex_file = c_out; st.session_state.rec_str = rec_l; st.session_state.lig_str = lig_l; st.success("Complexo consolidado!")
                                with open(c_out, "r") as f: st.download_button("📥 Baixar PDB do Complexo Holo", data=f.read(), file_name=c_out, mime="text/plain", type="secondary", use_container_width=True)

                with col_vis2:
                    if st.session_state.get('complex_generated', False):
                        st.subheader("Visualização Interativa")
                        v_comp = py3Dmol.view(width=700, height=450)
                        v_comp.addModel(st.session_state.rec_str, "pdb"); v_comp.setStyle({'model': 0}, {"cartoon": {'color': 'spectrum'}})
                        v_comp.addModel(st.session_state.lig_str, "pdb"); v_comp.setStyle({'model': 1}, {"stick": {'colorscheme': 'magentaCarbon', 'radius': 0.15}})
                        v_comp.zoomTo({'model': 1}); html(v_comp._make_html(), width=700, height=450)

# ==========================================
# ABA 8: Referências Bibliográficas
# ==========================================
with tab_referencias:
    st.header("📚 Referências Bibliográficas e Algoritmos")
    with st.container(border=True):
        st.markdown("""
        Este laboratório virtual é fundamentado nos rigorosos algoritmos de biologia estrutural. Para aprofundar-se, consulte as referências:

        * **AutoDock Vina:** Trott, O., & Olson, A. J. (2010). AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *J Comp Chem*, 31(2), 455-461.
        * **OpenBabel:** O'Boyle, N. M., et al. (2011). Open Babel: An open chemical toolbox. *J Cheminf*, 2(1), 5.
        * **RDKit:** Open-source cheminformatics for topology.
        * **PDBFixer / OpenMM:** Eastman, P., et al. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. *PLoS comp biol*, 13(7), e1005659.
        * **Química Medicinal:** Barreiro, E. J., & Fraga, C. A. M. (2015). *Química Medicinal: As Bases Farmacológicas da Ação dos Fármacos*. 3ª Ed. Artmed.
        """)
