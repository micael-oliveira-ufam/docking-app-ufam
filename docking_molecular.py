import streamlit as st
import os
import subprocess
import sys
import requests
import re
import glob
import time
from datetime import datetime
import numpy as np
import multiprocessing

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
        # Proteção Anti-Zumbi: Se o lock tem mais de 30 minutos, apaga (caso o app tenha travado antes)
        file_age = time.time() - os.path.getmtime(LOCK_FILE)
        if file_age > 1800: 
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
# FUNÇÕES AUXILIARES
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

# Configuração da página
st.set_page_config(page_title="Docking Molecular - FCF/UFAM", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# BARRA LATERAL (AUTORIA E LOGO)
# ==========================================
with st.sidebar:
    try:
        st.image("logo_ufam.jpg", use_container_width=True)
    except Exception:
        st.warning("Logo da UFAM ('logo_ufam.jpg') não encontrado no diretório.")
    
    st.markdown("---")
    st.markdown("### Autoria do Projeto")
    st.markdown("**Micael Davi Lima de Oliveira**")
    st.markdown("*Iniciação Científica*")
    st.markdown("**Faculdade de Ciências Farmacêuticas**")
    st.markdown("Universidade Federal do Amazonas (UFAM)")
    st.markdown("---")
    st.caption("Desenvolvido para ensino e pesquisa em Química Medicinal Computacional.")

# Cabeçalho Principal
st.title("🧬 Laboratório Virtual: Docking Molecular e Triagem")
st.markdown("Plataforma acadêmica para ensino de **Química Medicinal Computacional** e **Planejamento de Fármacos**.")

# Inicialização das variáveis de memória do Streamlit
if 'cx' not in st.session_state: st.session_state.cx = 0.0
if 'cy' not in st.session_state: st.session_state.cy = 0.0
if 'cz' not in st.session_state: st.session_state.cz = 0.0
if 'sx' not in st.session_state: st.session_state.sx = 20.0
if 'sy' not in st.session_state: st.session_state.sy = 20.0
if 'sz' not in st.session_state: st.session_state.sz = 20.0
if 'smiles' not in st.session_state: st.session_state.smiles = ""
if 'nome_ligante_salvar' not in st.session_state: st.session_state.nome_ligante_salvar = "ligante"
if 'rec_pdb_final' not in st.session_state: st.session_state.rec_pdb_final = "receptor_prep.pdb"
if 'rec_final' not in st.session_state: st.session_state.rec_final = "receptor.pdbqt"
if 'lig_final' not in st.session_state: st.session_state.lig_final = "ligante.pdbqt"
if 'original_pdb' not in st.session_state: st.session_state.original_pdb = "2XV7.pdb"
if 'redocking_mode' not in st.session_state: st.session_state.redocking_mode = False
if 'extracted_lig_pdb' not in st.session_state: st.session_state.extracted_lig_pdb = ""
if 'vs_mode' not in st.session_state: st.session_state.vs_mode = False
if 'vs_results_dir' not in st.session_state: st.session_state.vs_results_dir = ""
if 'vina_log_output' not in st.session_state: st.session_state.vina_log_output = "" 

# Abas
tab_install, tab_receptor, tab_ligante, tab_gridbox, tab_vina, tab_executar, tab_visualizar, tab_referencias = st.tabs([
    "🛠️ 1. Ambiente", "🧬 2. Receptor", "💊 3. Ligante", "📦 4. Grid Box", "⚙️ 5. Vina Config", "🚀 6. Docking (Triplicata)", "👁️ 7. Análise de Resultados", "📚 8. Referências"
])

# ==========================================
# ABA 1: Instalação de Dependências
# ==========================================
with tab_install:
    st.header("1. Verificação do Ambiente Computacional")
    
    with st.expander("📚 Fundamentos: O que é o Docking Molecular?", expanded=True):
        st.markdown("""
        O **Docking Molecular** é uma técnica fundamental no Planejamento de Fármacos Baseado em Estrutura (SBDD). O objetivo computacional é prever a conformação tridimensional preferencial (a **Pose**) de uma molécula pequena (fármaco/ligante) quando ligada a uma macromolécula (receptor/proteína), formando um complexo estável.
        """)

    st.markdown("### ☁️ Status de Nuvem (Streamlit Cloud)")
    st.markdown("Se você estiver rodando este software localmente, certifique-se de instalar as dependências. Na nuvem do Streamlit, as bibliotecas são carregadas via `requirements.txt` e o OpenBabel via `packages.txt`.")
    
    if LIBS_INSTALADAS:
        st.success("✅ **Sistema Operante:** Todas as bibliotecas de quimioinformática e bioinformática foram detectadas com sucesso. A plataforma está pronta para uso!")
    else:
        st.error("🚨 **Atenção:** Módulos fundamentais ausentes. Verifique os arquivos requirements.txt no repositório.")

# ==========================================
# ABA 2: Preparação do Receptor
# ==========================================
with tab_receptor:
    st.header("2. O Alvo Farmacológico (Receptor)")
    
    with st.expander("📚 Fundamentos: Preparação Rígida e Cargas Parciais", expanded=False):
        st.markdown("""
        ### Evitando Distorções Conformacionais
        Proteínas cristalográficas muitas vezes possuem "gaps". Tentar reconstruí-los no vácuo sem dinâmica molecular causa **clashes estéricos** e distorce a fenda cristalográfica original. Por isso, desativamos o preenchimento artificial.
        
        ### Cargas de Gasteiger
        A conversão para PDBQT requer a adição de cargas parciais. O método de Gasteiger-Marsili calcula iterativamente a distribuição da nuvem eletrônica com base na eletronegatividade, vital para que o Vina calcule forças de atração eletrostática de Coulomb.
        """)

    col1, col2 = st.columns([1, 2])
    with col1:
        pdb_id = st.text_input("Código PDB ID:", value="2XV7")
        if st.button("Baixar e Visualizar"):
            r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb")
            if r.status_code == 200:
                with open(f"{pdb_id.upper()}.pdb", "w") as f: f.write(r.text)
                st.session_state.original_pdb = f"{pdb_id.upper()}.pdb"
                st.success(f"PDB {pdb_id.upper()} salvo!")
            else:
                st.error("Falha ao buscar PDB no servidor.")

    with col2:
        if os.path.exists(st.session_state.original_pdb):
            with open(st.session_state.original_pdb, 'r') as f:
                viewer = py3Dmol.view(width=600, height=350)
                viewer.addModel(f.read(), "pdb")
                viewer.setStyle({"cartoon": {'color':'spectrum'}})
                viewer.zoomTo()
                html(viewer._make_html(), width=600, height=350)

    st.divider()

    col_prep1, col_prep2 = st.columns(2)
    with col_prep1:
        st.subheader("A. Limpeza Conservadora (PDBFixer)")
        if st.button("Remover Água e Protonar (pH 7.4)"):
            try:
                with st.spinner("Limpando sem distorcer..."):
                    fixer = PDBFixer(filename=st.session_state.original_pdb)
                    fixer.findNonstandardResidues()
                    fixer.replaceNonstandardResidues()
                    fixer.removeHeterogens(False)
                    fixer.addMissingHydrogens(7.4) 
                    out_pdb = f"{pdb_id.upper()}_prep.pdb"
                    PDBFile.writeFile(fixer.topology, fixer.positions, open(out_pdb, 'w'))
                    st.session_state.rec_pdb_final = out_pdb 
                    st.success(f"Receptor alvo isolado conservando topologia nativa.")
            except Exception as e: st.error(f"Erro no PDBFixer: {e}")

    with col_prep2:
        st.subheader("B. Atribuição Eletrostática (OpenBabel)")
        if st.button("Calcular Cargas Gasteiger e Rígidez (PDBQT)"):
            try:
                with st.spinner("Calculando cargas parciais iterativas..."):
                    out_pdbqt = st.session_state.rec_pdb_final.replace(".pdb", ".pdbqt")
                    comando = ["obabel", "-i", "pdb", st.session_state.rec_pdb_final, "-o", "pdbqt", "-O", out_pdbqt, "-xr", "--partialcharge", "gasteiger"]
                    subprocess.run(comando, capture_output=True, text=True)
                    if os.path.exists(out_pdbqt):
                        st.session_state.rec_final = out_pdbqt
                        st.success(f"Matriz de cargas gerada: {out_pdbqt}")
                    else:
                        st.error("Erro ao gerar PDBQT. O OpenBabel está no packages.txt?")
            except Exception as e: st.error(f"Erro: {e}")

# ==========================================
# ABA 3: Preparação do Ligante
# ==========================================
with tab_ligante:
    st.header("3. Preparação do(s) Fármaco(s)")
    
    with st.expander("📚 Fundamentos: Minimização e Triagem Virtual (Virtual Screening)", expanded=False):
        st.markdown("""
        ### Otimização Tridimensional Rápida (RDKit)
        A geração 3D e o relaxamento do campo de força (MMFF94) da molécula são executados via `RDKit (ETKDG)`, conferindo otimização em frações de segundo para nuvem.
        
        ### Triagem Virtual Sequencial (Virtual Screening)
        O algoritmo HTVS processa arquivos enviados de forma sequencial (uma molécula por vez), extraindo as matrizes via OpenBabel, gerando a conformação geométrica nativa 3D com RDKit, e finalmente convertendo para PDBQT, garantindo extrema estabilidade de memória em ambientes Cloud.
        """)

    modo_preparacao = st.radio("Selecione a Estratégia de Processamento:", [
        "🔬 Triagem Simples: Molécula Única (SMILES/Nome)", 
        "♻️ Validação do Método: Re-Docking (Extrair Fármaco do PDB)",
        "🚀 Triagem Virtual Automática: Lote de Ligantes (Upload .sdf/.mol2/.pdb)"
    ])

    if "Triagem Simples" in modo_preparacao:
        st.session_state.redocking_mode = False
        st.session_state.vs_mode = False
        col_input, col_2d, col_3d = st.columns([1.2, 1, 1])
        
        with col_input:
            tipo_entrada = st.radio("Formato de entrada:", ("Nome Comum", "Código SMILES"))
            entrada_ligante = st.text_input("Insira o valor químico:")
            
            if st.button("1. Gerar Topologia (2D) e Nomenclatura"):
                if entrada_ligante:
                    try:
                        smiles_obtido = entrada_ligante
                        nome_final = "mol_inedita"
                        with st.spinner("Analisando estrutura química..."):
                            if "Nome" in tipo_entrada:
                                comps = pcp.get_compounds(entrada_ligante, 'name')
                                if comps:
                                    smiles_obtido = comps[0].isomeric_smiles
                                    nome_final = sanitize_filename(entrada_ligante)
                                    st.success(f"SMILES Encontrado: {smiles_obtido}")
                                else:
                                    st.error("Molécula não encontrada. Tente inserir via SMILES.")
                                    smiles_obtido = ""
                            else:
                                try:
                                    comps = pcp.get_compounds(smiles_obtido, 'smiles')
                                    if comps and comps[0].iupac_name:
                                        nome_final = sanitize_filename(comps[0].iupac_name)
                                        st.success(f"IUPAC Localizado: {comps[0].iupac_name}")
                                    else:
                                        st.info("SMILES Inédito. Nomeado de forma genérica.")
                                except:
                                    st.info("SMILES Inédito / Nova entidade estrutural detectada.")
                                        
                        if smiles_obtido:
                            st.session_state.smiles = smiles_obtido
                            st.session_state.nome_ligante_salvar = nome_final
                            mol = Chem.MolFromSmiles(smiles_obtido)
                            if mol: 
                                st.session_state.img_2d = Draw.MolToImage(mol, size=(300, 300))
                            else:
                                st.error("Erro na leitura estrutural. O código SMILES pode estar incorreto.")
                    except Exception as e: st.error(f"Erro: {e}")

            if st.session_state.smiles:
                st.info(f"O sistema salvará os arquivos como: **{st.session_state.nome_ligante_salvar}**")
                
                if st.button("2. Minimizar (3D) Rápido e Gerar PDBQT", type="primary"):
                    sdf_file = f"{st.session_state.nome_ligante_salvar}.sdf"
                    pdbqt_file = f"{st.session_state.nome_ligante_salvar}.pdbqt"
                    
                    with st.spinner("Gerando 3D ultrarrápido (RDKit ETKDG + MMFF94)..."):
                        try:
                            mol_3d = Chem.MolFromSmiles(st.session_state.smiles)
                            mol_3d = Chem.AddHs(mol_3d) 
                            
                            AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) 
                            AllChem.MMFFOptimizeMolecule(mol_3d) 
                            
                            writer = Chem.SDWriter(sdf_file)
                            writer.write(mol_3d)
                            writer.close()
                            
                            subprocess.run(["obabel", "-isdf", sdf_file, "-opdbqt", "-O", pdbqt_file, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                            
                            if os.path.exists(sdf_file) and os.path.exists(pdbqt_file):
                                st.session_state.mol2_file_path = sdf_file 
                                st.session_state.lig_final = pdbqt_file 
                                st.success(f"Ligante 3D otimizado com sucesso: '{pdbqt_file}'")
                        except Exception as e:
                            st.error(f"Falha na geração 3D acelerada: {e}")

        with col_2d:
            st.markdown("### Topologia 2D")
            if 'img_2d' in st.session_state: st.image(st.session_state.img_2d)

        with col_3d:
            st.markdown("### Geometria Espacial 3D")
            if 'mol2_file_path' in st.session_state and os.path.exists(st.session_state.mol2_file_path):
                with open(st.session_state.mol2_file_path, 'r') as f:
                    viewer_lig = py3Dmol.view(width=300, height=300)
                    formato = "sdf" if st.session_state.mol2_file_path.endswith(".sdf") else "mol2"
                    viewer_lig.addModel(f.read(), formato)
                    viewer_lig.setStyle({"stick": {'colorscheme': 'greenCarbon'}})
                    viewer_lig.zoomTo()
                    html(viewer_lig._make_html(), width=300, height=300)

    elif "Validação" in modo_preparacao:
        st.session_state.redocking_mode = True
        st.session_state.vs_mode = False
        st.info("Buscando agentes químicos co-cristalizados no PDB original da Etapa 2.")
        if os.path.exists(st.session_state.original_pdb):
            ligantes = get_ligands_from_pdb(st.session_state.original_pdb)
            if ligantes:
                lig_selecionado = st.selectbox("Fármaco co-cristalizado detectado:", ligantes)
                if st.button("Extrair e Manter Coordenadas Naturais", type="primary"):
                    ext_pdb = f"{lig_selecionado}_redocking.pdb"
                    ext_pdbqt = f"{lig_selecionado}_redocking.pdbqt"
                    extract_ligand_from_pdb(st.session_state.original_pdb, lig_selecionado, ext_pdb)
                    subprocess.run(["obabel", "-ipdb", ext_pdb, "-opdbqt", "-O", ext_pdbqt, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                    st.session_state.lig_final = ext_pdbqt
                    st.session_state.extracted_lig_pdb = ext_pdb
                    st.success(f"Matriz de coordenadas extraída com sucesso! Salvo como: {ext_pdbqt}")
            else:
                st.warning("Nenhum ligante orgânico detectado.")
    
    else:
        st.session_state.redocking_mode = False
        st.session_state.vs_mode = True
        st.info("📦 Módulo de High-Throughput Virtual Screening (HTVS) - Sequencial Otimizado")
        st.write("Faça o upload de múltiplos arquivos individuais OU de um único arquivo contendo várias moléculas (ex: biblioteca.sdf).")
        
        uploaded_files = st.file_uploader("Arquivos de Biblioteca de Fármacos", type=['sdf', 'mol2', 'pdb'], accept_multiple_files=True)
        
        if uploaded_files:
            if st.button("Processar Lote e Converter para PDBQT", type="primary"):
                os.makedirs("Ligantes_temp", exist_ok=True)
                os.makedirs("Ligantes", exist_ok=True)
                
                # Limpeza de memória
                for f in glob.glob("Ligantes_temp/*"): os.remove(f)
                for f in glob.glob("Ligantes/*.pdbqt"): os.remove(f)
                
                temp_paths = []
                for uf in uploaded_files:
                    t_path = os.path.join("Ligantes_temp", sanitize_filename(uf.name))
                    with open(t_path, "wb") as f:
                        f.write(uf.getbuffer())
                    temp_paths.append(t_path)
                
                total_sucesso = 0
                total_falha = 0
                
                progress_text = "Iniciando processamento sequencial seguro..."
                my_bar = st.progress(0, text=progress_text)
                
                for idx, t_path in enumerate(temp_paths):
                    base_name = os.path.splitext(os.path.basename(t_path))[0]
                    out_sdf_prefix = f"Ligantes_temp/{base_name}_.sdf"
                    
                    # Passo 1: Separar o arquivo bruto em SDFs individuais (OpenBabel)
                    cmd_split = ["obabel", t_path, "-osdf", "-O", out_sdf_prefix, "-m", "-e"]
                    subprocess.run(cmd_split, capture_output=True, text=True)
                    
                    sdf_files = glob.glob(f"Ligantes_temp/{base_name}_*.sdf")
                    
                    # Passo 2: RDKit (Geometria 3D rápida) e OpenBabel (Cargas Gasteiger e PDBQT)
                    for sdf_split in sdf_files:
                        try:
                            suppl = Chem.SDMolSupplier(sdf_split)
                            mol = next(suppl) if suppl else None
                            
                            if mol is not None:
                                # RDKit: Gera 3D com hidrogênios
                                mol_3d = Chem.AddHs(mol)
                                res_embed = AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
                                
                                if res_embed == 0: 
                                    AllChem.MMFFOptimizeMolecule(mol_3d)
                                    
                                    # Sobrescreve o arquivo com a estrutura 3D minimizada
                                    writer = Chem.SDWriter(sdf_split)
                                    writer.write(mol_3d)
                                    writer.close()
                                    
                                    # OpenBabel: Cargas Gasteiger e conversão PDBQT
                                    base_m2 = os.path.basename(sdf_split).replace(".sdf", "")
                                    final_pdbqt = f"Ligantes/{base_m2}.pdbqt"
                                    
                                    subprocess.run(["obabel", "-isdf", sdf_split, "-opdbqt", "-O", final_pdbqt, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                                    
                                    if os.path.exists(final_pdbqt):
                                        total_sucesso += 1
                                    else:
                                        total_falha += 1
                                else:
                                    total_falha += 1
                            else:
                                total_falha += 1
                        except Exception:
                            total_falha += 1
                    
                    percentual = int(((idx + 1) / len(temp_paths)) * 100)
                    my_bar.progress(percentual, text=f"Processado arquivo de biblioteca {idx+1} de {len(temp_paths)}...")
                
                my_bar.empty() # Remove a barra ao concluir
                qtd_gerados = len(glob.glob("Ligantes/*.pdbqt"))
                
                if qtd_gerados > 0:
                    st.success(f"🎉 Triagem Virtual preparada! Foram separadas, otimizadas em 3D (RDKit) e convertidas (OpenBabel) {qtd_gerados} moléculas de forma segura.")
                    if total_falha > 0:
                        st.warning(f"⚠️ Atenção: {total_falha} molécula(s) descartadas do lote devido a erros químicos ou de topologia.")
                    st.session_state.lig_final = "Múltiplos Ligantes (Modo Lote Ativado)"
                else:
                    st.error("Nenhuma molécula estruturalmente válida pôde ser extraída. Verifique os arquivos submetidos.")

# ==========================================
# ABA 4: Grid Box (LaBOX)
# ==========================================
with tab_gridbox:
    st.header("4. Mapeamento do Espaço de Busca (Grid Box)")
    
    tipo_docking = st.radio("Selecione a Estratégia Computacional:", ["🎯 Site-Directed Docking (Focado num sítio ativo)", "🌍 Blind Docking (Busca Global Automática)"])
    st.divider()

    col_box1, col_box2 = st.columns([1.2, 1])
    
    if "Site-Directed" in tipo_docking:
        with col_box1:
            if st.session_state.redocking_mode and os.path.exists(st.session_state.extracted_lig_pdb):
                box_input_pdb = st.text_input("Ligante de Referência PDB:", value=st.session_state.extracted_lig_pdb)
            else:
                box_input_pdb = st.text_input("Ligante de Referência PDB:", value="ligante_referencia.pdb")
                
            if st.button("Calcular Dimensões do Sítio (LaBOX)"):
                if not os.path.exists(box_input_pdb):
                    st.error("Arquivo PDB de referência não encontrado.")
                else:
                    try:
                        with st.spinner("Mapeando vizinhança topológica com LaBOX..."):
                            if not os.path.exists("LaBOX.py"):
                                r_labox = requests.get("https://raw.githubusercontent.com/RyanZR/LaBOX/main/LaBOX.py")
                                with open("LaBOX.py", "w") as f: f.write(r_labox.text)
                                
                            res_labox = subprocess.run([sys.executable, "LaBOX.py", "-l", box_input_pdb, "-c"], capture_output=True, text=True)
                            
                            if res_labox.returncode == 0:
                                output = res_labox.stdout
                                match_center = re.search(r'X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z\s+([-\d.]+)', output)
                                match_size = re.search(r'W\s+([-\d.]+)\s+H\s+([-\d.]+)\s+D\s+([-\d.]+)', output)
                                if match_center and match_size:
                                    st.session_state.cx, st.session_state.cy, st.session_state.cz = map(float, match_center.groups())
                                    st.session_state.sx, st.session_state.sy, st.session_state.sz = map(float, match_size.groups())
                                    st.rerun() 
                            else:
                                st.error("Erro interno no LaBOX ao processar o ligante.")
                    except Exception as e: st.error(f"Erro: {e}")

    else:
        with col_box1:
            box_input_pdb_blind = st.text_input("Arquivo PDB do Receptor Inteiro:", value=st.session_state.rec_pdb_final)
            if st.button("Calcular Bounding Box Global (LaBOX)"):
                if not os.path.exists(box_input_pdb_blind):
                    st.error("Erro: Arquivo PDB não encontrado.")
                else:
                    try:
                        with st.spinner("Mapeando limites estruturais globais com LaBOX..."):
                            if not os.path.exists("LaBOX.py"):
                                r_labox = requests.get("https://raw.githubusercontent.com/RyanZR/LaBOX/main/LaBOX.py")
                                with open("LaBOX.py", "w") as f: f.write(r_labox.text)
                            
                            res_labox = subprocess.run([sys.executable, "LaBOX.py", "-l", box_input_pdb_blind, "-c"], capture_output=True, text=True)
                            
                            if res_labox.returncode == 0:
                                output = res_labox.stdout
                                match_center = re.search(r'X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z\s+([-\d.]+)', output)
                                match_size = re.search(r'W\s+([-\d.]+)\s+H\s+([-\d.]+)\s+D\s+([-\d.]+)', output)
                                if match_center and match_size:
                                    st.session_state.cx, st.session_state.cy, st.session_state.cz = map(float, match_center.groups())
                                    st.session_state.sx, st.session_state.sy, st.session_state.sz = map(float, match_size.groups())
                                    st.rerun() 
                            else:
                                st.error("Erro interno no LaBOX ao processar o receptor.")
                    except Exception as e: st.error(f"Erro: {e}")

    with col_box2:
        st.markdown("### Coordenadas Dinâmicas (Å)")
        c_x, c_y, c_z = st.columns(3)
        cx = c_x.number_input("Center X", key='cx', step=0.1, value=st.session_state.cx)
        cy = c_y.number_input("Center Y", key='cy', step=0.1, value=st.session_state.cy)
        cz = c_z.number_input("Center Z", key='cz', step=0.1, value=st.session_state.cz)
        sx = c_x.number_input("Size W", key='sx', step=0.1, value=st.session_state.sx)
        sy = c_y.number_input("Size H", key='sy', step=0.1, value=st.session_state.sy)
        sz = c_z.number_input("Size D", key='sz', step=0.1, value=st.session_state.sz)

        if st.checkbox("Visualizar Caixa 3D (Manter ativado)"):
            if os.path.exists(st.session_state.rec_pdb_final):
                with open(st.session_state.rec_pdb_final, 'r') as f:
                    viewer = py3Dmol.view(width=500, height=400)
                    viewer.addModel(f.read(), "pdb")
                    viewer.setStyle({"cartoon": {'color':'cyan'}})
                    viewer.addBox({'center': {'x': cx, 'y': cy, 'z': cz}, 'dimensions': {'w': sx, 'h': sy, 'd': sz}, 'color': 'red', 'wireframe': True})
                    viewer.zoomTo()
                    html(viewer._make_html(), width=500, height=400)

# ==========================================
# ABA 5: Configuração Vina
# ==========================================
with tab_vina:
    st.header("5. Geração de Protocolo do Vina")
    
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        vina_receptor = st.text_input("Receptor Alvo (.pdbqt):", value=st.session_state.rec_final)
        if st.session_state.vs_mode:
            st.info("🔹 Modo Triagem Virtual Ativado: O arquivo de configuração não exige o nome do ligante.")
        else:
            vina_ligante = st.text_input("Ligante Teste (.pdbqt):", value=st.session_state.lig_final)
        
        vina_config_name = st.text_input("Salvar job como:", value="config.txt")
    with col_conf2:
        vina_exhaustiveness = st.number_input("Poder Computacional (Exhaustiveness):", min_value=1, value=24)
        
        max_cpus = multiprocessing.cpu_count()
        st.success(f"⚡ Autodetecção Vina: O algoritmo alocará automaticamente os {max_cpus} núcleos lógicos desta máquina.")

    if st.button("Gerar Ordem de Cálculo 'config.txt'", type="primary"):
        if st.session_state.vs_mode:
            config_content = f"receptor = {vina_receptor}\n\ncenter_x = {st.session_state.cx}\ncenter_y = {st.session_state.cy}\ncenter_z = {st.session_state.cz}\n\nsize_x = {st.session_state.sx}\nsize_y = {st.session_state.sy}\nsize_z = {st.session_state.sz}\n\nexhaustiveness = {vina_exhaustiveness}\ncpu = {max_cpus}\n"
        else:
            config_content = f"receptor = {vina_receptor}\nligand = {vina_ligante}\n\ncenter_x = {st.session_state.cx}\ncenter_y = {st.session_state.cy}\ncenter_z = {st.session_state.cz}\n\nsize_x = {st.session_state.sx}\nsize_y = {st.session_state.sy}\nsize_z = {st.session_state.sz}\n\nexhaustiveness = {vina_exhaustiveness}\ncpu = {max_cpus}\n"
            
        with open(vina_config_name, "w") as f: f.write(config_content)
        st.success(f"✅ Arquivo de configuração compilado com sucesso.")
        with open(vina_config_name, "r") as f: st.code(f.read(), language="ini")

# ==========================================
# ABA 6: Execução do Docking Molecular (C/ Trava de Fila)
# ==========================================
with tab_executar:
    st.header("6. Simulação Termodinâmica em Triplicata")
    st.info("💡 **Atenção Pesquisador:** Seguindo o rigor científico para publicações, o algoritmo Vina será executado 3 vezes independentes para garantir que o resultado encontrado é o mínimo global verdadeiro (redução do viés estocástico).")
        
    vina_exe = "vina_1.2.7_linux_x86_64"
    config_file_exec = st.text_input("Configuração a ser lida:", value="config.txt")
    
    if st.session_state.vs_mode:
        nome_rec_base = st.session_state.rec_final.replace('.pdbqt', '')
        data_atual = datetime.now().strftime("%Y%m%d_%H%M")
        dir_name_padrao = f"Screening_results_{nome_rec_base}_{data_atual}"
        
        st.warning("⚠️ **Modo Triagem Virtual:** A simulação será executada em LOTE (Batch) contra todos os arquivos na pasta `Ligantes/`.")
        output_dir_input = st.text_input("Diretório de Saída Base:", value=dir_name_padrao)
        
        if st.button("▶️ Iniciar Triagem HTVS em Triplicata", type="primary"):
            if not os.path.exists(config_file_exec):
                st.error("Arquivo de configuração não encontrado.")
            elif is_server_busy():
                st.error("⏳ **Servidor Ocupado:** Outro pesquisador está executando um cálculo termodinâmico no momento. Por favor, aguarde alguns minutos para não sobrecarregar a nuvem gratuita.")
            else:
                try:
                    lock_server() # Bloqueia o servidor
                    
                    if not os.path.exists(vina_exe):
                        st.info("Adquirindo binários Vina (Linux)...")
                        r_vina = requests.get(f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{vina_exe}")
                        with open(vina_exe, 'wb') as f: f.write(r_vina.content)
                        os.chmod(vina_exe, 0o755)
                    
                    st.session_state.vs_results_dir = output_dir_input
                    progress_bar = st.progress(0, text="Iniciando cálculos termodinâmicos...")
                    
                    with st.spinner(f"Rodando biblioteca de compostos em 3 corridas independentes (Multicore)..."):
                        log_outputs = ""
                        for rep in range(1, 4):
                            rep_dir = os.path.join(output_dir_input, f"rep{rep}")
                            os.makedirs(rep_dir, exist_ok=True)
                            
                            cmd_batch = f"./{vina_exe} --config {config_file_exec} --batch Ligantes/*.pdbqt --dir {rep_dir}"
                            res_vs = subprocess.run(cmd_batch, shell=True, capture_output=True, text=True)
                            log_outputs += f"\n--- REPLICATA {rep} ---\n" + res_vs.stdout
                            
                            progress_bar.progress(int((rep / 3.0) * 100), text=f"Calculando Replicata {rep} de 3...")
                            
                        st.session_state.vina_log_output = log_outputs 
                        progress_bar.empty()
                        st.success(f"🎉 Triagem Virtual em Triplicata concluída! Os resultados foram separados nas pastas `rep1`, `rep2` e `rep3` dentro de `{output_dir_input}/`")
                except Exception as e: 
                    st.error(f"Erro do sistema: {e}")
                finally:
                    unlock_server() # Libera o servidor mesmo se der erro

    else:
        # Modo Individual em Triplicata
        nome_lig_base = st.session_state.lig_final.replace('.pdbqt', '')
        nome_rec_base = st.session_state.rec_final.replace('.pdbqt', '')
        nome_saida_padrao = f"resultado_docking_{nome_lig_base}_{nome_rec_base}"
        
        output_pdbqt_base = st.text_input("Nome base para as Poses:", value=nome_saida_padrao)
        
        if st.button("▶️ Iniciar Docking em Triplicata", type="primary"):
            if not os.path.exists(config_file_exec):
                st.error("⚠️ Arquivo de configuração não encontrado.")
            elif is_server_busy():
                st.error("⏳ **Servidor Ocupado:** Outro pesquisador está executando um cálculo termodinâmico no momento. Por favor, aguarde alguns minutos para não sobrecarregar a nuvem gratuita.")
            else:
                try:
                    lock_server() # Bloqueia o servidor
                    
                    if not os.path.exists(vina_exe):
                        r_vina = requests.get(f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{vina_exe}")
                        with open(vina_exe, 'wb') as f: f.write(r_vina.content)
                        os.chmod(vina_exe, 0o755)
                    
                    progress_bar = st.progress(0, text="Calculando energias de interação...")
                    log_outputs = ""
                    with st.spinner(f"Computando interações termodinâmicas usando {multiprocessing.cpu_count()} núcleos (3x vezes)..."):
                        for rep in range(1, 4):
                            out_rep = f"{output_pdbqt_base}_rep{rep}.pdbqt"
                            res_vina = subprocess.run([f"./{vina_exe}", "--config", config_file_exec, "--out", out_rep], capture_output=True, text=True)
                            log_outputs += f"\n--- REPLICATA {rep} ---\n" + res_vina.stdout
                            
                            progress_bar.progress(int((rep / 3.0) * 100), text=f"Replicata {rep} de 3 concluída...")
                        
                        progress_bar.empty()
                        st.session_state.vina_log_output = log_outputs 
                        st.session_state.single_result_base = output_pdbqt_base
                        st.success("Simulação em triplicata concluída! Vá para a Aba 7 para ver as médias.")
                except Exception as e: 
                    st.error(f"Erro: {e}")
                finally:
                    unlock_server() # Libera o servidor mesmo se der erro
                
    # --- VISUALIZAÇÃO DE LOG PERSISTENTE ---
    if st.session_state.vina_log_output:
        st.divider()
        if st.checkbox("Exibir Log de Execução do AutoDock Vina (Terminal)"):
            st.text_area("Saída Bruta do Algoritmo Termodinâmico:", value=st.session_state.vina_log_output, height=400)

# ==========================================
# ABA 7: Análise Químico-Estrutural
# ==========================================
with tab_visualizar:
    st.header("7. Análise de Resultados e Síntese (Triplicata)")
    
    st.divider()
    
    st.subheader("📈 Tabela Termodinâmica Global")

    if st.session_state.vs_mode:
        # TABELA HTVS
        if not st.session_state.get('vs_results_dir') or not os.path.exists(st.session_state.vs_results_dir):
            st.warning("Execute o Docking em lote (Aba 6) primeiro para carregar a tabela estatística.")
        else:
            with st.spinner("Lendo resultados e calculando médias estocásticas..."):
                ligand_files = glob.glob(os.path.join(st.session_state.vs_results_dir, "rep1", "*.pdbqt"))
                data_results = []
                
                for f in ligand_files:
                    basename = os.path.basename(f)
                    rep1_path = os.path.join(st.session_state.vs_results_dir, "rep1", basename)
                    rep2_path = os.path.join(st.session_state.vs_results_dir, "rep2", basename)
                    rep3_path = os.path.join(st.session_state.vs_results_dir, "rep3", basename)
                    
                    v1 = get_vina_affinity(rep1_path)
                    v2 = get_vina_affinity(rep2_path)
                    v3 = get_vina_affinity(rep3_path)
                    
                    vals = [v for v in [v1, v2, v3] if not np.isnan(v)]
                    mean_val = round(np.mean(vals), 2) if vals else np.nan
                    std_val = round(np.std(vals), 2) if len(vals) > 1 else 0.0
                    
                    nome_limpo = basename.replace('_out.pdbqt', '').replace('.pdbqt', '')
                    
                    data_results.append({
                        "Ligante": nome_limpo,
                        "Média (kcal/mol)": mean_val,
                        "Desvio Padrão": std_val,
                        "Rep 1": v1, "Rep 2": v2, "Rep 3": v3
                    })
                
                if data_results:
                    df_results = pd.DataFrame(data_results).sort_values(by="Média (kcal/mol)")
                    st.dataframe(df_results, use_container_width=True, hide_index=True)
                    
                    # --- GRÁFICO INTERATIVO DE AFINIDADE E DESVIO PADRÃO ---
                    st.markdown("### 📊 Gráfico de Afinidade Termodinâmica")
                    fig = px.bar(df_results, x="Ligante", y="Média (kcal/mol)", error_y="Desvio Padrão", 
                                 title="Energia Livre de Gibbs (ΔG) por Ligante",
                                 labels={"Média (kcal/mol)": "Afinidade (kcal/mol)", "Ligante": "Molécula"},
                                 color="Média (kcal/mol)", color_continuous_scale="Viridis")
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning("Não foi possível ler os arquivos de afinidade gerados.")

            st.divider()
            
            # VISUALIZAÇÃO E SÍNTESE HTVS
            col_vis1, col_vis2 = st.columns([1, 2])
            with col_vis1:
                st.markdown("**Síntese de PDB Interativo**")
                
                all_poses = glob.glob(os.path.join(st.session_state.vs_results_dir, "rep*", "*.pdbqt"))
                opcoes_poses = [f"{os.path.basename(os.path.dirname(p))}/{os.path.basename(p)}" for p in all_poses]
                
                if opcoes_poses:
                    selected_pose_rel = st.selectbox("Escolha um resultado exato para visualizar e sintetizar:", opcoes_poses)
                    
                    if st.button("Sintetizar Ligante Selecionado"):
                        full_path_selected = os.path.join(st.session_state.vs_results_dir, selected_pose_rel)
                        with st.spinner("Fundindo matriz..."):
                            best_pose_lines = []
                            in_model_1 = False
                            with open(full_path_selected, 'r') as f:
                                for linha in f:
                                    if linha.startswith("MODEL 1"): in_model_1 = True
                                    if in_model_1: best_pose_lines.append(linha)
                                    if linha.startswith("ENDMDL") and in_model_1: break
                            
                            with open("melhor_pose.pdbqt", "w") as f: f.writelines(best_pose_lines)
                            subprocess.run(["obabel", "-ipdbqt", "melhor_pose.pdbqt", "-opdb", "-O", "melhor_pose.pdb"])
                            
                            with open(st.session_state.rec_pdb_final, 'r') as f: rec_lines = [l for l in f.readlines() if not l.startswith("END")]
                            
                            if os.path.exists("melhor_pose.pdb"):
                                lig_lines = []
                                with open("melhor_pose.pdb", 'r') as f:
                                    for l in f.readlines():
                                        if l.startswith("ATOM") or l.startswith("HETATM"):
                                            linha_fixa = "HETATM" + l[6:17] + "UNL" + l[20:]
                                            lig_lines.append(linha_fixa)
                                            
                                complex_str = "".join(rec_lines + lig_lines + ["END\n"])
                                
                                nome_comp = selected_pose_rel.replace('/', '_').replace('.pdbqt', '').replace('_out', '')
                                temp_complex = f"complexo_{nome_comp}.pdb"
                                with open(temp_complex, "w") as f: f.write(complex_str)
                                
                                st.session_state.complex_generated = True
                                st.session_state.complex_file = temp_complex
                                st.session_state.rec_str = "".join(rec_lines)
                                st.session_state.lig_str = "".join(lig_lines)
                
                st.write("---")
                st.markdown("**Pós-Processamento em Lote**")
                if st.button("Sintetizar PDB da Replicata 1 para TODOS os Ligantes"):
                    out_complexes_dir = f"{st.session_state.vs_results_dir}_ComplexosPDB"
                    os.makedirs(out_complexes_dir, exist_ok=True)
                    with st.spinner("Criando complexos PDB baseados na Rep 1..."):
                        with open(st.session_state.rec_pdb_final, 'r') as f: rec_lines = [l for l in f.readlines() if not l.startswith("END")]
                        
                        for p_file in ligand_files:
                            clean_name = os.path.basename(p_file).replace('.pdbqt', '').replace('_out', '')
                            comp_out_path = os.path.join(out_complexes_dir, f"complexo_{clean_name}_rep1.pdb")
                            
                            best_pose = []
                            in_model_1 = False
                            with open(p_file, 'r') as f:
                                for line in f:
                                    if line.startswith("MODEL 1"): in_model_1 = True
                                    if in_model_1: best_pose.append(line)
                                    if line.startswith("ENDMDL") and in_model_1: break
                            
                            with open("temp_vs_pose.pdbqt", "w") as f: f.writelines(best_pose)
                            subprocess.run(["obabel", "-ipdbqt", "temp_vs_pose.pdbqt", "-opdb", "-O", "temp_vs_pose.pdb"])
                            
                            if os.path.exists("temp_vs_pose.pdb"):
                                lig_lines = []
                                with open("temp_vs_pose.pdb", 'r') as f:
                                    for l in f.readlines():
                                        if l.startswith("ATOM") or l.startswith("HETATM"):
                                            lig_lines.append("HETATM" + l[6:17] + "UNL" + l[20:])
                                complex_str = "".join(rec_lines + lig_lines + ["END\n"])
                                with open(comp_out_path, "w") as f: f.write(complex_str)
                        st.success(f"✅ Todos os complexos PDB foram salvos em: `{out_complexes_dir}`")

            if st.session_state.get('complex_generated', False):
                with col_vis1:
                    with open(st.session_state.complex_file, "r") as f:
                        st.download_button("⬇️ Baixar PDB do Complexo (Interativo)", data=f.read(), file_name=st.session_state.complex_file, mime="text/plain", type="primary")

            with col_vis2:
                if st.session_state.get('complex_generated', False):
                    viewer_comp = py3Dmol.view(width=700, height=450)
                    viewer_comp.addModel(st.session_state.rec_str, "pdb")
                    viewer_comp.setStyle({'model': 0}, {"cartoon": {'color': 'spectrum'}})
                    viewer_comp.addModel(st.session_state.lig_str, "pdb")
                    viewer_comp.setStyle({'model': 1}, {"stick": {'colorscheme': 'magentaCarbon', 'radius': 0.15}})
                    viewer_comp.zoomTo({'model': 1})
                    html(viewer_comp._make_html(), width=700, height=450)

    else:
        # TABELA E VISUALIZAÇÃO SINGLE
        base_name = st.session_state.get('single_result_base', '')
        if not base_name or not os.path.exists(f"{base_name}_rep1.pdbqt"):
            st.warning("Execute o Docking na Aba 6 primeiro para carregar a tabela estatística.")
        else:
            with st.spinner("Compilando estatística termodinâmica..."):
                v1 = get_vina_affinity(f"{base_name}_rep1.pdbqt")
                v2 = get_vina_affinity(f"{base_name}_rep2.pdbqt")
                v3 = get_vina_affinity(f"{base_name}_rep3.pdbqt")
                
                vals = [v for v in [v1, v2, v3] if not np.isnan(v)]
                mean_val = round(np.mean(vals), 2) if vals else np.nan
                std_val = round(np.std(vals), 2) if len(vals) > 1 else 0.0
                
                df_single = pd.DataFrame([{
                    "Ligante (Alvo Único)": os.path.basename(base_name),
                    "Média (kcal/mol)": mean_val,
                    "Desvio Padrão": std_val,
                    "Rep 1": v1, "Rep 2": v2, "Rep 3": v3
                }])
                st.dataframe(df_single, use_container_width=True, hide_index=True)
                
                # --- GRÁFICO INTERATIVO DE AFINIDADE E DESVIO PADRÃO (SINGLE) ---
                st.markdown("### 📊 Gráfico de Afinidade Termodinâmica")
                fig = px.bar(df_single, x="Ligante (Alvo Único)", y="Média (kcal/mol)", error_y="Desvio Padrão", 
                             title="Energia Livre de Gibbs (ΔG) do Complexo",
                             labels={"Média (kcal/mol)": "Afinidade (kcal/mol)", "Ligante (Alvo Único)": "Molécula"},
                             color="Média (kcal/mol)", color_continuous_scale="Viridis")
                st.plotly_chart(fig, use_container_width=True)

            st.divider()

            col_vis1, col_vis2 = st.columns([1, 2])
            
            with col_vis1:
                st.markdown("**Síntese de PDB Interativo**")
                rep_escolhida = st.selectbox("Escolha a Replicata para sintetizar o complexo:", ["rep1", "rep2", "rep3"])
                
                if st.button("Sintetizar Complexo PDB"):
                    pose_alvo = f"{base_name}_{rep_escolhida}.pdbqt"
                    if os.path.exists(st.session_state.rec_pdb_final) and os.path.exists(pose_alvo):
                        with st.spinner("Fundindo matriz atômica..."):
                            best_pose_lines = []
                            in_model_1 = False
                            with open(pose_alvo, 'r') as f:
                                for linha in f:
                                    if linha.startswith("MODEL 1"): in_model_1 = True
                                    if in_model_1: best_pose_lines.append(linha)
                                    if linha.startswith("ENDMDL") and in_model_1: break
                            with open("melhor_pose.pdbqt", "w") as f: f.writelines(best_pose_lines)
                            subprocess.run(["obabel", "-ipdbqt", "melhor_pose.pdbqt", "-opdb", "-O", "melhor_pose.pdb"])
                            
                            with open(st.session_state.rec_pdb_final, 'r') as f: rec_lines = [l for l in f.readlines() if not l.startswith("END")]
                            if os.path.exists("melhor_pose.pdb"):
                                lig_lines = []
                                with open("melhor_pose.pdb", 'r') as f:
                                    for l in f.readlines():
                                        if l.startswith("ATOM") or l.startswith("HETATM"):
                                            lig_lines.append("HETATM" + l[6:17] + "UNL" + l[20:])
                                            
                                complex_str = "".join(rec_lines + lig_lines + ["END\n"])
                                complexo_out = f"complexo_{os.path.basename(base_name)}_{rep_escolhida}.pdb"
                                with open(complexo_out, "w") as f: f.write(complex_str)
                                st.session_state.complex_generated = True
                                st.session_state.complex_file = complexo_out
                                st.session_state.rec_str = "".join(rec_lines)
                                st.session_state.lig_str = "".join(lig_lines)
                                st.success("Modelo Holo consolidado com nomenclatura farmacofórica padrão (HETATM/UNL)!")
                    else:
                        st.error("Arquivos bases não encontrados.")
            
                if st.session_state.get('complex_generated', False):
                    with open(st.session_state.complex_file, "r") as f:
                        st.download_button("⬇️ Baixar PDB do Complexo Interativo", data=f.read(), file_name=st.session_state.complex_file, mime="text/plain", type="primary")

            with col_vis2:
                if st.session_state.get('complex_generated', False):
                    viewer_comp = py3Dmol.view(width=700, height=450)
                    viewer_comp.addModel(st.session_state.rec_str, "pdb")
                    viewer_comp.setStyle({'model': 0}, {"cartoon": {'color': 'spectrum'}})
                    viewer_comp.addModel(st.session_state.lig_str, "pdb")
                    viewer_comp.setStyle({'model': 1}, {"stick": {'colorscheme': 'magentaCarbon', 'radius': 0.15}})
                    viewer_comp.zoomTo({'model': 1})
                    html(viewer_comp._make_html(), width=700, height=450)

    # ---------------------------------------------------------
    # MÓDULO: CÁLCULO DE RMSD PARA RE-DOCKING
    # ---------------------------------------------------------
    if st.session_state.get('redocking_mode', False) and st.session_state.get('complex_generated', False):
        st.divider()
        st.subheader("🎯 Validação de Re-Docking (Cálculo de RMSD)")
        st.write("Verifica matematicamente o desvio da sua simulação comparado à posição cristalográfica original.")
        
        if st.button("📐 Calcular RMSD de Alinhamento"):
            try:
                cmd_rmsd = ["obrms", st.session_state.extracted_lig_pdb, "melhor_pose.pdb"]
                res_rmsd = subprocess.run(cmd_rmsd, capture_output=True, text=True)
                
                rmsd_val = None
                if res_rmsd.returncode == 0 and "RMSD" in res_rmsd.stdout:
                    match = re.search(r'RMSD.*?([\d\.]+)', res_rmsd.stdout)
                    if match:
                        rmsd_val = float(match.group(1))
                        
                if rmsd_val is None:
                    # Método de Contingência Matemático em Python nativo
                    parser = PDBParser(QUIET=True)
                    ref_st = parser.get_structure('ref', st.session_state.extracted_lig_pdb)
                    pose_st = parser.get_structure('pose', "melhor_pose.pdb")
                    
                    ref_atoms = sorted([a for a in ref_st.get_atoms() if a.element != 'H'], key=lambda x: x.name)
                    pose_atoms = sorted([a for a in pose_st.get_atoms() if a.element != 'H'], key=lambda x: x.name)
                    
                    if len(ref_atoms) == len(pose_atoms) and len(ref_atoms) > 0:
                        diff = np.array([a.coord for a in ref_atoms]) - np.array([a.coord for a in pose_atoms])
                        rmsd_val = round(np.sqrt(np.mean(np.sum(diff**2, axis=1))), 3)

                if rmsd_val is not None:
                    if rmsd_val <= 2.0:
                        st.success(f"✅ **Protocolo Validado com Excelência!** \nO algoritmo reproduziu o complexo nativo com alta fidelidade.\n\n**RMSD = {rmsd_val} Å**")
                        st.balloons()
                    elif rmsd_val <= 3.0:
                        st.warning(f"⚠️ **Validação Aceitável:** \nO encaixe foi próximo, mas há desvios estruturais.\n\n**RMSD = {rmsd_val} Å**")
                    else:
                        st.error(f"❌ **Protocolo Inválido:** \nA simulação errou a pose do cristal. Sugere-se revisar a conformação original ou redimensionar a Grid Box.\n\n**RMSD = {rmsd_val} Å**")
                else:
                    st.error("Falha no cálculo do desvio: O OpenBabel (obrms) não está disponível e ocorreu uma assimetria no número de átomos pesados entre a estrutura de referência e a pose do Vina.")
            except Exception as e:
                st.error(f"Erro matemático durante a sobreposição: {e}")

    st.divider()

    st.subheader("📊 Diagramas de Interação Químico-Estrutural (Para Artigos)")
    st.info("Para gerar mapas 2D e 3D detalhados de interações (Pontes de Hidrogênio, Pi-Stacking, Contatos de Van der Waals) adequados para publicação acadêmica, recomendamos fortemente a exportação para as ferramentas padrão-ouro da indústria:")
    
    st.markdown("""
    1. **Discovery Studio Visualizer** (BIOVIA / Dassault Systèmes)
    2. **LigPlot+** (EMBL-EBI)
    3. **Schrödinger Maestro** (Academic Version)
    
    **Como utilizar:** Basta baixar os arquivos **PDB do Complexo Interativo** gerados nesta aba e abri-los diretamente em qualquer um destes softwares. Eles identificarão automaticamente a proteína e o fármaco (HETATM) para desenhar o diagrama.
    """)

# ==========================================
# ABA 8: Referências e Algoritmos
# ==========================================
with tab_referencias:
    st.header("📚 Referências Bibliográficas e Algoritmos")
    st.markdown("""
    Este ambiente computacional é embasado nos mais rigorosos algoritmos de biologia estrutural. Para aprofundar-se nos estudos de Iniciação Científica, consulte as referências abaixo:

    ### Livros de Química Farmacêutica e Medicinal (Essenciais)
    * **Barreiro, E. J., & Fraga, C. A. M. (2015).** *Química Medicinal: As Bases Farmacológicas da Ação dos Fármacos*. 3ª Ed. Artmed. *(Referência nacional absoluta e indispensável para o entendimento das bases moleculares e desenho de fármacos no Brasil).*
    * **Patrick, G. L. (2013).** *An Introduction to Medicinal Chemistry*. 5th Ed. Oxford University Press.
    * **Silverman, R. B., & Holladay, M. W. (2014).** *The Organic Chemistry of Drug Design and Drug Action*. 3rd Ed. Academic Press.

    ### Softwares e Algoritmos Implementados
    * **AutoDock Vina:** Trott, O., & Olson, A. J. (2010). AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. *Journal of Computational Chemistry*, 31(2), 455-461.
    * **OpenBabel:** O'Boyle, N. M., et al. (2011). Open Babel: An open chemical toolbox. *Journal of Cheminformatics*, 2(1), 5.
    * **RDKit:** Open-source cheminformatics. *(Utilizado para topologia química bidimensional).*
    * **PDBFixer / OpenMM:** Eastman, P., et al. (2017). OpenMM 7: Rapid development of high performance algorithms for molecular dynamics. *PLoS computational biology*, 13(7), e1005659.
    * **Biopython:** Cock, P. J., et al. (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics*, 25(11), 1422-1423.
    """)
