import streamlit as st
import os
import subprocess
import sys
import requests
import re
import glob
from datetime import datetime
import numpy as np
import pandas as pd

# Importações específicas (o Streamlit Cloud instalará via requirements.txt)
import py3Dmol
from streamlit.components.v1 import html
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
from Bio.PDB import PDBParser

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
        if os.path.exists("logo_ufam.jpg"):
            st.image("logo_ufam.jpg", use_container_width=True)
    except Exception:
        pass
    
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

# Inicialização das variáveis de memória
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

# Abas
tab_install, tab_receptor, tab_ligante, tab_gridbox, tab_vina, tab_executar, tab_visualizar, tab_referencias = st.tabs([
    "🛠️ 1. Ambiente", "🧬 2. Receptor", "💊 3. Ligante", "📦 4. Grid Box", "⚙️ 5. Vina Config", "🚀 6. Docking", "👁️ 7. Análise", "📚 8. Referências"
])

# ==========================================
# ABA 1: Instalação de Dependências
# ==========================================
with tab_install:
    st.header("1. Verificação do Ambiente Computacional")
    st.success("✅ **Sistema Operante:** O Streamlit Cloud configurou automaticamente o OpenBabel e as bibliotecas via `packages.txt` e `requirements.txt`.")
    
    with st.expander("📚 Fundamentos: O que é o Docking Molecular?", expanded=True):
        st.markdown("""
        O **Docking Molecular** é uma técnica fundamental no Planejamento de Fármacos Baseado em Estrutura (SBDD). O objetivo computacional é prever a conformação tridimensional preferencial (a **Pose**) de uma molécula pequena (fármaco/ligante) quando ligada a uma macromolécula (receptor/proteína), formando um complexo estável.
        """)

# ==========================================
# ABA 2: Preparação do Receptor
# ==========================================
with tab_receptor:
    st.header("2. O Alvo Farmacológico (Receptor)")
    
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
                st.error("Falha ao buscar PDB.")

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
            except Exception as e: st.error(f"Erro: {e}")

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
            except Exception as e: st.error(f"Erro: {e}")

# ==========================================
# ABA 3: Preparação do Ligante
# ==========================================
with tab_ligante:
    st.header("3. Preparação do(s) Fármaco(s)")
    
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
                if st.button("2. Minimizar (3D) e Gerar PDBQT", type="primary"):
                    mol2_file = f"{st.session_state.nome_ligante_salvar}.mol2"
                    pdbqt_file = f"{st.session_state.nome_ligante_salvar}.pdbqt"
                    with st.spinner("Calculando conformação de menor energia (MMFF94)..."):
                        subprocess.run(["obabel", f"-:{st.session_state.smiles}", "-O", mol2_file, "--gen3d"], capture_output=True)
                        if os.path.exists(mol2_file):
                            st.session_state.mol2_file_path = mol2_file 
                            subprocess.run(["obabel", "-imol2", mol2_file, "-opdbqt", "-O", pdbqt_file, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                            st.session_state.lig_final = pdbqt_file 
                            st.success(f"Ligante 3D otimizado com cargas de Gasteiger: '{pdbqt_file}'")

        with col_2d:
            st.markdown("### Topologia 2D")
            if 'img_2d' in st.session_state: st.image(st.session_state.img_2d)

        with col_3d:
            st.markdown("### Geometria Espacial 3D")
            if 'mol2_file_path' in st.session_state and os.path.exists(st.session_state.mol2_file_path):
                with open(st.session_state.mol2_file_path, 'r') as f:
                    viewer_lig = py3Dmol.view(width=300, height=300)
                    viewer_lig.addModel(f.read(), "mol2")
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
                    st.success(f"Matriz extraída com sucesso! Salvo como: {ext_pdbqt}")
            else:
                st.warning("Nenhum ligante orgânico detectado.")
    
    else:
        st.session_state.redocking_mode = False
        st.session_state.vs_mode = True
        st.info("📦 Módulo de High-Throughput Virtual Screening (HTVS)")
        uploaded_files = st.file_uploader("Arquivos de Biblioteca de Fármacos", type=['sdf', 'mol2', 'pdb'], accept_multiple_files=True)
        
        if uploaded_files:
            os.makedirs("Ligantes_temp", exist_ok=True)
            os.makedirs("Ligantes", exist_ok=True)
            
            if st.button("Processar, Otimizar e Converter para PDBQT", type="primary"):
                for f in glob.glob("Ligantes_temp/*"): os.remove(f)
                for f in glob.glob("Ligantes/*.pdbqt"): os.remove(f)
                
                temp_paths = []
                for uf in uploaded_files:
                    t_path = os.path.join("Ligantes_temp", sanitize_filename(uf.name))
                    with open(t_path, "wb") as f: f.write(uf.getbuffer())
                    temp_paths.append(t_path)
                
                with st.spinner("Minimizando em 3D e convertendo para PDBQT (Pode demorar)..."):
                    for t_path in temp_paths:
                        base_name = os.path.splitext(os.path.basename(t_path))[0]
                        out_mol2_prefix = f"Ligantes_temp/{base_name}_.mol2"
                        subprocess.run(["obabel", t_path, "-omol2", "-O", out_mol2_prefix, "-m", "--gen3d", "-e"], capture_output=True, text=True)
                        
                    mol2_files = glob.glob("Ligantes_temp/*.mol2")
                    for m2 in mol2_files:
                        base_m2 = os.path.basename(m2).replace(".mol2", "")
                        out_pdbqt = f"Ligantes/{base_m2}.pdbqt"
                        subprocess.run(["obabel", "-imol2", m2, "-opdbqt", "-O", out_pdbqt, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                    
                    qtd_gerados = len(glob.glob("Ligantes/*.pdbqt"))
                    if qtd_gerados > 0:
                        st.success(f"🎉 {qtd_gerados} moléculas preparadas para PDBQT!")
                        st.session_state.lig_final = "Múltiplos Ligantes"
                    else:
                        st.error("Nenhuma molécula estruturalmente válida pôde ser extraída.")

# ==========================================
# ABA 4: Grid Box
# ==========================================
with tab_gridbox:
    st.header("4. Mapeamento do Espaço de Busca (Grid Box)")
    tipo_docking = st.radio("Estratégia:", ["🎯 Site-Directed Docking", "🌍 Blind Docking"])
    
    col_box1, col_box2 = st.columns([1.2, 1])
    
    with col_box1:
        if "Site-Directed" in tipo_docking:
            box_input_pdb = st.text_input("Ligante Referência:", value=st.session_state.extracted_lig_pdb if st.session_state.redocking_mode else "ligante.pdb")
            if st.button("Calcular Sítio (LaBOX)"):
                if os.path.exists(box_input_pdb):
                    with st.spinner("Mapeando..."):
                        if not os.path.exists("LaBOX.py"):
                            r_labox = requests.get("https://raw.githubusercontent.com/RyanZR/LaBOX/main/LaBOX.py")
                            with open("LaBOX.py", "w") as f: f.write(r_labox.text)
                        
                        res = subprocess.run([sys.executable, "LaBOX.py", "-l", box_input_pdb, "-c"], capture_output=True, text=True)
                        match_center = re.search(r'X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z\s+([-\d.]+)', res.stdout)
                        match_size = re.search(r'W\s+([-\d.]+)\s+H\s+([-\d.]+)\s+D\s+([-\d.]+)', res.stdout)
                        if match_center and match_size:
                            st.session_state.cx, st.session_state.cy, st.session_state.cz = map(float, match_center.groups())
                            st.session_state.sx, st.session_state.sy, st.session_state.sz = map(float, match_size.groups())
                            st.rerun() 
                else:
                    st.error("Arquivo referência não encontrado.")
        else:
            box_input_pdb_blind = st.text_input("Receptor Inteiro:", value=st.session_state.rec_pdb_final)
            if st.button("Calcular Centro de Massa (Biopython)"):
                if os.path.exists(box_input_pdb_blind):
                    parser = PDBParser(QUIET=True)
                    structure = parser.get_structure('receptor', box_input_pdb_blind)
                    center = structure.center_of_mass()
                    st.session_state.cx, st.session_state.cy, st.session_state.cz = round(center[0],3), round(center[1],3), round(center[2],3)
                    coords = [atom.coord for atom in structure.get_atoms()]
                    min_x, max_x = min(c[0] for c in coords), max(c[0] for c in coords)
                    min_y, max_y = min(c[1] for c in coords), max(c[1] for c in coords)
                    min_z, max_z = min(c[2] for c in coords), max(c[2] for c in coords)
                    st.session_state.sx, st.session_state.sy, st.session_state.sz = round((max_x-min_x)+10,3), round((max_y-min_y)+10,3), round((max_z-min_z)+10,3)
                    st.rerun()

    with col_box2:
        st.markdown("### Coordenadas Dinâmicas (Å)")
        c_x, c_y, c_z = st.columns(3)
        cx = c_x.number_input("Center X", key='cx', step=0.1)
        cy = c_y.number_input("Center Y", key='cy', step=0.1)
        cz = c_z.number_input("Center Z", key='cz', step=0.1)
        sx = c_x.number_input("Size W", key='sx', step=0.1)
        sy = c_y.number_input("Size H", key='sy', step=0.1)
        sz = c_z.number_input("Size D", key='sz', step=0.1)

        if st.button("Validar Caixa"):
            if os.path.exists(st.session_state.rec_pdb_final):
                with open(st.session_state.rec_pdb_final, 'r') as f:
                    viewer = py3Dmol.view(width=400, height=300)
                    viewer.addModel(f.read(), "pdb")
                    viewer.setStyle({"cartoon": {'color':'cyan'}})
                    viewer.addBox({'center': {'x': cx, 'y': cy, 'z': cz}, 'dimensions': {'w': sx, 'h': sy, 'd': sz}, 'color': 'red', 'wireframe': True})
                    viewer.zoomTo()
                    html(viewer._make_html(), width=400, height=300)

# ==========================================
# ABA 5: Configuração Vina
# ==========================================
with tab_vina:
    st.header("5. Geração de Protocolo")
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        vina_receptor = st.text_input("Receptor Alvo (.pdbqt):", value=st.session_state.rec_final)
        if not st.session_state.vs_mode:
            vina_ligante = st.text_input("Ligante Teste (.pdbqt):", value=st.session_state.lig_final)
        vina_config_name = st.text_input("Salvar job como:", value="config.txt")
    with col_conf2:
        vina_exhaustiveness = st.number_input("Exhaustiveness:", min_value=1, value=24)

    if st.button("Gerar 'config.txt'", type="primary"):
        config_content = f"receptor = {vina_receptor}\n"
        if not st.session_state.vs_mode: config_content += f"ligand = {vina_ligante}\n"
        config_content += f"\ncenter_x = {st.session_state.cx}\ncenter_y = {st.session_state.cy}\ncenter_z = {st.session_state.cz}\n\nsize_x = {st.session_state.sx}\nsize_y = {st.session_state.sy}\nsize_z = {st.session_state.sz}\n\nexhaustiveness = {vina_exhaustiveness}\n"
        with open(vina_config_name, "w") as f: f.write(config_content)
        st.success("Configuração compilada.")

# ==========================================
# ABA 6: Execução
# ==========================================
with tab_executar:
    st.header("6. Simulação")
    vina_exe = "vina_1.2.7_linux_x86_64"
    config_file_exec = st.text_input("Configuração a ser lida:", value="config.txt")
    
    if st.button("▶️ Iniciar Docking (Triplicata)", type="primary"):
        if os.path.exists(config_file_exec):
            if not os.path.exists(vina_exe):
                st.info("Adquirindo binários Vina...")
                r_vina = requests.get(f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{vina_exe}")
                with open(vina_exe, 'wb') as f: f.write(r_vina.content)
                os.chmod(vina_exe, 0o755)
            
            progress_bar = st.progress(0)
            
            if st.session_state.vs_mode:
                output_dir = f"Screening_{datetime.now().strftime('%H%M')}"
                st.session_state.vs_results_dir = output_dir
                for rep in range(1, 4):
                    rep_dir = os.path.join(output_dir, f"rep{rep}")
                    os.makedirs(rep_dir, exist_ok=True)
                    subprocess.run(f"./{vina_exe} --config {config_file_exec} --batch Ligantes/*.pdbqt --dir {rep_dir}", shell=True)
                    progress_bar.progress(rep * 33)
                st.success(f"Triagem em Triplicata salva em {output_dir}")
            else:
                out_base = "resultado_docking"
                for rep in range(1, 4):
                    subprocess.run([f"./{vina_exe}", "--config", config_file_exec, "--out", f"{out_base}_rep{rep}.pdbqt"])
                    progress_bar.progress(rep * 33)
                st.session_state.single_result_base = out_base
                st.success("Simulação individual concluída!")
        else:
            st.error("Arquivo de configuração não encontrado.")

# ==========================================
# ABA 7: Análise (Simplificada para Nuvem)
# ==========================================
with tab_visualizar:
    st.header("7. Análise de Resultados")
    st.info("Para analisar as afinidades (kcal/mol), verifique os arquivos PDBQT gerados. (Interface em desenvolvimento para visualização web).")
