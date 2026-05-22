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
import signal
import traceback

# Configuração da página (COM ÍCONE PERSONALIZADO)
st.set_page_config(page_title="BioDockUfam", page_icon="🧬", layout="wide", initial_sidebar_state="expanded")

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
except ImportError as e:
    LIBS_INSTALADAS = False
    ERRO_IMPORT = str(e)

# ==========================================
# FUNÇÕES DE CONTROLE DE SERVIDOR (MULTIPROCESSOS)
# ==========================================
LOCK_DIR = "active_runs"
os.makedirs(LOCK_DIR, exist_ok=True)

def register_run(pid):
    """Registra um processo ativo."""
    with open(os.path.join(LOCK_DIR, f"{pid}.lock"), 'w') as f:
        f.write(str(time.time()))

def unregister_run(pid):
    """Remove o registro de um processo que finalizou."""
    try:
        os.remove(os.path.join(LOCK_DIR, f"{pid}.lock"))
    except OSError:
        pass

def get_active_runs_count():
    """Retorna o número de pessoas rodando o Vina simultaneamente."""
    current_time = time.time()
    count = 0
    for f in glob.glob(f"{LOCK_DIR}/*.lock"):
        # Proteção Anti-Zumbi: Se o lock tem mais de 45 minutos, considera morto e apaga
        if current_time - os.path.getmtime(f) > 2700:
            try:
                os.remove(f)
            except OSError:
                pass
        else:
            count += 1
    return count

def kill_all_processes():
    """Mata todos os processos Vina rodando e limpa os locks."""
    count = 0
    for f in glob.glob(f"{LOCK_DIR}/*.lock"):
        try:
            pid = int(os.path.basename(f).replace('.lock', ''))
            os.kill(pid, signal.SIGTERM) # Envia sinal de término seguro
            count += 1
        except OSError:
            pass # Processo já não existe
        finally:
            try:
                os.remove(f)
            except OSError:
                pass
                
    # Comando de segurança caso o PID não tenha sido pego corretamente
    try:
        subprocess.run(["pkill", "-f", "vina_1.2.7_linux_x86_64"], capture_output=True)
    except Exception:
        pass
        
    return count

# ==========================================
# FUNÇÕES AUXILIARES COM TRATEMENTO DE ERRO
# ==========================================
def get_ligands_from_pdb(pdb_file):
    ligands = set()
    filtros = ["HOH", "WAT", "DOD", "NA", "CL", "MG", "K", "SO4", "PO4", "EDO", "GOL", "FMT", "ACT"]
    try:
        if os.path.exists(pdb_file):
            with open(pdb_file, 'r') as f:
                for line in f:
                    if line.startswith("HETATM"):
                        res_name = line[17:20].strip()
                        if res_name not in filtros:
                            ligands.add(res_name)
    except Exception as e:
        st.error(f"Erro ao ler ligantes do PDB: {e}")
    return list(ligands)

def extract_ligand_from_pdb(pdb_file, res_name, output_file):
    try:
        with open(pdb_file, 'r') as f_in, open(output_file, 'w') as f_out:
            for line in f_in:
                if line.startswith("HETATM") and line[17:20].strip() == res_name:
                    f_out.write(line)
            f_out.write("END
")
    except Exception as e:
        st.error(f"Erro ao extrair ligante: {e}")

def sanitize_filename(name):
    sanitized = re.sub(r'[\\/*?:"<>| ,()\[\]{}]', "_", str(name))
    sanitized = re.sub(r'_+', '_', sanitized).strip('_')
    return sanitized.lower()

def get_vina_affinity(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.startswith("REMARK VINA RESULT:"):
                        match = re.search(r'REMARK VINA RESULT:\s+([-\d\.]+)', line)
                        if match:
                            return float(match.group(1))
    except Exception:
        pass
    return np.nan


# ==========================================
# BARRA LATERAL E PAINEL ADMIN
# ==========================================
with st.sidebar:
    st.markdown("### Autoria do Projeto")
    st.markdown("**Micael Davi Lima de Oliveira**")
    st.markdown("*Iniciação Científica*")
    st.markdown("**Faculdade de Ciências Farmacêuticas**")
    st.markdown("Universidade Federal do Amazonas (UFAM)")
    st.markdown("---")
    
    st.markdown("### 🚦 Status do Servidor")
    simultaneos = get_active_runs_count()
    if simultaneos == 0:
        st.success(f"🟢 **{simultaneos}** cálculos rodando agora.")
    elif simultaneos < 3:
        st.warning(f"🟡 **{simultaneos}** cálculos rodando agora.")
    else:
        st.error(f"🔴 **{simultaneos}** cálculos simultâneos (Alto Uso!).")
        
    with st.expander("🛠️ Painel Admin (Emergência)"):
        st.caption("Acesse para resetar o servidor em caso de travamento.")
        senha_admin = st.text_input("Senha Mestre:", type="password")
        if st.button("⛔ Parar Todos os Cálculos"):
            if senha_admin == "239151":
                try:
                    mortos = kill_all_processes()
                    st.success(f"Sistema limpo. {mortos} processos foram abortados à força.")
                    time.sleep(2)
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao tentar parar processos: {e}")
            else:
                st.error("Acesso negado. Senha incorreta.")
                
    st.markdown("---")
    st.caption("Desenvolvido para ensino e pesquisa em Química Medicinal Computacional.")

# Cabeçalho Principal
st.title("🧬 BioDockUfam: Uma ferramenta automatizada de docking molecular")
st.markdown("Plataforma acadêmica para ensino de **Química Medicinal Computacional** e **Planejamento de Fármacos**.")

# Inicialização das variáveis de memória do Streamlit
for key, default in [
    ('cx', 0.0), ('cy', 0.0), ('cz', 0.0), ('sx', 20.0), ('sy', 20.0), ('sz', 20.0),
    ('smiles', ""), ('nome_ligante_salvar', "ligante"), ('rec_pdb_final', "receptor_prep.pdb"),
    ('rec_final', "receptor.pdbqt"), ('lig_final', "ligante.pdbqt"), ('original_pdb', "2XV7.pdb"),
    ('redocking_mode', False), ('extracted_lig_pdb', ""), ('vs_mode', False),
    ('vs_results_dir', ""), ('vina_log_output', ""), ('sdf_ligand_generated', ""),
    ('global_zip_ready', False), ('global_zip_data', b"")
]:
    if key not in st.session_state: st.session_state[key] = default

# Abas
tab_install, tab_receptor, tab_ligante, tab_gridbox, tab_vina, tab_executar, tab_visualizar, tab_referencias = st.tabs([
    "🛠️ 1. Ambiente", "🧬 2. Receptor", "💊 3. Ligante", "📦 4. Grid Box", "⚙️ 5. Vina Config", "🚀 6. Docking", "👁️ 7. Análise de Resultados", "📚 8. Referências"
])

# ==========================================
# ABA 1: Instalação de Dependências
# ==========================================
with tab_install:
    st.header("1. Verificação do Ambiente Computacional")
    if LIBS_INSTALADAS:
        st.success("✅ **Sistema Operante:** Todas as bibliotecas de quimioinformática e bioinformática foram detectadas com sucesso.")
    else:
        st.error(f"🚨 **Atenção:** Erro ao importar módulos: {ERRO_IMPORT}. Verifique requirements.txt e packages.txt.")

# ==========================================
# ABA 2: Preparação do Receptor
# ==========================================
with tab_receptor:
    st.header("2. O Alvo Farmacológico (Receptor)")
    col1, col2 = st.columns([1, 2])
    with col1:
        pdb_id = st.text_input("Código PDB ID:", value="2XV7")
        if st.button("Baixar e Visualizar"):
            try:
                r = requests.get(f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb")
                if r.status_code == 200:
                    with open(f"{pdb_id.upper()}.pdb", "w") as f: f.write(r.text)
                    st.session_state.original_pdb = f"{pdb_id.upper()}.pdb"
                    st.success(f"PDB {pdb_id.upper()} salvo!")
                else:
                    st.error("Falha ao buscar PDB no servidor.")
            except Exception as e:
                st.error(f"Falha de conexão: {e}")

    with col2:
        if os.path.exists(st.session_state.original_pdb):
            try:
                with open(st.session_state.original_pdb, 'r') as f:
                    viewer = py3Dmol.view(width=600, height=350)
                    viewer.addModel(f.read(), "pdb")
                    viewer.setStyle({"cartoon": {'color':'spectrum'}})
                    viewer.zoomTo()
                    html(viewer._make_html(), width=600, height=350)
            except Exception as e:
                st.warning(f"Não foi possível renderizar o modelo 3D: {e}")

    st.divider()

    col_prep1, col_prep2 = st.columns(2)
    with col_prep1:
        st.subheader("A. Limpeza Conservadora")
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
            except Exception as e: 
                st.error(f"Erro crítico no PDBFixer: {e}")

    with col_prep2:
        st.subheader("B. Atribuição Eletrostática")
        if st.button("Calcular Cargas Gasteiger e Rígidez (PDBQT)"):
            try:
                with st.spinner("Calculando cargas parciais iterativas..."):
                    out_pdbqt = st.session_state.rec_pdb_final.replace(".pdb", ".pdbqt")
                    comando = ["obabel", "-i", "pdb", st.session_state.rec_pdb_final, "-o", "pdbqt", "-O", out_pdbqt, "-xr", "--partialcharge", "gasteiger"]
                    res = subprocess.run(comando, capture_output=True, text=True)
                    if os.path.exists(out_pdbqt):
                        st.session_state.rec_final = out_pdbqt
                        st.success(f"Matriz de cargas gerada: {out_pdbqt}")
                    else:
                        st.error(f"Erro ao gerar PDBQT. Saída do OpenBabel: {res.stderr}")
            except Exception as e: 
                st.error(f"Erro na execução do OpenBabel: {e}")

# ==========================================
# ABA 3: Preparação do Ligante
# ==========================================
with tab_ligante:
    st.header("3. Preparação do(s) Fármaco(s)")
    modo_preparacao = st.radio("Selecione a Estratégia de Processamento:", [
        "🔬 Triagem Simples: Molécula Única (SMILES/Nome)", 
        "♻️ Validação do Método: Re-Docking (Extrair Fármaco do PDB)",
        "🚀 Triagem Virtual Automática: Lote de Ligantes (Upload .sdf/.mol2/.pdb)",
        "📝 Triagem Automática (SMILES): Lista de Códigos"
    ])

    if "Triagem Simples" in modo_preparacao:
        st.session_state.redocking_mode = False
        st.session_state.vs_mode = False
        col_input, col_2d, col_3d = st.columns([1.2, 1, 1])
        
        with col_input:
            tipo_entrada = st.radio("Formato de entrada:", ("Nome Comum", "Código SMILES"))
            entrada_ligante = st.text_input("Insira o valor químico:")
            
            if st.button("1. Gerar Topologia (2D)"):
                try:
                    smiles_obtido = entrada_ligante
                    nome_final = "mol_inedita"
                    with st.spinner("Analisando estrutura..."):
                        if "Nome" in tipo_entrada:
                            comps = pcp.get_compounds(entrada_ligante, 'name')
                            if comps:
                                smiles_obtido = comps[0].isomeric_smiles
                                nome_final = sanitize_filename(entrada_ligante)
                                st.success(f"SMILES: {smiles_obtido}")
                            else:
                                st.error("Molécula não encontrada.")
                                smiles_obtido = ""
                        else:
                            try:
                                comps = pcp.get_compounds(smiles_obtido, 'smiles')
                                if comps and comps[0].iupac_name:
                                    nome_final = sanitize_filename(comps[0].iupac_name)
                                    st.success(f"IUPAC: {comps[0].iupac_name}")
                            except:
                                st.info("SMILES Inédito detectado.")
                                    
                    if smiles_obtido:
                        st.session_state.smiles = smiles_obtido
                        st.session_state.nome_ligante_salvar = nome_final
                        mol = Chem.MolFromSmiles(smiles_obtido)
                        if mol: 
                            st.session_state.img_2d = Draw.MolToImage(mol, size=(300, 300))
                        else:
                            st.error("Código SMILES incorreto.")
                except Exception as e: 
                    st.error(f"Erro na busca: {e}")

            if st.session_state.smiles:
                if st.button("2. Minimizar (3D) Rápido e Gerar PDBQT", type="primary"):
                    try:
                        sdf_file = f"{st.session_state.nome_ligante_salvar}.sdf"
                        pdbqt_file = f"{st.session_state.nome_ligante_salvar}.pdbqt"
                        
                        with st.spinner("Gerando 3D ultrarrápido (RDKit)..."):
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
                                st.success(f"Ligante 3D otimizado com sucesso.")
                            else:
                                st.error("Erro na conversão pelo OpenBabel.")
                    except Exception as e:
                        st.error(f"Falha na geração 3D acelerada: {e}")

        with col_2d:
            if 'img_2d' in st.session_state: st.image(st.session_state.img_2d)

        with col_3d:
            if 'mol2_file_path' in st.session_state and os.path.exists(st.session_state.mol2_file_path):
                try:
                    with open(st.session_state.mol2_file_path, 'r') as f:
                        viewer_lig = py3Dmol.view(width=300, height=300)
                        viewer_lig.addModel(f.read(), "sdf")
                        viewer_lig.setStyle({"stick": {'colorscheme': 'greenCarbon'}})
                        viewer_lig.zoomTo()
                        html(viewer_lig._make_html(), width=300, height=300)
                except Exception:
                    pass

    elif "Validação" in modo_preparacao:
        st.session_state.redocking_mode = True
        st.session_state.vs_mode = False
        if os.path.exists(st.session_state.original_pdb):
            ligantes = get_ligands_from_pdb(st.session_state.original_pdb)
            if ligantes:
                lig_selecionado = st.selectbox("Fármaco co-cristalizado detectado:", ligantes)
                if st.button("Extrair e Manter Coordenadas Naturais", type="primary"):
                    try:
                        ext_pdb = f"{lig_selecionado}_redocking.pdb"
                        ext_pdbqt = f"{lig_selecionado}_redocking.pdbqt"
                        extract_ligand_from_pdb(st.session_state.original_pdb, lig_selecionado, ext_pdb)
                        subprocess.run(["obabel", "-ipdb", ext_pdb, "-opdbqt", "-O", ext_pdbqt, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                        st.session_state.lig_final = ext_pdbqt
                        st.session_state.extracted_lig_pdb = ext_pdb
                        st.success(f"Coordenadas extraídas! Salvo como: {ext_pdbqt}")
                    except Exception as e:
                        st.error(f"Erro na extração: {e}")
            else:
                st.warning("Nenhum ligante orgânico detectado.")
                
    elif "SMILES" in modo_preparacao:
        st.session_state.redocking_mode = False
        st.session_state.vs_mode = True
        texto_smiles = st.text_area("Insira os códigos SMILES (um por linha):")
        
        if st.button("Processar SMILES", type="primary"):
            try:
                if texto_smiles.strip():
                    os.makedirs("Ligantes", exist_ok=True)
                    for f in glob.glob("Ligantes/*.pdbqt"): os.remove(f) 
                    
                    linhas = [l for l in texto_smiles.split('
') if l.strip()]
                    total_sucesso, total_falha = 0, 0
                    my_bar = st.progress(0)
                    
                    for idx, linha in enumerate(linhas):
                        partes = linha.split(',')
                        smi = partes[0].strip()
                        nome = sanitize_filename(partes[1].strip()) if len(partes) > 1 else f"ligante_{idx+1}"
                        
                        try:
                            mol = Chem.MolFromSmiles(smi)
                            if mol:
                                mol_3d = Chem.AddHs(mol)
                                if AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG()) == 0:
                                    AllChem.MMFFOptimizeMolecule(mol_3d)
                                    sdf_temp = f"Ligantes/{nome}.sdf"
                                    pdbqt_final = f"Ligantes/{nome}.pdbqt"
                                    Chem.SDWriter(sdf_temp).write(mol_3d)
                                    
                                    subprocess.run(["obabel", "-isdf", sdf_temp, "-opdbqt", "-O", pdbqt_final, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                                    if os.path.exists(pdbqt_final):
                                        total_sucesso += 1
                                    else:
                                        total_falha += 1
                                else:
                                    total_falha += 1
                            else:
                                total_falha += 1
                        except Exception:
                            total_falha += 1
                            
                        my_bar.progress(int(((idx + 1) / len(linhas)) * 100))
                    
                    my_bar.empty()
                    st.success(f"{total_sucesso} moléculas processadas. {total_falha} descartadas.")
                    st.session_state.lig_final = "Múltiplos Ligantes (Modo Lote Ativado)"
            except Exception as e:
                st.error(f"Erro no processamento em lote: {e}")

    else:
        st.session_state.redocking_mode = False
        st.session_state.vs_mode = True
        uploaded_files = st.file_uploader("Arquivos .sdf/.mol2/.pdb", accept_multiple_files=True)
        if uploaded_files:
            if st.button("Processar Lote"):
                try:
                    os.makedirs("Ligantes", exist_ok=True)
                    for f in glob.glob("Ligantes/*.pdbqt"): os.remove(f)
                    total_sucesso = 0
                    for uf in uploaded_files:
                        t_path = os.path.join("Ligantes", sanitize_filename(uf.name))
                        with open(t_path, "wb") as f:
                            f.write(uf.getbuffer())
                        out_pdbqt = t_path.replace(os.path.splitext(t_path)[1], ".pdbqt")
                        subprocess.run(["obabel", t_path, "-opdbqt", "-O", out_pdbqt, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
                        if os.path.exists(out_pdbqt):
                            total_sucesso += 1
                            if t_path != out_pdbqt: os.remove(t_path)
                    st.success(f"{total_sucesso} arquivos convertidos com sucesso para a pasta de triagem.")
                    st.session_state.lig_final = "Múltiplos Ligantes (Modo Lote Ativado)"
                except Exception as e:
                    st.error(f"Erro ao processar lote: {e}")

# ==========================================
# ABA 4: Grid Box (LaBOX)
# ==========================================
with tab_gridbox:
    st.header("4. Mapeamento do Espaço de Busca (Grid Box)")
    tipo_docking = st.radio("Estratégia:", ["🎯 Site-Directed", "🌍 Blind Docking"])
    
    col_box1, col_box2 = st.columns([1.2, 1])
    with col_box1:
        if st.button("Calcular Dimensões (LaBOX)"):
            try:
                box_input = st.session_state.extracted_lig_pdb if "Site-Directed" in tipo_docking else st.session_state.rec_pdb_final
                with st.spinner("Mapeando..."):
                    if not os.path.exists("LaBOX.py"):
                        r_labox = requests.get("https://raw.githubusercontent.com/RyanZR/LaBOX/main/LaBOX.py")
                        with open("LaBOX.py", "w") as f: f.write(r_labox.text)
                    
                    res_labox = subprocess.run([sys.executable, "LaBOX.py", "-l", box_input, "-c"], capture_output=True, text=True)
                    if res_labox.returncode == 0:
                        match_c = re.search(r'X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z\s+([-\d.]+)', res_labox.stdout)
                        match_s = re.search(r'W\s+([-\d.]+)\s+H\s+([-\d.]+)\s+D\s+([-\d.]+)', res_labox.stdout)
                        if match_c and match_s:
                            st.session_state.cx, st.session_state.cy, st.session_state.cz = map(float, match_c.groups())
                            st.session_state.sx, st.session_state.sy, st.session_state.sz = map(float, match_s.groups())
                            st.rerun() 
                    else:
                        st.error("Erro interno no LaBOX.")
            except Exception as e:
                st.error(f"Falha de execução do LaBOX: {e}")

    with col_box2:
        c_x, c_y, c_z = st.columns(3)
        cx = c_x.number_input("Center X", key='cx', step=0.1, value=st.session_state.cx)
        cy = c_y.number_input("Center Y", key='cy', step=0.1, value=st.session_state.cy)
        cz = c_z.number_input("Center Z", key='cz', step=0.1, value=st.session_state.cz)
        sx = c_x.number_input("Size W", key='sx', step=0.1, value=st.session_state.sx)
        sy = c_y.number_input("Size H", key='sy', step=0.1, value=st.session_state.sy)
        sz = c_z.number_input("Size D", key='sz', step=0.1, value=st.session_state.sz)

# ==========================================
# ABA 5: Configuração Vina
# ==========================================
with tab_vina:
    st.header("5. Geração de Protocolo do Vina")
    vina_exhaustiveness = st.number_input("Poder (Exhaustiveness):", value=8)
    vina_cpus = st.number_input("Núcleos (CPU):", min_value=1, value=1)
    
    if st.button("Gerar 'config.txt'", type="primary"):
        try:
            config_content = f"receptor = {st.session_state.rec_final}\n"
            if not st.session_state.vs_mode: config_content += f"ligand = {st.session_state.lig_final}\n\n"
            config_content += f"center_x = {st.session_state.cx}\ncenter_y = {st.session_state.cy}\ncenter_z = {st.session_state.cz}\n\n"
            config_content += f"size_x = {st.session_state.sx}\nsize_y = {st.session_state.sy}\nsize_z = {st.session_state.sz}\n\n"
            config_content += f"exhaustiveness = {vina_exhaustiveness}\ncpu = {vina_cpus}\n"
            
            with open("config.txt", "w") as f: f.write(config_content)
            st.success("Configuração compilada.")
        except Exception as e:
            st.error(f"Erro ao salvar configuração: {e}")

# ==========================================
# ABA 6: Execução do Docking Molecular
# ==========================================
with tab_executar:
    st.header("6. Simulação Termodinâmica em Triplicata")
    vina_exe = "vina_1.2.7_linux_x86_64"
    config_file_exec = "config.txt"
    
    output_pdbqt_base = st.text_input("Nome base saída:", value=f"resultado_docking")
    
    if st.button("▶️ Iniciar Cálculo em Triplicata", type="primary"):
        if not os.path.exists(config_file_exec):
            st.error("Configuração não encontrada. Vá na aba 5 primeiro.")
        else:
            try:
                if not os.path.exists(vina_exe):
                    r_vina = requests.get(f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{vina_exe}")
                    with open(vina_exe, 'wb') as f: f.write(r_vina.content)
                    os.chmod(vina_exe, 0o755)

                log_outputs = ""
                log_placeholder = st.empty()
                
                if st.session_state.vs_mode:
                    # Virtual Screening Mode
                    st.session_state.vs_results_dir = f"VS_Saida_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    
                    for rep in range(1, 4):
                        rep_dir = os.path.join(st.session_state.vs_results_dir, f"rep{rep}")
                        os.makedirs(rep_dir, exist_ok=True)
                        process_args = f"./{vina_exe} --config {config_file_exec} --batch Ligantes/*.pdbqt --dir {rep_dir}"
                        
                        log_outputs += f"\n--- INICIANDO BATCH VS: REPLICATA {rep} DE 3 ---\n"
                        process = subprocess.Popen(process_args, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        register_run(process.pid)
                        
                        try:
                            for line in iter(process.stdout.readline, ''):
                                log_outputs += line
                                log_placeholder.code(log_outputs, language="text")
                            process.wait()
                        finally:
                            unregister_run(process.pid)
                else:
                    # Single Ligand Mode in Triplicate
                    st.session_state.single_result_base = output_pdbqt_base
                    for rep in range(1, 4):
                        out_rep = f"{output_pdbqt_base}_rep{rep}.pdbqt"
                        process_args = [f"./{vina_exe}", "--config", config_file_exec, "--out", out_rep]
                        
                        log_outputs += f"\n--- INICIANDO SINGLE DOCKING: REPLICATA {rep} DE 3 ---\n"
                        process = subprocess.Popen(process_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                        register_run(process.pid)
                        
                        try:
                            for line in iter(process.stdout.readline, ''):
                                log_outputs += line
                                log_placeholder.code(log_outputs, language="text")
                            process.wait()
                        finally:
                            unregister_run(process.pid)
                            
                st.session_state.vina_log_output = log_outputs
                st.success("Simulação em triplicata concluída com sucesso!")
            except Exception as e:
                st.error(f"Erro grave na simulação: {e}\n{traceback.format_exc()}")

# ==========================================
# ABA 7: Análise Químico-Estrutural e Exportação
# ==========================================
with tab_visualizar:
    st.header("7. Análise de Resultados e Exportação")
    
    if st.session_state.vs_mode:
        if not st.session_state.get('vs_results_dir') or not os.path.exists(st.session_state.vs_results_dir):
            st.warning("Execute o Docking em lote (Aba 6) primeiro para gerar resultados.")
        else:
            ligand_files = glob.glob(os.path.join(st.session_state.vs_results_dir, "rep1", "*.pdbqt"))
            data_results = []
            for f in ligand_files:
                basename = os.path.basename(f)
                v1 = get_vina_affinity(os.path.join(st.session_state.vs_results_dir, "rep1", basename))
                v2 = get_vina_affinity(os.path.join(st.session_state.vs_results_dir, "rep2", basename))
                v3 = get_vina_affinity(os.path.join(st.session_state.vs_results_dir, "rep3", basename))
                vals = [v for v in [v1, v2, v3] if not np.isnan(v)]
                mean_val = round(np.mean(vals), 2) if vals else np.nan
                std_val = round(np.std(vals), 2) if len(vals) > 1 else 0.0
                data_results.append({"Ligante": basename.replace('.pdbqt',''), "Média (kcal/mol)": mean_val, "Desvio Padrão": std_val})
            
            if data_results:
                df_res = pd.DataFrame(data_results).sort_values(by="Média (kcal/mol)")
                st.dataframe(df_res, use_container_width=True, hide_index=True)
    else:
        base_name = st.session_state.get('single_result_base', '')
        if base_name and os.path.exists(f"{base_name}_rep1.pdbqt"):
            v1 = get_vina_affinity(f"{base_name}_rep1.pdbqt")
            v2 = get_vina_affinity(f"{base_name}_rep2.pdbqt")
            v3 = get_vina_affinity(f"{base_name}_rep3.pdbqt")
            vals = [v for v in [v1, v2, v3] if not np.isnan(v)]
            mean_val = round(np.mean(vals), 2) if vals else np.nan
            std_val = round(np.std(vals), 2) if len(vals) > 1 else 0.0
            st.metric("Afinidade Média (Gibbs)", f"{mean_val} kcal/mol", f"± {std_val} SD")
            
            if st.button("Sintetizar Complexo PDB (Rep 1)"):
                try:
                    # Gera arquivos estruturais rápidos de visualização
                    subprocess.run(["obabel", "-ipdbqt", f"{base_name}_rep1.pdbqt", "-opdb", "-O", "melhor_pose.pdb"])
                    subprocess.run(["obabel", "-ipdbqt", f"{base_name}_rep1.pdbqt", "-osdf", "-O", f"ligante_final.sdf"])
                    st.session_state.complex_generated = True
                    st.session_state.complex_file = "melhor_pose.pdb"
                    st.session_state.sdf_ligand_generated = "ligante_final.sdf"
                    st.success("Arquivos individuais gerados para auditoria.")
                except Exception as e:
                    st.error(f"Erro na síntese: {e}")

    # ==========================================
    # NOVO: COMPACTAÇÃO GLOBAL COMPLETA SOLICITADA
    # ==========================================
    st.divider()
    st.subheader("📦 Pacote de Exportação Completo (.ZIP)")
    st.markdown("Clique no botão abaixo para consolidar absolutamente **todos os arquivos de entrada, parametrizações, arquivos config, registros de log de terminal e resultados gerados (poses e matrizes PDBQT)** em um único pacote auditável.")

    if st.button("🎁 Compactar Todos os Dados do Experimento", type="primary", use_container_width=True):
        try:
            with st.spinner("Varrendo diretórios do servidor e estruturando o ZIP..."):
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    # 1. Configurações base
                    if os.path.exists("config.txt"):
                        zf.write("config.txt", "config.txt")
                    
                    # 2. Receptores (Original e Tratado)
                    if os.path.exists(st.session_state.rec_pdb_final):
                        zf.write(st.session_state.rec_pdb_final, os.path.basename(st.session_state.rec_pdb_final))
                    if os.path.exists(st.session_state.rec_final):
                        zf.write(st.session_state.rec_final, os.path.basename(st.session_state.rec_final))
                    if os.path.exists(st.session_state.original_pdb):
                        zf.write(st.session_state.original_pdb, os.path.basename(st.session_state.original_pdb))
                    
                    # 3. Estruturas de Entrada dos Ligantes
                    if st.session_state.vs_mode:
                        if os.path.exists("Ligantes"):
                            for f in glob.glob("Ligantes/*"):
                                zf.write(f, os.path.join("arquivos_entrada_ligantes", os.path.basename(f)))
                    else:
                        if os.path.exists(st.session_state.lig_final):
                            zf.write(st.session_state.lig_final, os.path.basename(st.session_state.lig_final))
                        if 'mol2_file_path' in st.session_state and os.path.exists(st.session_state.mol2_file_path):
                            zf.write(st.session_state.mol2_file_path, os.path.basename(st.session_state.mol2_file_path))
                        if os.path.exists(st.session_state.extracted_lig_pdb):
                            zf.write(st.session_state.extracted_lig_pdb, os.path.basename(st.session_state.extracted_lig_pdb))

                    # 4. Resultados das Poses (Output)
                    if st.session_state.vs_mode and st.session_state.get('vs_results_dir'):
                        if os.path.exists(st.session_state.vs_results_dir):
                            for root, dirs, files in os.walk(st.session_state.vs_results_dir):
                                for file in files:
                                    fp = os.path.join(root, file)
                                    arcname = os.path.join("resultados_triagem_lote", os.path.relpath(fp, st.session_state.vs_results_dir))
                                    zf.write(fp, arcname)
                    else:
                        base_name = st.session_state.get('single_result_base', '')
                        if base_name:
                            for rep in range(1, 4):
                                out_rep = f"{base_name}_rep{rep}.pdbqt"
                                if os.path.exists(out_rep):
                                    zf.write(out_rep, os.path.basename(out_rep))
                            if os.path.exists("melhor_pose.pdb"):
                                zf.write("melhor_pose.pdb", "complexo_sintetizado_rep1.pdb")
                            if os.path.exists("ligante_final.sdf"):
                                zf.write("ligante_final.sdf", "farmaco_isolado_rep1.sdf")

                    # 5. Saída de Logs Brutos
                    if st.session_state.vina_log_output:
                        with open("vina_terminal_log.txt", "w") as log_f:
                            log_f.write(st.session_state.vina_log_output)
                        zf.write("vina_terminal_log.txt", "vina_terminal_log.txt")
                        os.remove("vina_terminal_log.txt")

                st.session_state.global_zip_ready = True
                st.session_state.global_zip_data = zip_buffer.getvalue()
                st.success("🎉 Arquivo compactado criado com sucesso!")
        except Exception as e:
            st.error(f"Erro durante a compactação: {e}")

    if st.session_state.get('global_zip_ready', False):
        st.download_button(
            label="📥 Baixar Experimento Completo (.ZIP)",
            data=st.session_state.global_zip_data,
            file_name=f"BioDockUfam_Sessao_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            use_container_width=True,
            type="primary"
        )

# ==========================================
# ABA 8: Referências
# ==========================================
with tab_referencias:
    st.header("📚 Referências Bibliográficas e Algoritmos")
    st.markdown("* **AutoDock Vina:** Trott, O., & Olson, A. J. (2010). Journal of Computational Chemistry, 31(2), 455-461.")
    st.markdown("* **Barreiro, E. J., & Fraga, C. A. M. (2015).** *Química Medicinal: As Bases Farmacológicas da Ação dos Fármacos*. 3ª Ed. Artmed.")
