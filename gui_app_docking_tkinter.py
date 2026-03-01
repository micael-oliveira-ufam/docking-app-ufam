import os
import sys
import subprocess
import re
import glob
import threading
from datetime import datetime
import webbrowser
import numpy as np
import pandas as pd
import requests

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# Bio e Quimioinformática
try:
    import py3Dmol
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    import pubchempy as pcp
    from rdkit import Chem
    from rdkit.Chem import Draw
    from Bio.PDB import PDBParser
    LIBS_INSTALADAS = True
except ImportError:
    LIBS_INSTALADAS = False

# =======================================================
# FUNÇÕES AUXILIARES
# =======================================================
def get_ligands_from_pdb(pdb_file):
    ligands = set()
    filtros = ["HOH", "WAT", "DOD", "NA", "CL", "MG", "K", "SO4", "PO4", "EDO", "GOL", "FMT", "ACT"]
    if os.path.exists(pdb_file):
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith("HETATM"):
                    res_name = line[17:20].strip()
                    if res_name not in filtros: ligands.add(res_name)
    return list(ligands)

def extract_ligand(pdb_file, res_name, output_file):
    with open(pdb_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            if line.startswith("HETATM") and line[17:20].strip() == res_name: f_out.write(line)
        f_out.write("END\n")

def sanitize_name(name):
    sanitized = re.sub(r'[\\/*?:"<>| ,()\[\]{}]', "_", str(name))
    return re.sub(r'_+', '_', sanitized).strip('_').lower()

def get_vina_affinity(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith("REMARK VINA RESULT:"):
                    match = re.search(r'REMARK VINA RESULT:\s+([-\d\.]+)', line)
                    if match: return float(match.group(1))
    return np.nan

def show_3d_in_browser(html_content, filename="temp_3d.html"):
    """Salva o HTML do py3Dmol e abre no navegador padrão."""
    filepath = os.path.abspath(filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(html_content)
    webbrowser.open(f"file://{filepath}")

# =======================================================
# APLICATIVO TKINTER
# =======================================================
class DockingAppTk(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Laboratório Virtual: Docking Molecular - FCF/UFAM (Micael Davi)")
        self.geometry("1100x750")
        
        # Variáveis de Estado
        self.cx = self.cy = self.cz = 0.0
        self.sx = self.sy = self.sz = 20.0
        self.original_pdb = ""
        self.rec_pdb_final = "receptor_prep.pdb"
        self.rec_final = "receptor.pdbqt"
        self.lig_final = "ligante.pdbqt"
        self.smiles = ""
        self.nome_ligante_salvar = "ligante"
        self.extracted_lig_pdb = ""
        self.vs_mode = tk.BooleanVar(value=False)
        self.redocking_mode = tk.BooleanVar(value=False)
        self.vs_results_dir = ""
        self.single_result_base = ""
        self.complex_file = ""

        # Configuração de Estilo (ttk)
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        
        self.create_widgets()
        
        if not LIBS_INSTALADAS:
            messagebox.showerror("Erro de Dependências", "Bibliotecas faltando. Instale-as via pip.")

    def create_widgets(self):
        # Cabeçalho
        header_frame = tk.Frame(self, bg="#1e3a8a", pady=10)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="🧬 DockEasy-Ufam (Ferramenta de docking molecular automatizado para o AutoDock Vina v.1.2.7)", font=("Helvetica", 16, "bold"), fg="white", bg="#1e3a8a").pack()
        tk.Label(header_frame, text="Desenvolvido por Micael Davi Lima de Oliveira em Fevereiro de 2026", font=("Helvetica", 10), fg="#93c5fd", bg="#1e3a8a").pack()

        # Notebook (Abas)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.setup_tab_receptor()
        self.setup_tab_ligante()
        self.setup_tab_gridbox()
        self.setup_tab_docking()
        self.setup_tab_analysis()

    # ---------------------------------------------------
    # THREADING WRAPPER (Impede o congelamento da UI)
    # ---------------------------------------------------
    def run_in_thread(self, target_func, *args):
        """Executa funções pesadas em segundo plano."""
        thread = threading.Thread(target=target_func, args=args, daemon=True)
        thread.start()

    # ================= TAB 1: RECEPTOR =================
    def setup_tab_receptor(self):
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="1. Receptor")

        lf1 = ttk.LabelFrame(tab, text=" 1. Download e Visualização (RCSB PDB) ", padding=15)
        lf1.pack(fill=tk.X, pady=10)
        
        ttk.Label(lf1, text="Código PDB ID:").grid(row=0, column=0, padx=5, pady=5)
        self.ent_pdb_id = ttk.Entry(lf1, width=15)
        self.ent_pdb_id.insert(0, "2XV7")
        self.ent_pdb_id.grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(lf1, text="Baixar e Abrir 3D (Navegador)", command=self.download_pdb).grid(row=0, column=2, padx=10)

        lf2 = ttk.LabelFrame(tab, text=" 2. Preparação e Limpeza (PDBFixer & OpenBabel) ", padding=15)
        lf2.pack(fill=tk.X, pady=10)
        
        ttk.Button(lf2, text="Remover Água e Protonar (pH 7.4)", command=lambda: self.run_in_thread(self.run_pdbfixer)).grid(row=0, column=0, padx=10, pady=10)
        ttk.Button(lf2, text="Gerar PDBQT (Cargas Gasteiger)", command=lambda: self.run_in_thread(self.run_obabel_rec)).grid(row=0, column=1, padx=10, pady=10)

        self.lbl_rec_status = ttk.Label(tab, text="Status: Aguardando...", foreground="gray")
        self.lbl_rec_status.pack(pady=20)

    def download_pdb(self):
        pdb_id = self.ent_pdb_id.get().upper()
        if not pdb_id: return
        r = requests.get(f"https://files.rcsb.org/download/{pdb_id}.pdb")
        if r.status_code == 200:
            self.original_pdb = f"{pdb_id}.pdb"
            with open(self.original_pdb, "w") as f: f.write(r.text)
            messagebox.showinfo("Sucesso", f"PDB {pdb_id} salvo!")
            
            v = py3Dmol.view(width=800, height=600)
            v.addModel(r.text, "pdb")
            v.setStyle({"cartoon": {'color':'spectrum'}}); v.zoomTo()
            show_3d_in_browser(v._make_html(), "viewer_receptor.html")
            self.lbl_rec_status.config(text=f"Status: PDB {pdb_id} baixado com sucesso.")
        else:
            messagebox.showerror("Erro", "PDB não encontrado no banco de dados.")

    def run_pdbfixer(self):
        if not self.original_pdb: 
            self.after(0, lambda: messagebox.showwarning("Aviso", "Baixe o PDB primeiro."))
            return
        self.after(0, lambda: self.lbl_rec_status.config(text="Status: Processando PDBFixer...", foreground="blue"))
        try:
            fixer = PDBFixer(filename=self.original_pdb)
            fixer.findNonstandardResidues()
            fixer.replaceNonstandardResidues()
            fixer.removeHeterogens(False)
            fixer.addMissingHydrogens(7.4) 
            PDBFile.writeFile(fixer.topology, fixer.positions, open(self.rec_pdb_final, 'w'))
            self.after(0, lambda: self.lbl_rec_status.config(text="Status: Receptor isolado e protonado (pH 7.4)!", foreground="green"))
            self.after(0, lambda: messagebox.showinfo("Sucesso", "Receptor protonado com sucesso!"))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Erro", str(e)))

    def run_obabel_rec(self):
        self.after(0, lambda: self.lbl_rec_status.config(text="Status: Calculando cargas parciais...", foreground="blue"))
        cmd = ["obabel", "-i", "pdb", self.rec_pdb_final, "-o", "pdbqt", "-O", self.rec_final, "-xr", "--partialcharge", "gasteiger"]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(self.rec_final):
            self.after(0, lambda: self.lbl_rec_status.config(text=f"Status: Matriz gerada ({self.rec_final})", foreground="green"))
            self.after(0, lambda: messagebox.showinfo("Sucesso", "Receptor PDBQT finalizado!"))

    # ================= TAB 2: LIGANTE =================
    def setup_tab_ligante(self):
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="2. Ligantes")

        # Modos de Triagem
        mode_frame = ttk.LabelFrame(tab, text=" Estratégia de Obtenção ", padding=10)
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.lig_mode = tk.StringVar(value="single")
        ttk.Radiobutton(mode_frame, text="Triagem Simples (SMILES)", variable=self.lig_mode, value="single", command=self.update_lig_ui).grid(row=0, column=0, padx=10)
        ttk.Radiobutton(mode_frame, text="Re-Docking (Extrair PDB)", variable=self.lig_mode, value="redock", command=self.update_lig_ui).grid(row=0, column=1, padx=10)
        ttk.Radiobutton(mode_frame, text="HTVS (Triagem em Lote)", variable=self.lig_mode, value="htvs", command=self.update_lig_ui).grid(row=0, column=2, padx=10)

        # Containers
        self.frm_single = ttk.Frame(tab)
        self.frm_redock = ttk.Frame(tab)
        self.frm_htvs = ttk.Frame(tab)

        # Build Single
        ttk.Label(self.frm_single, text="SMILES ou Nome:").grid(row=0, column=0, pady=5)
        self.ent_smiles = ttk.Entry(self.frm_single, width=40)
        self.ent_smiles.grid(row=0, column=1, pady=5, padx=5)
        ttk.Button(self.frm_single, text="1. Topologia 2D", command=lambda: self.run_in_thread(self.proc_single)).grid(row=0, column=2, padx=5)
        ttk.Button(self.frm_single, text="2. Minimizar 3D (PDBQT)", command=lambda: self.run_in_thread(self.min_single)).grid(row=0, column=3, padx=5)
        self.lbl_img_2d = tk.Label(self.frm_single)
        self.lbl_img_2d.grid(row=1, column=0, columnspan=4, pady=10)

        # Build Redock
        ttk.Button(self.frm_redock, text="Carregar Ligantes do PDB", command=self.load_redock).grid(row=0, column=0, padx=5, pady=5)
        self.cb_redock = ttk.Combobox(self.frm_redock, state="readonly", width=15)
        self.cb_redock.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(self.frm_redock, text="Extrair Coordenadas Naturais", command=lambda: self.run_in_thread(self.ext_redock)).grid(row=0, column=2, padx=5, pady=5)

        # Build HTVS
        ttk.Label(self.frm_htvs, text="Faça upload de múltiplos arquivos (.sdf, .mol2) para o lote.").pack(pady=5)
        ttk.Button(self.frm_htvs, text="Upload e Processamento Automático", command=lambda: self.run_in_thread(self.proc_htvs)).pack(pady=10)

        self.frm_single.pack(fill=tk.BOTH, expand=True, pady=10) # Default show

        # Log
        self.txt_lig_log = tk.Text(tab, height=10, bg="black", fg="#00ff00", font=("Consolas", 9))
        self.txt_lig_log.pack(fill=tk.X, side=tk.BOTTOM, pady=10)

    def log_lig(self, msg):
        self.after(0, lambda: self.txt_lig_log.insert(tk.END, msg + "\n"))
        self.after(0, lambda: self.txt_lig_log.see(tk.END))

    def update_lig_ui(self):
        val = self.lig_mode.get()
        self.vs_mode.set(val == "htvs")
        self.redocking_mode.set(val == "redock")
        
        self.frm_single.pack_forget()
        self.frm_redock.pack_forget()
        self.frm_htvs.pack_forget()
        
        if val == "single": self.frm_single.pack(fill=tk.BOTH, expand=True, pady=10)
        elif val == "redock": self.frm_redock.pack(fill=tk.BOTH, expand=True, pady=10)
        else: self.frm_htvs.pack(fill=tk.BOTH, expand=True, pady=10)

    def proc_single(self):
        val = self.ent_smiles.get()
        if not val: return
        self.log_lig(f"Buscando: {val}...")
        try:
            comps = pcp.get_compounds(val, 'name')
            smiles_obtido = comps[0].isomeric_smiles if comps else val
            self.smiles = smiles_obtido
            self.nome_ligante_salvar = sanitize_name(val)
            
            mol = Chem.MolFromSmiles(self.smiles)
            if mol:
                Draw.MolToFile(mol, "temp_2d.png", size=(250, 250))
                img = Image.open("temp_2d.png")
                photo = ImageTk.PhotoImage(img)
                self.after(0, lambda: self.lbl_img_2d.config(image=photo))
                self.after(0, lambda: setattr(self.lbl_img_2d, 'image', photo)) # Mantenha a referência
                self.log_lig(f"Sucesso. SMILES: {self.smiles}")
        except Exception as e:
            self.log_lig(f"Erro ao buscar: {e}")

    def min_single(self):
        self.log_lig("Iniciando minimização MMFF94...")
        m2 = f"{self.nome_ligante_salvar}.mol2"
        pq = f"{self.nome_ligante_salvar}.pdbqt"
        subprocess.run(["obabel", f"-:{self.smiles}", "-O", m2, "--gen3d"], capture_output=True)
        subprocess.run(["obabel", "-imol2", m2, "-opdbqt", "-O", pq, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
        self.lig_final = pq
        self.log_lig(f"Concluído. Salvo como {pq}")
        self.after(0, lambda: messagebox.showinfo("Sucesso", "Ligante 3D otimizado com sucesso!"))

    def load_redock(self):
        ligs = get_ligands_from_pdb(self.original_pdb)
        self.cb_redock['values'] = ligs
        if ligs: self.cb_redock.current(0)
        else: messagebox.showwarning("Aviso", "Sem ligantes encontrados.")

    def ext_redock(self):
        lig = self.cb_redock.get()
        if not lig: return
        self.extracted_lig_pdb = f"{lig}_redocking.pdb"
        self.lig_final = f"{lig}_redocking.pdbqt"
        extract_ligand(self.original_pdb, lig, self.extracted_lig_pdb)
        subprocess.run(["obabel", "-ipdb", self.extracted_lig_pdb, "-opdbqt", "-O", self.lig_final, "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)
        self.log_lig(f"Extração Concluída: {self.lig_final}")
        self.after(0, lambda: messagebox.showinfo("Re-docking", "Coordenadas originais preservadas."))

    def proc_htvs(self):
        files = filedialog.askopenfilenames(title="Selecione os arquivos", filetypes=(("Molecules", "*.sdf *.mol2 *.pdb"), ("All Files", "*.*")))
        if not files: return
        
        self.log_lig("Limpando diretórios HTVS antigos...")
        os.makedirs("Ligantes_temp", exist_ok=True)
        os.makedirs("Ligantes", exist_ok=True)
        for f in glob.glob("Ligantes_temp/*") + glob.glob("Ligantes/*.pdbqt"): os.remove(f)

        temp_paths = []
        for fpath in files:
            t_path = os.path.join("Ligantes_temp", sanitize_name(os.path.basename(fpath)))
            with open(fpath, "rb") as fin, open(t_path, "wb") as fout: fout.write(fin.read())
            temp_paths.append(t_path)
            
        total_sucesso = 0; total_falha = 0
        self.log_lig("Processando quebra e geração 3D...")
        for t_path in temp_paths:
            base = os.path.splitext(os.path.basename(t_path))[0]
            cmd = ["obabel", t_path, "-omol2", "-O", f"Ligantes_temp/{base}_.mol2", "-m", "--gen3d", "-e"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            log_out = res.stdout + res.stderr
            sc = re.search(r'(\d+)\s+molecules?\s+converted', log_out, re.I)
            er = re.search(r'(\d+)\s+errors?', log_out, re.I)
            if sc: total_sucesso += int(sc.group(1))
            if er: total_falha += int(er.group(1))

        self.log_lig("Convertendo para PDBQT e calculando cargas...")
        for m2 in glob.glob("Ligantes_temp/*.mol2"):
            bm2 = os.path.basename(m2).replace(".mol2", "")
            subprocess.run(["obabel", "-imol2", m2, "-opdbqt", "-O", f"Ligantes/{bm2}.pdbqt", "-p", "7.4", "--partialcharge", "gasteiger"], capture_output=True)

        qtd = len(glob.glob("Ligantes/*.pdbqt"))
        self.log_lig(f"HTVS Concluído! Sucesso: {qtd} moléculas prontas. Falhas ignoradas: {total_falha}.")
        self.lig_final = "LOTE_HTVS"
        self.after(0, lambda: messagebox.showinfo("HTVS Pronto", f"{qtd} moléculas preparadas para a triagem."))

    # ================= TAB 3: GRID BOX =================
    def setup_tab_gridbox(self):
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="3. Grid Box")

        lf1 = ttk.LabelFrame(tab, text=" Automação de Cálculo (LaBOX/Biopython) ", padding=15)
        lf1.pack(fill=tk.X, pady=10)
        
        self.grid_mode = tk.StringVar(value="site")
        ttk.Radiobutton(lf1, text="Site-Directed", variable=self.grid_mode, value="site").grid(row=0, column=0, padx=10)
        ttk.Radiobutton(lf1, text="Blind Docking (Global)", variable=self.grid_mode, value="blind").grid(row=0, column=1, padx=10)
        
        ttk.Label(lf1, text="Referência PDB:").grid(row=1, column=0, pady=10)
        self.ent_grid_ref = ttk.Entry(lf1, width=30)
        self.ent_grid_ref.insert(0, "ligante_referencia.pdb")
        self.ent_grid_ref.grid(row=1, column=1)
        ttk.Button(lf1, text="Calcular Dimensões", command=lambda: self.run_in_thread(self.calc_grid)).grid(row=1, column=2, padx=10)

        lf2 = ttk.LabelFrame(tab, text=" Coordenadas Resultantes (Å) ", padding=15)
        lf2.pack(fill=tk.X, pady=10)

        # X,Y,Z variables
        self.var_cx = tk.DoubleVar(value=0.0); self.var_cy = tk.DoubleVar(value=0.0); self.var_cz = tk.DoubleVar(value=0.0)
        self.var_sx = tk.DoubleVar(value=20.0); self.var_sy = tk.DoubleVar(value=20.0); self.var_sz = tk.DoubleVar(value=20.0)

        ttk.Label(lf2, text="Center X:").grid(row=0, column=0, padx=5, pady=5); ttk.Entry(lf2, textvariable=self.var_cx, width=10).grid(row=0, column=1)
        ttk.Label(lf2, text="Center Y:").grid(row=0, column=2, padx=5, pady=5); ttk.Entry(lf2, textvariable=self.var_cy, width=10).grid(row=0, column=3)
        ttk.Label(lf2, text="Center Z:").grid(row=0, column=4, padx=5, pady=5); ttk.Entry(lf2, textvariable=self.var_cz, width=10).grid(row=0, column=5)

        ttk.Label(lf2, text="Size W:").grid(row=1, column=0, padx=5, pady=5); ttk.Entry(lf2, textvariable=self.var_sx, width=10).grid(row=1, column=1)
        ttk.Label(lf2, text="Size H:").grid(row=1, column=2, padx=5, pady=5); ttk.Entry(lf2, textvariable=self.var_sy, width=10).grid(row=1, column=3)
        ttk.Label(lf2, text="Size D:").grid(row=1, column=4, padx=5, pady=5); ttk.Entry(lf2, textvariable=self.var_sz, width=10).grid(row=1, column=5)

        ttk.Button(tab, text="Renderizar Caixa no Navegador 3D", command=self.render_gridbox).pack(pady=20)

    def calc_grid(self):
        ref = self.ent_grid_ref.get()
        if self.grid_mode.get() == "site":
            if not os.path.exists("LaBOX.py"):
                with open("LaBOX.py", "w") as f: f.write(requests.get("https://raw.githubusercontent.com/RyanZR/LaBOX/main/LaBOX.py").text)
            res = subprocess.run([sys.executable, "LaBOX.py", "-l", ref, "-c"], capture_output=True, text=True)
            if res.returncode == 0:
                mc = re.search(r'X\s+([-\d.]+)\s+Y\s+([-\d.]+)\s+Z\s+([-\d.]+)', res.stdout)
                ms = re.search(r'W\s+([-\d.]+)\s+H\s+([-\d.]+)\s+D\s+([-\d.]+)', res.stdout)
                if mc and ms:
                    self.cx, self.cy, self.cz = map(float, mc.groups())
                    self.sx, self.sy, self.sz = map(float, ms.groups())
                    self.after(0, lambda: self.var_cx.set(self.cx)); self.after(0, lambda: self.var_cy.set(self.cy)); self.after(0, lambda: self.var_cz.set(self.cz))
                    self.after(0, lambda: self.var_sx.set(self.sx)); self.after(0, lambda: self.var_sy.set(self.sy)); self.after(0, lambda: self.var_sz.set(self.sz))
                    self.after(0, lambda: messagebox.showinfo("LaBOX", "Sítio Ativo mapeado com sucesso!"))
        else:
            parser = PDBParser(QUIET=True)
            struct = parser.get_structure('rec', self.rec_pdb_final)
            c = struct.center_of_mass(); coords = [a.coord for a in struct.get_atoms()]
            self.cx, self.cy, self.cz = round(c[0],3), round(c[1],3), round(c[2],3)
            self.sx = round(max(x[0] for x in coords) - min(x[0] for x in coords) + 10, 3)
            self.sy = round(max(x[1] for x in coords) - min(x[1] for x in coords) + 10, 3)
            self.sz = round(max(x[2] for x in coords) - min(x[2] for x in coords) + 10, 3)
            
            self.after(0, lambda: self.var_cx.set(self.cx)); self.after(0, lambda: self.var_cy.set(self.cy)); self.after(0, lambda: self.var_cz.set(self.cz))
            self.after(0, lambda: self.var_sx.set(self.sx)); self.after(0, lambda: self.var_sy.set(self.sy)); self.after(0, lambda: self.var_sz.set(self.sz))
            self.after(0, lambda: messagebox.showinfo("Biopython", "Blind Docking (Centro de Massa) configurado!"))

    def render_gridbox(self):
        if os.path.exists(self.rec_pdb_final):
            with open(self.rec_pdb_final, 'r') as f:
                v = py3Dmol.view(width=800, height=600)
                v.addModel(f.read(), "pdb"); v.setStyle({"cartoon": {'color':'cyan'}})
                v.addBox({'center': {'x': self.var_cx.get(), 'y': self.var_cy.get(), 'z': self.var_cz.get()}, 
                          'dimensions': {'w': self.var_sx.get(), 'h': self.var_sy.get(), 'd': self.var_sz.get()}, 
                          'color': 'red', 'wireframe': True})
                v.zoomTo()
                show_3d_in_browser(v._make_html(), "viewer_gridbox.html")

    # ================= TAB 4: DOCKING =================
    def setup_tab_docking(self):
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="4. Docking")

        lf1 = ttk.LabelFrame(tab, text=" 1. Geração do Protocolo ", padding=15)
        lf1.pack(fill=tk.X, pady=10)
        
        ttk.Label(lf1, text="Exhaustiveness (Força Bruta):").grid(row=0, column=0, padx=5)
        self.var_exhaus = tk.IntVar(value=24)
        ttk.Entry(lf1, textvariable=self.var_exhaus, width=10).grid(row=0, column=1, padx=5)
        ttk.Button(lf1, text="Salvar config.txt", command=self.generate_config).grid(row=0, column=2, padx=20)

        lf2 = ttk.LabelFrame(tab, text=" 2. Motor AutoDock Vina (Triplicata) ", padding=15)
        lf2.pack(fill=tk.BOTH, expand=True, pady=10)

        ttk.Button(lf2, text="▶ INICIAR SIMULAÇÃO", command=lambda: self.run_in_thread(self.run_docking)).pack(pady=10)
        
        self.progress = ttk.Progressbar(lf2, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self.progress.pack(pady=10)

        self.txt_vina_log = tk.Text(lf2, bg="#0f172a", fg="#38bdf8", font=("Consolas", 10))
        self.txt_vina_log.pack(fill=tk.BOTH, expand=True, pady=5)

    def log_vina(self, msg):
        self.after(0, lambda: self.txt_vina_log.insert(tk.END, msg + "\n"))
        self.after(0, lambda: self.txt_vina_log.see(tk.END))

    def generate_config(self):
        self.cx, self.cy, self.cz = self.var_cx.get(), self.var_cy.get(), self.var_cz.get()
        self.sx, self.sy, self.sz = self.var_sx.get(), self.var_sy.get(), self.var_sz.get()
        cfg = f"receptor = {self.rec_final}\n"
        if not self.vs_mode.get(): cfg += f"ligand = {self.lig_final}\n"
        cfg += f"\ncenter_x = {self.cx}\ncenter_y = {self.cy}\ncenter_z = {self.cz}\n"
        cfg += f"size_x = {self.sx}\nsize_y = {self.sy}\nsize_z = {self.sz}\n"
        cfg += f"exhaustiveness = {self.var_exhaus.get()}\n"
        with open("config.txt", "w") as f: f.write(cfg)
        messagebox.showinfo("Sucesso", "Arquivo config.txt atualizado!")

    def run_docking(self):
        vina_exe = "vina_1.2.7_linux_x86_64"
        if not os.path.exists(vina_exe):
            self.log_vina("Baixando Vina Executable...")
            with open(vina_exe, 'wb') as f: f.write(requests.get(f"https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/{vina_exe}").content)
            os.chmod(vina_exe, 0o755)

        self.after(0, lambda: self.txt_vina_log.delete('1.0', tk.END))
        self.after(0, lambda: self.progress.config(value=0))

        if self.vs_mode.get():
            rec_b = self.rec_final.replace('.pdbqt', '')
            self.vs_results_dir = f"Screening_{rec_b}_{datetime.now().strftime('%H%M')}"
            os.makedirs(self.vs_results_dir, exist_ok=True)
            
            for rep in range(1, 4):
                self.log_vina(f"\n--- Iniciando Triagem Lote: Replicata {rep}/3 ---")
                rep_dir = os.path.join(self.vs_results_dir, f"rep{rep}")
                os.makedirs(rep_dir, exist_ok=True)
                cmd = f"./{vina_exe} --config config.txt --batch Ligantes/*.pdbqt --dir {rep_dir}"
                res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
                self.log_vina(res.stdout)
                self.after(0, lambda r=rep: self.progress.config(value=(r/3.0)*100))
        else:
            rec_b = self.rec_final.replace('.pdbqt', '')
            lig_b = self.lig_final.replace('.pdbqt', '')
            self.single_result_base = f"resultado_{lig_b}_{rec_b}"
            
            for rep in range(1, 4):
                self.log_vina(f"\n--- Iniciando Docking Singular: Replicata {rep}/3 ---")
                out_rep = f"{self.single_result_base}_rep{rep}.pdbqt"
                cmd = [f"./{vina_exe}", "--config", "config.txt", "--out", out_rep]
                res = subprocess.run(cmd, capture_output=True, text=True)
                self.log_vina(res.stdout)
                self.after(0, lambda r=rep: self.progress.config(value=(r/3.0)*100))
        
        self.log_vina("\n=== CÁLCULO TERMODINÂMICO FINALIZADO ===")
        self.after(0, lambda: messagebox.showinfo("Concluído", "Docking concluído! Vá para a Aba 5."))

    # ================= TAB 5: ANÁLISE =================
    def setup_tab_analysis(self):
        tab = ttk.Frame(self.notebook, padding=20)
        self.notebook.add(tab, text="5. Análise & PDB")

        # Tabela
        lf_table = ttk.LabelFrame(tab, text=" Tabela de Afinidade Termodinâmica ", padding=10)
        lf_table.pack(fill=tk.BOTH, expand=True, pady=5)
        
        ttk.Button(lf_table, text="Atualizar Tabela", command=self.load_results_table).pack(pady=5)

        columns = ("Ligante", "Média (kcal/mol)", "Desvio Padrão", "Rep 1", "Rep 2", "Rep 3")
        self.tree = ttk.Treeview(lf_table, columns=columns, show="headings", height=8)
        for col in columns: self.tree.heading(col, text=col)
        self.tree.pack(fill=tk.BOTH, expand=True)

        # Controles
        lf_ctrl = ttk.LabelFrame(tab, text=" Síntese e Validação ", padding=10)
        lf_ctrl.pack(fill=tk.X, pady=10)

        self.cb_poses = ttk.Combobox(lf_ctrl, state="readonly", width=50)
        self.cb_poses.grid(row=0, column=0, padx=5, pady=5)
        
        ttk.Button(lf_ctrl, text="Sintetizar PDB Selecionado (Exibe 3D)", command=lambda: self.run_in_thread(self.synth_complex)).grid(row=0, column=1, padx=5)
        ttk.Button(lf_ctrl, text="Gerar Lote Inteiro (Se HTVS)", command=lambda: self.run_in_thread(self.synth_batch)).grid(row=0, column=2, padx=5)
        ttk.Button(lf_ctrl, text="Calcular RMSD", command=self.calc_rmsd).grid(row=0, column=3, padx=5)

    def load_results_table(self):
        for item in self.tree.get_children(): self.tree.delete(item)
        data = []; poses = []
        
        if self.vs_mode.get() and os.path.exists(self.vs_results_dir):
            for f in glob.glob(os.path.join(self.vs_results_dir, "rep1", "*.pdbqt")):
                bn = os.path.basename(f)
                v1 = get_vina_affinity(os.path.join(self.vs_results_dir, "rep1", bn))
                v2 = get_vina_affinity(os.path.join(self.vs_results_dir, "rep2", bn))
                v3 = get_vina_affinity(os.path.join(self.vs_results_dir, "rep3", bn))
                vals = [v for v in [v1, v2, v3] if not np.isnan(v)]
                data.append((bn.replace('_out.pdbqt',''), round(np.mean(vals),2) if vals else 0, round(np.std(vals),2) if len(vals)>1 else 0, v1, v2, v3))
                poses.append(os.path.join(self.vs_results_dir, "rep1", bn))
        elif not self.vs_mode.get() and self.single_result_base:
            v1 = get_vina_affinity(f"{self.single_result_base}_rep1.pdbqt")
            v2 = get_vina_affinity(f"{self.single_result_base}_rep2.pdbqt")
            v3 = get_vina_affinity(f"{self.single_result_base}_rep3.pdbqt")
            vals = [v for v in [v1, v2, v3] if not np.isnan(v)]
            data.append((os.path.basename(self.single_result_base), round(np.mean(vals),2) if vals else 0, round(np.std(vals),2) if len(vals)>1 else 0, v1, v2, v3))
            poses.extend([f"{self.single_result_base}_rep1.pdbqt", f"{self.single_result_base}_rep2.pdbqt", f"{self.single_result_base}_rep3.pdbqt"])

        data.sort(key=lambda x: x[1]) # Sort by mean
        for row in data: self.tree.insert("", tk.END, values=row)
        
        self.cb_poses['values'] = poses
        if poses: self.cb_poses.current(0)

    def synth_complex(self):
        pose_alvo = self.cb_poses.get()
        if not pose_alvo or not os.path.exists(pose_alvo): return
        
        best_pose = []
        in_m1 = False
        with open(pose_alvo, 'r') as f:
            for l in f:
                if l.startswith("MODEL 1"): in_m1 = True
                if in_m1: best_pose.append(l)
                if l.startswith("ENDMDL") and in_m1: break
                
        with open("melhor_pose.pdbqt", "w") as f: f.writelines(best_pose)
        subprocess.run(["obabel", "-ipdbqt", "melhor_pose.pdbqt", "-opdb", "-O", "melhor_pose.pdb"])
        
        with open(self.rec_pdb_final, 'r') as f: rec = [l for l in f.readlines() if not l.startswith("END")]
        lig = []
        with open("melhor_pose.pdb", 'r') as f:
            for l in f:
                if l.startswith("ATOM") or l.startswith("HETATM"): lig.append("HETATM" + l[6:17] + "UNL" + l[20:])
                
        c_name = f"complex_{os.path.basename(pose_alvo).replace('.pdbqt', '.pdb')}"
        c_name = c_name.replace("/", "_").replace("\\", "_")
        
        with open(c_name, "w") as f: f.write("".join(rec + lig + ["END\n"]))
        
        v = py3Dmol.view(width=800, height=600)
        v.addModel("".join(rec), "pdb"); v.setStyle({'model':0}, {"cartoon": {'color': 'spectrum'}})
        v.addModel("".join(lig), "pdb"); v.setStyle({'model':1}, {"stick": {'colorscheme': 'magentaCarbon'}})
        v.zoomTo({'model':1})
        show_3d_in_browser(v._make_html(), "viewer_complex.html")
        
        self.after(0, lambda: messagebox.showinfo("Sucesso", f"Complexo {c_name} gerado. Pronto para Discovery Studio/LigPlot+!"))

    def synth_batch(self):
        if not self.vs_mode.get() or not self.vs_results_dir: return
        out_dir = f"{self.vs_results_dir}_ComplexosPDB"
        os.makedirs(out_dir, exist_ok=True)
        
        with open(self.rec_pdb_final, 'r') as f: rec_lines = [l for l in f.readlines() if not l.startswith("END")]
        
        for p_file in glob.glob(os.path.join(self.vs_results_dir, "rep1", "*.pdbqt")):
            clean_name = os.path.basename(p_file).replace('.pdbqt', '').replace('_out', '')
            comp_path = os.path.join(out_dir, f"complexo_{clean_name}_rep1.pdb")
            
            best_pose = []
            in_m1 = False
            with open(p_file, 'r') as f:
                for line in f:
                    if line.startswith("MODEL 1"): in_m1 = True
                    if in_m1: best_pose.append(line)
                    if line.startswith("ENDMDL") and in_m1: break
            
            with open("temp_vs.pdbqt", "w") as f: f.writelines(best_pose)
            subprocess.run(["obabel", "-ipdbqt", "temp_vs.pdbqt", "-opdb", "-O", "temp_vs.pdb"])
            
            lig_lines = []
            with open("temp_vs.pdb", 'r') as f:
                for l in f.readlines():
                    if l.startswith("ATOM") or l.startswith("HETATM"):
                        lig_lines.append("HETATM" + l[6:17] + "UNL" + l[20:])
            with open(comp_path, "w") as f: f.write("".join(rec_lines + lig_lines + ["END\n"]))
            
        self.after(0, lambda: messagebox.showinfo("Lote Concluído", f"Todos os PDBs foram gerados na pasta {out_dir}."))

    def calc_rmsd(self):
        if not self.redocking_mode.get():
            messagebox.showwarning("Aviso", "O modo Re-docking não foi selecionado na Aba 2.")
            return
            
        res = subprocess.run(["obrms", self.extracted_lig_pdb, "melhor_pose.pdb"], capture_output=True, text=True)
        rmsd = re.search(r'RMSD.*?([\d\.]+)', res.stdout)
        if rmsd:
            val = float(rmsd.group(1))
            status = "✅ Válido" if val <= 2.0 else "❌ Inválido"
            messagebox.showinfo("RMSD", f"{status}\n\nO desvio calculado é de {val} Å.")
        else:
            messagebox.showerror("Erro", "Falha matemática ao calcular RMSD. Moléculas assimétricas.")

# =======================================================
# EXECUÇÃO DO APLICATIVO
# =======================================================
if __name__ == "__main__":
    app = DockingAppTk()
    app.mainloop()