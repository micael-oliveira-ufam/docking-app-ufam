# 🧬 DockEasy-UFAM (Ferramenta automatizada de docking molecular com AutoDock Vina v.1.2.7)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![Bioinformatics](https://img.shields.io/badge/Bioinformatics-CADD-brightgreen)
![Status](https://img.shields.io/badge/Status-Ativo-success)

Plataforma unificada para **Planejamento de Fármacos Baseado em Estrutura (SBDD)**, desenvolvida para unir o rigor da pesquisa de bancada à didática necessária para o ensino acadêmico.

👤 **Autoria:** Micael Davi Lima de Oliveira (Iniciação Científica)  
🏛️ **Instituição:** Faculdade de Ciências Farmacêuticas (FCF) - Universidade Federal do Amazonas (UFAM)

---

## 🎯 Finalidade: Ensino e Pesquisa

Este projeto nasceu da necessidade de democratizar o acesso à **Quimioinformática** e à **Biologia Estrutural Computacional**. A ferramenta possui um duplo propósito fundamental:

1. **Para Pesquisa (Bancada/Iniciação Científica):** Oferece um pipeline automatizado de *High-Throughput Virtual Screening* (HTVS), com cálculos termodinâmicos realizados rigorosamente em **triplicata estocástica** e ferramentas de validação (cálculo de RMSD nativo). Reduz o tempo de preparação de dezenas de ligantes e automatiza as etapas complexas do AutoDock Vina.
2. **Para o Ensino (Salas de Aula):** A interface foi desenhada como um "Laboratório Virtual". Cada aba possui blocos expansíveis de **Fundamentos de Química Medicinal**, explicando o "porquê" físico, químico e biológico por trás de cada clique (cargas de Gasteiger, otimização MMFF94, campos de força, etc.), tornando-se uma ferramenta didática poderosa para disciplinas de Química Farmacêutica.

---

## 💡 A Importância do Design de Fármacos Assistido por Computador (CADD)

O desenvolvimento tradicional de um novo fármaco é um processo que pode levar mais de uma década e custar bilhões de dólares, apresentando altas taxas de falha nas fases clínicas. 

O **CADD (Computer-Aided Drug Design)** revoluciona este cenário ao atuar como um filtro racional (Virtual Screening). Através do Docking Molecular, somos capazes de prever a conformação tridimensional preferencial (a *Pose*) de uma molécula e calcular sua **Afinidade de Ligação (Energia Livre de Gibbs - $\Delta G$)** com o alvo macromolecular. Isso permite que pesquisadores testem milhares de moléculas *in silico* de forma rápida e barata, selecionando apenas os compostos mais promissores e termodinamicamente viáveis para seguirem aos testes *in vitro* e *in vivo*, poupando tempo, recursos e vidas.

---

## ✨ Principais Funcionalidades

* **🧬 Preparação do Receptor:** Download automatizado via RCSB PDB, limpeza conservadora para evitar distorções espaciais, protonação em pH fisiológico (7.4) via PDBFixer e cálculo de cargas de Gasteiger.
* **💊 Processamento de Ligantes:** * Busca de moléculas inéditas via SMILES e otimização geométrica (MMFF94).
    * Extração inteligente de ligantes co-cristalizados para validação de protocolos (Re-docking).
    * **Módulo HTVS:** Upload em lote de bibliotecas (`.sdf`, `.mol2`), com tratamento de erros automático que ignora moléculas corrompidas sem travar a triagem.
* **📦 Grid Box Dinâmica:** Cálculo automatizado das fronteiras terapêuticas para *Site-Directed Docking* (via LaBOX) ou cálculo de Centro de Massa para *Blind Docking* global (via Biopython).
* **🚀 Motor Termodinâmico:** Integração perfeita com o AutoDock Vina, executando simulações rigorosas em **triplicata** para mitigação de viés estocástico.
* **📊 Análise e Síntese:** Geração de tabelas de médias e desvio padrão. Síntese do complexo proteico (HETATM/UNL) pronto para exportação para renderizadores padrão-ouro (*Discovery Studio, LigPlot+, Maestro*). Inclui cálculo automático de **RMSD** para validação de re-docking.

---

## ⚙️ Instalação e Configuração

Recomenda-se fortemente a utilização de um ambiente virtual isolado para evitar conflitos de dependências no seu sistema operacional.

**1. Clone o repositório:**
```bash
git clone [https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git](https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git)
cd SEU-REPOSITORIO
