# ==============================================================================
# SEÇÃO: IMPORTAÇÃO DE BIBLIOTECAS E CONFIGURAÇÕES INICIAIS
# Descrição: Esta seção importa todas as bibliotecas Python necessárias para
#            o funcionamento do script e define algumas configurações globais
#            para a visualização de gráficos.
# ==============================================================================

import pandas as pd  # Importa a biblioteca pandas para manipulação de dados tabulares (DataFrames).
import numpy as np  # Importa a biblioteca numpy para operações numéricas, especialmente arrays.
import matplotlib.pyplot as plt  # Importa pyplot da matplotlib para criação de gráficos estáticos.
import seaborn as sns  # Importa a biblioteca seaborn para visualizações estatísticas mais elaboradas.
import random  # Importa a biblioteca random para geração de números e seleções aleatórias.
import os  # Importa a biblioteca os para interagir com o sistema operacional, como manipulação de caminhos de arquivos.

# --- Configurações Globais de Visualização ---
sns.set_style('whitegrid')  # Define o estilo dos gráficos seaborn para 'whitegrid' (fundo branco com grades).
plt.rcParams['figure.figsize'] = (12, 7)  # Define o tamanho padrão das figuras matplotlib para 12x7 polegadas.

# ==============================================================================
# SEÇÃO: CARREGAMENTO DO DATASET ORIGINAL
# Descrição: Esta seção define o caminho do arquivo CSV contendo os dados
#            originais e tenta carregá-lo em um DataFrame pandas.
#            Inclui tratamento de erros para arquivo não encontrado ou
#            problemas de formatação.
# ==============================================================================

# --- Definição do Caminho do Arquivo ---
nome_arquivo = 'C:\\Users\\maiqu\\Documents\\Mestrado\\01 - Aprendizado de Maquina\\Trabalho final\\Dataset\\link_r7tjn68rmw-1\\r7tjn68rmw-1\\Original\\SOIL DATA GR.csv'  # Define o caminho completo para o arquivo CSV. Substitua pelo seu caminho, se necessário.

# --- Tentativa de Leitura do Arquivo CSV ---
try:  # Inicia um bloco de tratamento de exceções.
    df_original = pd.read_csv(nome_arquivo, sep=';',
                              decimal=',')  # Lê o arquivo CSV, especificando ';' como separador de colunas e ',' como separador decimal.
    print("Dataset original carregado com sucesso!")  # Imprime mensagem de sucesso se o carregamento ocorrer bem.
except FileNotFoundError:  # Captura a exceção se o arquivo não for encontrado.
    print(
        f"Erro: O arquivo '{nome_arquivo}' não foi encontrado. Verifique o caminho.")  # Imprime mensagem de erro específica para arquivo não encontrado.
    exit()  # Termina a execução do script.
except Exception as e:  # Captura qualquer outra exceção que possa ocorrer durante o carregamento.
    print(f"Ocorreu um erro ao carregar o CSV: {e}")  # Imprime a mensagem de erro da exceção capturada.
    exit()  # Termina a execução do script.

print(
    f"\nDimensões originais do dataset: {df_original.shape[0]} linhas, {df_original.shape[1]} colunas.")  # Imprime as dimensões (linhas, colunas) do DataFrame original.

# ==============================================================================
# SEÇÃO: LIMPEZA INICIAL DO DATASET - TRATAMENTO DE VALORES NULOS E ESPAÇOS
# Descrição: Esta seção realiza a limpeza inicial dos dados, removendo colunas
#            e linhas que contêm apenas valores nulos. Também remove espaços
#            em branco extras dos nomes das colunas.
# ==============================================================================

# --- Remoção de Colunas Totalmente Nulas ---
colunas_antes_remocao = df_original.columns.tolist()  # Armazena a lista de nomes das colunas antes de qualquer remoção.
df = df_original.dropna(axis='columns',
                        how='all')  # Remove colunas (axis='columns') onde todos os valores são nulos (how='all').
colunas_depois_remocao = df.columns.tolist()  # Armazena a lista de nomes das colunas após a remoção.
colunas_removidas_df = [col for col in colunas_antes_remocao if
                        col not in colunas_depois_remocao]  # Identifica as colunas que foram removidas.
if colunas_removidas_df:  # Verifica se alguma coluna foi removida.
    print(f"\nColunas totalmente nulas removidas: {colunas_removidas_df}")  # Imprime a lista de colunas removidas.

# --- Remoção de Linhas Totalmente Nulas ---
df = df.dropna(how='all', axis=0)  # Remove linhas (axis=0) onde todos os valores são nulos (how='all').
print(
    f"Dimensões do dataset após remover colunas e linhas totalmente nulas: {df.shape[0]} linhas, {df.shape[1]} colunas.")  # Imprime as novas dimensões do DataFrame.

# --- Verificação se o DataFrame está Vazio ---
if df.empty:  # Verifica se o DataFrame ficou vazio após as remoções.
    print("O DataFrame está vazio após a remoção de nulos. Não é possível prosseguir.")  # Imprime mensagem de aviso.
    exit()  # Termina a execução do script.

# --- Limpeza de Espaços nos Nomes das Colunas ---
df.columns = df.columns.str.strip()  # Remove espaços em branco do início e do fim de cada nome de coluna.
print("\nNomes das colunas após limpeza de espaços:",
      df.columns.tolist())  # Imprime a lista de nomes de colunas atualizada.

# ==============================================================================
# SEÇÃO: DEFINIÇÃO DE CRITÉRIOS E CLASSIFICAÇÃO INICIAL PARA MILHO
# Descrição: Define os critérios agronômicos para a cultura do milho.
#            Cria funções para calcular a pontuação de adequação de cada amostra
#            de solo e para classificar essa pontuação em categorias
#            (Baixa, Média, Alta adequacao). Aplica essas funções ao dataset.
# ==============================================================================

# --- Definição dos Critérios para Milho ---
criterios_milho = {  # Dicionário contendo os atributos do solo e suas faixas ideais para milho.
    'pH': (5.5, 6.5), 'Sand %': (30, 50), 'Clay %': (20, 35), 'Silt %': (20, 40),
    'EC mS/cm': (0, 1), 'O.M. %': (2.0, float('inf')), 'CACO3 %': (0, 5),
    'N_NO3 ppm': (20, float('inf')), 'P ppm': (12, float('inf')), 'K ppm': (120, float('inf')),
    'Mg ppm': (50, 150), 'Fe ppm': (4, 8), 'Zn ppm': (1, 2), 'Mn ppm': (5, 20),
    'Cu ppm': (0.5, 2), 'B ppm': (0.5, 1.5)
}


# --- Função para Calcular Pontuação de Adequação ---
def calcular_pontuacao_amostra(amostra, criterios):
    """
    Calcula a pontuação de adequação de uma amostra de solo com base em critérios definidos.
    Args:
        amostra (pd.Series ou dict): Amostra de solo.
        criterios (dict): Dicionário de critérios.
    Returns:
        int: Pontuação da amostra.
    """
    pontuacao = 0
    for atributo, faixa_crit in criterios.items():
        if atributo in amostra and pd.notna(amostra[atributo]):
            valor_atributo = amostra[atributo]
            try:
                valor_atributo_num = float(valor_atributo)
                if faixa_crit[1] != float('inf'):
                    if faixa_crit[0] <= valor_atributo_num <= faixa_crit[1]:
                        pontuacao += 1
                else:
                    if valor_atributo_num >= faixa_crit[0]:
                        pontuacao += 1
            except (ValueError, TypeError):
                pass
    return pontuacao


df['Pontuacao_Milho'] = df.apply(lambda row: calcular_pontuacao_amostra(row, criterios_milho), axis=1)


# --- Função para Classificar a Adequação Baseada na Pontuação ---
def classificar_adequacao_milho(pontuacao):
    """
    Classifica a adequação do solo para milho com base na pontuação.
    Args:
        pontuacao (float ou int): Pontuação de adequação.
    Returns:
        str: Classe de adequação.
    """
    if pd.isna(pontuacao): return 'N/A_Pontuacao_Ausente'
    if 0 <= pontuacao <= 5:
        return 'Baixa adequacao'
    elif 6 <= pontuacao <= 11:
        return 'Media adequacao'
    elif 12 <= pontuacao <= len(criterios_milho):
        return 'Alta adequacao'
    return 'Indefinido_Fora_Faixa'


ordem_desejada = ['Baixa adequacao', 'Media adequacao', 'Alta adequacao']
df['Adequacao_Milho'] = df['Pontuacao_Milho'].apply(classificar_adequacao_milho)

# ==============================================================================
# SEÇÃO: ANÁLISE DA DISTRIBUIÇÃO DAS CLASSES NO DATASET ORIGINAL
# Descrição: Exibe a contagem e percentual das classes e gera um gráfico.
# ==============================================================================
print("\n--- PERCENTUAL DE CADA CLASSIFICAÇÃO PARA MILHO (ORIGINAL) ---")
contagem_classes_milho_original = df['Adequacao_Milho'].value_counts().reindex(ordem_desejada, fill_value=0)
if not contagem_classes_milho_original.empty:
    percentual_classes_milho_original = (df['Adequacao_Milho'].value_counts(normalize=True) * 100).reindex(
        ordem_desejada, fill_value=0)
    print("Contagem por classe de adequação para Milho (Original):")
    print(contagem_classes_milho_original)
    print("\nPercentual por classe de adequação para Milho (Original) (%):")
    print(percentual_classes_milho_original.round(3))
    plt.figure(figsize=(10, 7))
    ax_orig = sns.countplot(x=df['Adequacao_Milho'], hue=df['Adequacao_Milho'], order=ordem_desejada, palette="viridis",
                            legend=False)
    plt.title('Distribuição Original da Adequação do Solo para Milho', fontsize=15)
    plt.xlabel('Classe de Adequação', fontsize=12);
    plt.ylabel('Número de Amostras', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    total_amostras_original_plot = len(df['Adequacao_Milho'].dropna())
    for p in ax_orig.patches:
        altura = p.get_height()
        percent = (altura / total_amostras_original_plot) * 100 if total_amostras_original_plot > 0 else 0
        ax_orig.text(p.get_x() + p.get_width() / 2., altura + total_amostras_original_plot * 0.005,
                     f'{altura}\n({percent:.1f}%)', ha='center', va='bottom', fontsize=10)
    plt.tight_layout();
    plt.show()
else:
    print("Não foi possível calcular a contagem/percentual das classes de adequação no dataset original.")

# ==============================================================================
# SEÇÃO: PREPARAÇÃO PARA GERAÇÃO DE DADOS SINTÉTICOS - CÁLCULO DE ESTATÍSTICAS
# Descrição: Calcula estatísticas descritivas (std, min, max) para colunas
#            numéricas, que serão usadas para a geração de ruído.
# ==============================================================================
print("\n--- PREPARANDO PARA GERAÇÃO DE DADOS SINTÉTICOS (MÉTODO DE RUÍDO GAUSSIANO) ---")

# Coletar estatísticas do DataFrame original para guiar a adição de ruído
original_df_stats = {}  # Dicionário para armazenar estatísticas por coluna
colunas_para_ruido = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col]) and col not in ['ID',
                                                                                                          'Pontuacao_Milho']]  # Colunas numéricas candidatas a adição de ruído

for col in colunas_para_ruido:  # Itera sobre as colunas numéricas
    if not df[col].dropna().empty:  # Verifica se a coluna tem valores não nulos
        original_df_stats[col] = {  # Armazena estatísticas relevantes
            'std': df[col].std(skipna=True),  # Desvio padrão
            'min_orig': df[col].min(skipna=True),  # Mínimo original
            'max_orig': df[col].max(skipna=True)  # Máximo original
        }
    else:  # Caso a coluna seja inteiramente nula ou não tenha dados
        original_df_stats[col] = {  # Define valores default para evitar erros
            'std': 0,  # Desvio padrão zero
            'min_orig': 0,  # Mínimo zero
            'max_orig': 0  # Máximo zero
        }


# ==============================================================================
# SEÇÃO: FUNÇÃO PARA GERAR AMOSTRA SINTÉTICA COM RUÍDO GAUSSIANO CONTROLADO
# Descrição: Define a função principal para criar uma nova amostra sintética.
#            Esta função pega uma amostra base, adiciona ruído gaussiano
#            controlado aos seus atributos numéricos e lida com as restrições
#            especiais para 'Sand %', 'Clay %' e 'Silt %'.
# ==============================================================================

def gerar_amostra_com_ruido_gaussiano(amostra_base_series, df_stats, cols_numericas_com_ruido,
                                      cols_textura_scl, fracao_std_ruido,
                                      criterios_textura, classe_alvo_para_textura_especial):
    """
    Gera uma nova amostra sintética adicionando ruído Gaussiano a uma amostra base.

    Aplica ruído Gaussiano controlado (fração do desvio padrão) aos atributos numéricos.
    Possui tratamento especial para as colunas de textura ('Sand %', 'Clay %', 'Silt %'):
    - Se a `classe_alvo_para_textura_especial` for 'Alta adequacao', tenta gerar
      valores para textura que atendam aos `criterios_textura` E somem 100%.
    - Caso contrário, ou se a tentativa acima falhar, adiciona ruído aos valores de textura
      da amostra base e depois normaliza-os para somar 100%.

    Args:
        amostra_base_series (pd.Series): A amostra original da qual a nova será derivada.
        df_stats (dict): Dicionário com estatísticas ('std', 'min_orig', 'max_orig') por coluna.
        cols_numericas_com_ruido (list): Lista de colunas numéricas onde o ruído será aplicado
                                         (excluindo colunas de textura inicialmente).
        cols_textura_scl (list): Lista com os nomes das colunas de textura (ex: ['Sand %', 'Clay %', 'Silt %']).
        fracao_std_ruido (float): Fator multiplicativo para o desvio padrão, controlando a magnitude do ruído.
        criterios_textura (dict): Dicionário com os critérios específicos para as colunas de textura (usado para 'Alta adequacao').
        classe_alvo_para_textura_especial (str): A classe de adequação alvo. Se for 'Alta adequacao',
                                                  ativa a lógica especial para textura.
    Returns:
        dict: Um dicionário representando a nova amostra sintética gerada.
    """
    nova_amostra_dict = amostra_base_series.to_dict()  # Converte a amostra base para um dicionário para facilitar modificações.

    # --- Adicionar Ruído Gaussiano aos Atributos Numéricos (exceto textura por enquanto) ---
    for col in cols_numericas_com_ruido:  # Itera sobre as colunas designadas para adição de ruído.
        if col in nova_amostra_dict and pd.notna(
                nova_amostra_dict[col]):  # Verifica se a coluna existe na amostra e tem valor.
            if col in df_stats and pd.notna(df_stats[col]['std']) and df_stats[col][
                'std'] > 0:  # Verifica se há desvio padrão válido.
                desvio_ruido = df_stats[col]['std'] * fracao_std_ruido  # Calcula o desvio padrão do ruído.
                ruido = random.gauss(0, desvio_ruido)  # Gera ruído Gaussiano com média 0.
                valor_com_ruido = nova_amostra_dict[col] + ruido  # Adiciona o ruído ao valor original.
                # --- Garante que o valor com ruído permaneça dentro dos limites originais do dataset ---
                min_o = df_stats[col]['min_orig']  # Mínimo original da coluna.
                max_o = df_stats[col]['max_orig']  # Máximo original da coluna.
                nova_amostra_dict[col] = np.clip(valor_com_ruido, min_o, max_o)  # Limita o valor entre min_o e max_o.

    # --- Tratamento Especial para Colunas de Textura ('Sand %', 'Clay %', 'Silt %') ---
    sand_col, clay_col, silt_col = cols_textura_scl[0], cols_textura_scl[1], cols_textura_scl[2]  # Nomes das colunas.

    # Condição para tratamento especial de textura: ser 'Alta adequacao' E as colunas de textura existirem nos critérios.
    tentar_textura_especial = (classe_alvo_para_textura_especial == 'Alta adequacao' and
                               all(c in criterios_textura for c in cols_textura_scl))

    if tentar_textura_especial:  # Se deve tentar a lógica especial para 'Alta adequacao'.
        s_min, s_max = criterios_textura[sand_col]  # Critérios para Areia.
        c_min, c_max = criterios_textura[clay_col]  # Critérios para Argila.
        l_min, l_max = criterios_textura[silt_col]  # Critérios para Silte.
        textura_gerada_com_sucesso = False  # Flag.
        for _ in range(500):  # Tenta gerar combinação válida.
            s_val = random.uniform(s_min, s_max)  # Gera Areia.
            c_val_lower = max(c_min, 100.0 - s_val - l_max)  # Limite inferior para Argila.
            c_val_upper = min(c_max, 100.0 - s_val - l_min)  # Limite superior para Argila.
            if c_val_lower <= c_val_upper:  # Se Argila é possível.
                c_val = random.uniform(c_val_lower, c_val_upper)  # Gera Argila.
                l_val = 100.0 - s_val - c_val  # Calcula Silte.
                if l_min <= l_val <= l_max and abs(s_val + c_val + l_val - 100.0) < 1e-5:  # Se Silte válido e soma 100.
                    nova_amostra_dict[sand_col] = s_val  # Atribui Areia.
                    nova_amostra_dict[clay_col] = c_val  # Atribui Argila.
                    nova_amostra_dict[silt_col] = l_val  # Atribui Silte.
                    textura_gerada_com_sucesso = True  # Sucesso.
                    break  # Para as tentativas.
        if not textura_gerada_com_sucesso:  # Fallback se a geração especial falhou.
            # Adiciona ruído aos valores base e normaliza (pode não atender aos critérios de 'Alta adequacao').
            # Usar os valores da amostra_base_series como ponto de partida para o ruído aqui.
            s_base = amostra_base_series.get(sand_col, 33.3)  # Default se não existir.
            c_base = amostra_base_series.get(clay_col, 33.3)  # Default.
            l_base = amostra_base_series.get(silt_col, 33.4)  # Default.

            s_std = df_stats.get(sand_col, {}).get('std', 0);
            ruido_s = random.gauss(0, s_std * fracao_std_ruido) if s_std > 0 else 0
            c_std = df_stats.get(clay_col, {}).get('std', 0);
            ruido_c = random.gauss(0, c_std * fracao_std_ruido) if c_std > 0 else 0
            l_std = df_stats.get(silt_col, {}).get('std', 0);
            ruido_l = random.gauss(0, l_std * fracao_std_ruido) if l_std > 0 else 0

            s_com_ruido = max(0, s_base + ruido_s)  # Garante não negativo.
            c_com_ruido = max(0, c_base + ruido_c)  # Garante não negativo.
            l_com_ruido = max(0, l_base + ruido_l)  # Garante não negativo.

            soma_textura_com_ruido = s_com_ruido + c_com_ruido + l_com_ruido  # Soma.
            if soma_textura_com_ruido > 1e-5:  # Se a soma não for zero.
                nova_amostra_dict[sand_col] = (s_com_ruido / soma_textura_com_ruido) * 100.0  # Normaliza Areia.
                nova_amostra_dict[clay_col] = (c_com_ruido / soma_textura_com_ruido) * 100.0  # Normaliza Argila.
                nova_amostra_dict[silt_col] = 100.0 - nova_amostra_dict[sand_col] - nova_amostra_dict[
                    clay_col]  # Ajusta Silte.
            else:  # Se a soma for zero (improvável com defaults > 0).
                nova_amostra_dict[sand_col], nova_amostra_dict[clay_col], nova_amostra_dict[
                    silt_col] = 33.33, 33.33, 33.34  # Distribuição igual.
    else:  # Caso geral (não é 'Alta adequacao' para textura ou critério de textura não existe).
        # Adiciona ruído aos valores de textura da amostra base e normaliza para somar 100%.
        s_base = amostra_base_series.get(sand_col, 33.3)  # Default.
        c_base = amostra_base_series.get(clay_col, 33.3)  # Default.
        l_base = amostra_base_series.get(silt_col, 33.4)  # Default.

        s_std = df_stats.get(sand_col, {}).get('std', 0);
        ruido_s = random.gauss(0, s_std * fracao_std_ruido) if s_std > 0 else 0
        c_std = df_stats.get(clay_col, {}).get('std', 0);
        ruido_c = random.gauss(0, c_std * fracao_std_ruido) if c_std > 0 else 0
        l_std = df_stats.get(silt_col, {}).get('std', 0);
        ruido_l = random.gauss(0, l_std * fracao_std_ruido) if l_std > 0 else 0

        s_com_ruido = max(0, s_base + ruido_s)  # Garante não negativo.
        c_com_ruido = max(0, c_base + ruido_c)  # Garante não negativo.
        l_com_ruido = max(0, l_base + ruido_l)  # Garante não negativo.

        soma_textura_com_ruido = s_com_ruido + c_com_ruido + l_com_ruido  # Soma.
        if soma_textura_com_ruido > 1e-5:  # Se a soma não for zero.
            nova_amostra_dict[sand_col] = (s_com_ruido / soma_textura_com_ruido) * 100.0  # Normaliza Areia.
            nova_amostra_dict[clay_col] = (c_com_ruido / soma_textura_com_ruido) * 100.0  # Normaliza Argila.
            nova_amostra_dict[silt_col] = 100.0 - nova_amostra_dict[sand_col] - nova_amostra_dict[
                clay_col]  # Ajusta Silte.
        else:  # Se a soma for zero.
            nova_amostra_dict[sand_col], nova_amostra_dict[clay_col], nova_amostra_dict[
                silt_col] = 33.33, 33.33, 33.34  # Distribuição igual.

    return nova_amostra_dict  # Retorna a amostra sintética com ruído.


# ==============================================================================
# SEÇÃO: GERAÇÃO PRINCIPAL DE DADOS SINTÉTICOS COM VALIDAÇÃO (USANDO RUÍDO GAUSSIANO)
# Descrição: Itera sobre cada classe de adequação alvo, determina quantas
#            amostras sintéticas são necessárias. Para cada uma, seleciona uma
#            amostra base da classe alvo, aplica ruído gaussiano controlado e
#            valida se a nova amostra ainda pertence à classe alvo.
# ==============================================================================
dados_sinteticos_lista = []  # Lista para armazenar as amostras sintéticas geradas.
mapa_pontuacao_classe = {  # Mapeamento de classes para faixas de pontuação.
    'Baixa adequacao': (0, 5),
    'Media adequacao': (6, 11),
    'Alta adequacao': (12, len(criterios_milho))
}
proximo_id_sintetico = 782  # ID inicial para dados sintéticos.
MAX_RETRIES_PER_INDIVIDUAL_SAMPLE = 30  # Máximo de tentativas para gerar UMA amostra válida.
FRACAO_STD_RUIDO = 0.05  # Fração do desvio padrão a ser usada como magnitude do ruído (ex: 5%).

if not contagem_classes_milho_original.empty:  # Se o dataset original tem contagens de classe.
    contagem_alvo_por_classe = contagem_classes_milho_original.max()  # Define o alvo como a contagem da classe majoritária.
    if contagem_alvo_por_classe == 0 and not df.empty:  # Ajuste se a classe majoritária for 0.
        contagem_alvo_por_classe = max(10,
                                       len(df) // len(mapa_pontuacao_classe) if len(mapa_pontuacao_classe) > 0 else 10)
    elif df.empty:
        contagem_alvo_por_classe = 10  # Default se original vazio.
    print(f"\nContagem alvo por classe para balanceamento: {contagem_alvo_por_classe}")

    for classe_alvo_geracao, _ in mapa_pontuacao_classe.items():  # Itera sobre as classes alvo.
        contagem_atual_da_classe = contagem_classes_milho_original.get(classe_alvo_geracao, 0)  # Contagem original.
        num_sinteticos_necessarios = max(0,
                                         contagem_alvo_por_classe - contagem_atual_da_classe)  # Nº de sintéticos a gerar.
        print(
            f"  Classe '{classe_alvo_geracao}': Atual = {contagem_atual_da_classe}, Sintéticos necessários = {num_sinteticos_necessarios}")

        sinteticos_adicionados_para_esta_classe = 0  # Contador para esta classe.
        total_geral_tentativas_para_classe = 0  # Contador de tentativas para esta classe.

        amostras_base_da_classe = df[
            df['Adequacao_Milho'] == classe_alvo_geracao]  # Filtra amostras originais da classe alvo.
        if amostras_base_da_classe.empty and num_sinteticos_necessarios > 0:  # Se não há amostras base.
            print(
                f"    Aviso: Nenhuma amostra base encontrada para a classe '{classe_alvo_geracao}'. Não é possível gerar sintéticos com ruído para esta classe.")
            continue  # Pula para a próxima classe.

        while sinteticos_adicionados_para_esta_classe < num_sinteticos_necessarios:  # Loop até gerar o necessário.
            if total_geral_tentativas_para_classe > num_sinteticos_necessarios * MAX_RETRIES_PER_INDIVIDUAL_SAMPLE * 1.5:  # Limite de segurança.
                print(f"    Aviso: Limite total de tentativas para a classe '{classe_alvo_geracao}' atingido.")
                break  # Para esta classe.

            tentativas_para_este_ponto_especifico = 0  # Tentativas para esta amostra específica.
            amostra_valida_encontrada_para_este_ponto = False  # Flag.

            while tentativas_para_este_ponto_especifico < MAX_RETRIES_PER_INDIVIDUAL_SAMPLE:  # Loop de retentativa.
                total_geral_tentativas_para_classe += 1  # Incrementa tentativas.

                amostra_base_selecionada = amostras_base_da_classe.sample(1).iloc[
                    0]  # Seleciona uma amostra base aleatória.

                # --- Geração da Amostra com Ruído ---
                amostra_bruta_dict = gerar_amostra_com_ruido_gaussiano(  # Chama a nova função.
                    amostra_base_selecionada,
                    original_df_stats,
                    colunas_para_ruido,  # Lista de colunas numéricas onde adicionar ruído
                    ['Sand %', 'Clay %', 'Silt %'],  # Nomes das colunas de textura
                    FRACAO_STD_RUIDO,
                    criterios_milho,  # Passa todos os critérios, a função interna usará para textura se necessário
                    classe_alvo_geracao  # Passa a classe alvo para a lógica especial de textura
                )

                # --- Validação Imediata da Amostra Gerada ---
                pontuacao_obtida = calcular_pontuacao_amostra(amostra_bruta_dict, criterios_milho)  # Calcula pontuação.
                classificacao_obtida = classificar_adequacao_milho(pontuacao_obtida)  # Classifica.
                tentativas_para_este_ponto_especifico += 1  # Incrementa tentativas para este ponto.

                if classificacao_obtida == classe_alvo_geracao:  # Se a classificação bate com o alvo.
                    amostra_bruta_dict['ID'] = proximo_id_sintetico;
                    proximo_id_sintetico += 1  # ID.
                    amostra_bruta_dict['FonteDados'] = 'Sintetico';
                    amostra_bruta_dict['ClasseAlvoGeracao'] = classe_alvo_geracao  # Metadados.
                    amostra_bruta_dict['Pontuacao_Milho'] = pontuacao_obtida  # Pontuação verificada.
                    amostra_bruta_dict['Adequacao_Milho'] = classificacao_obtida  # Classificação verificada.
                    dados_sinteticos_lista.append(amostra_bruta_dict)  # Adiciona à lista.
                    sinteticos_adicionados_para_esta_classe += 1  # Incrementa contador da classe.
                    amostra_valida_encontrada_para_este_ponto = True  # Sucesso.
                    break  # Sai do loop de retentativa.

            if not amostra_valida_encontrada_para_este_ponto:  # Se não encontrou amostra válida para este ponto específico.
                # Se falhou muitas vezes para este ponto, o loop while externo continuará para o próximo "necessário".
                # Não há um "break" aqui, pois queremos tentar preencher todos os `num_sinteticos_necessarios`
                # A mensagem de alerta final da classe indicará se o total não foi atingido.
                pass

        if sinteticos_adicionados_para_esta_classe < num_sinteticos_necessarios:  # Checagem final para a classe.
            print(
                f"  Alerta Final para Classe '{classe_alvo_geracao}': Gerou apenas {sinteticos_adicionados_para_esta_classe} de {num_sinteticos_necessarios} amostras desejadas.")
else:
    print("Dataset original não tem classificações para basear a geração sintética.")

df_sinteticos = pd.DataFrame(dados_sinteticos_lista)  # Cria DataFrame com os sintéticos.

# ==============================================================================
# SEÇÃO: COMBINAÇÃO DOS DATASETS E RECALCULO FINAL DA CLASSIFICAÇÃO
# Descrição: Combina o DataFrame original com o DataFrame de dados sintéticos.
#            Em seguida, recalcula a pontuação e a classificação de adequação
#            para todas as amostras no DataFrame combinado.
# ==============================================================================
if not df_sinteticos.empty:  # Se foram gerados sintéticos.
    df_com_fonte = df.assign(FonteDados='Original', ClasseAlvoGeracao=df['Adequacao_Milho'])  # Prepara original.
    for col in df_com_fonte.columns:  # Alinha colunas.
        if col not in df_sinteticos.columns:
            df_sinteticos[col] = np.nan
    if not df_sinteticos.empty:
        df_sinteticos = df_sinteticos.reindex(columns=df_com_fonte.columns)
    df_combinado = pd.concat([df_com_fonte, df_sinteticos], ignore_index=True)  # Concatena.
    num_sint_efetivos = len(
        df_sinteticos[df_sinteticos['FonteDados'] == 'Sintetico']) if 'FonteDados' in df_sinteticos.columns else 0
    print(f"\n{num_sint_efetivos} amostras sintéticas foram criadas e adicionadas.")
else:  # Se não foram gerados sintéticos.
    df_combinado = df.assign(FonteDados='Original', ClasseAlvoGeracao=df['Adequacao_Milho'])  # Combinado é só original.
    print("\nNenhuma amostra sintética foi gerada.")

df_combinado['Pontuacao_Milho'] = df_combinado.apply(lambda row: calcular_pontuacao_amostra(row, criterios_milho),
                                                     axis=1)  # Recalcula pontuação.
df_combinado['Adequacao_Milho'] = df_combinado['Pontuacao_Milho'].apply(
    classificar_adequacao_milho)  # Recalcula classificação.

# ==============================================================================
# SEÇÃO: ANÁLISE DA DISTRIBUIÇÃO DAS CLASSES NO DATASET COMBINADO
# Descrição: Exibe contagem e percentual das classes no dataset combinado e gera gráfico.
# ==============================================================================
print("\n--- PERCENTUAL DE CADA CLASSIFICAÇÃO PARA MILHO (COMBINADO FINAL) ---")
contagem_classes_combinado = df_combinado['Adequacao_Milho'].value_counts().reindex(ordem_desejada, fill_value=0)
if not contagem_classes_combinado.empty:
    percentual_classes_combinado = (df_combinado['Adequacao_Milho'].value_counts(normalize=True) * 100).reindex(
        ordem_desejada, fill_value=0)
    print("Contagem por classe de adequação para Milho (Dataset Combinado):")
    print(contagem_classes_combinado)
    print("\nPercentual por classe de adequação para Milho (Dataset Combinado) (%):")
    print(percentual_classes_combinado.round(3))
    plt.figure(figsize=(10, 7))
    ax_comb = sns.countplot(x=df_combinado['Adequacao_Milho'], hue=df_combinado['Adequacao_Milho'],
                            order=ordem_desejada, palette="magma", legend=False)
    plt.title('Distribuição da Adequação do Solo para Milho (com Dados Sintéticos)', fontsize=15)
    plt.xlabel('Classe de Adequação', fontsize=12);
    plt.ylabel('Número de Amostras', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    total_amostras_combinado_plot = len(df_combinado['Adequacao_Milho'].dropna())
    for p in ax_comb.patches:
        altura = p.get_height()
        percent = (altura / total_amostras_combinado_plot) * 100 if total_amostras_combinado_plot > 0 else 0
        ax_comb.text(p.get_x() + p.get_width() / 2., altura + total_amostras_combinado_plot * 0.005,
                     f'{altura}\n({percent:.1f}%)', ha='center', va='bottom', fontsize=10)
    plt.tight_layout();
    plt.show()
else:
    print("Não foi possível calcular a contagem/percentual das classes no dataset combinado.")

# ==============================================================================
# SEÇÃO: SALVAMENTO DOS DADOS SINTÉTICOS GERADOS
# Descrição: Filtra e salva os dados sintéticos em um novo arquivo CSV.
# ==============================================================================
df_sinteticos_para_salvar = df_combinado[df_combinado['FonteDados'] == 'Sintetico'].copy()  # Filtra sintéticos.
if not df_sinteticos_para_salvar.empty:  # Se há sintéticos para salvar.
    print(f"\n--- {len(df_sinteticos_para_salvar)} DADOS SINTÉTICOS CRIADOS (APÓS RECALCULO FINAL) ---")
    colunas_prefixo_ordenado = ['ID', 'FonteDados', 'ClasseAlvoGeracao', 'Pontuacao_Milho',
                                'Adequacao_Milho']  # Ordem das colunas.
    colunas_restantes_df_sinteticos = [col for col in df_sinteticos_para_salvar.columns if
                                       col not in colunas_prefixo_ordenado]
    colunas_finais_para_sinteticos = colunas_prefixo_ordenado + colunas_restantes_df_sinteticos
    colunas_finais_para_sinteticos = [col for col in colunas_finais_para_sinteticos if
                                      col in df_sinteticos_para_salvar.columns]
    print(f"Mostrando as primeiras {min(20, len(df_sinteticos_para_salvar))} amostras sintéticas:")
    print(df_sinteticos_para_salvar[colunas_finais_para_sinteticos].head(min(20, len(df_sinteticos_para_salvar))))
    if len(df_sinteticos_para_salvar) > 20: print("...")
    try:
        diretorio_base = os.path.dirname(nome_arquivo) if nome_arquivo and os.path.dirname(
            nome_arquivo) else os.getcwd()
        nome_arquivo_sintetico = "Arquivos_Sinteticos.csv"  # Nome do arquivo.
        caminho_saida_sinteticos = os.path.join(diretorio_base, nome_arquivo_sintetico)  # Caminho completo.
        df_sinteticos_para_salvar[colunas_finais_para_sinteticos].to_csv(
            caminho_saida_sinteticos, index=False, sep=';', decimal=','  # Salva.
        )
        print(f"\nDados sintéticos salvos com sucesso em: {caminho_saida_sinteticos}")
    except Exception as e:
        print(f"\nErro ao salvar os dados sintéticos: {e}")
else:
    print("\nNenhum dado sintético foi gerado/encontrado para salvar.")

print("\n--- FIM DO SCRIPT ---")