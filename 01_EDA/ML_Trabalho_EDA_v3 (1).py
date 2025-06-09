# ==============================================================================
# SEÇÃO 1: IMPORTAÇÃO DE BIBLIOTECAS
# ==============================================================================
# Esta seção importa todas as bibliotecas Python necessárias para a execução do script.
# pandas é usado para manipulação de dados, numpy para operações numéricas,
# matplotlib.pyplot para criação de gráficos e seaborn para visualizações estatísticas aprimoradas.

import pandas as pd                     # Importa a biblioteca pandas e a apelida de 'pd' para manipulação de DataFrames.
import numpy as np                      # Importa a biblioteca numpy e a apelida de 'np' para operações numéricas, especialmente com arrays.
import matplotlib.pyplot as plt         # Importa o submódulo pyplot da biblioteca matplotlib e o apelida de 'plt' para criar gráficos.
import seaborn as sns                   # Importa a biblioteca seaborn e a apelida de 'sns' para visualizações estatísticas mais atraentes.

# ==============================================================================
# SEÇÃO 2: CONFIGURAÇÕES GLOBAIS
# ==============================================================================
# Esta seção define configurações globais que afetam a aparência dos gráficos gerados.

sns.set_style('whitegrid')                  # Define o estilo dos gráficos seaborn como 'whitegrid' (fundo branco com grades).
plt.rcParams['figure.figsize'] = (10, 6)    # Define o tamanho padrão das figuras matplotlib para 10 polegadas de largura por 6 de altura.
coluna_a_remover = "ID"                     # Define o nome da coluna a ser removida
valor_correlacao_deletar = 0.50             # Define o valor para sinalizar quais atributos do dataset 1 estão mais correlacionados
valor_correlacao_deletar2 = 0.50            # Define o valor para sinalizar quais atributos do dataset 2 estão mais correlacionados

# ==============================================================================
# SEÇÃO 3: DEFINIÇÕES DE ARQUIVOS E NOMES DE DATASETS
# ==============================================================================
# Define os nomes dos arquivos CSV a serem lidos e nomes descritivos para os datasets,
# que serão usados em logs e títulos de gráficos. Ajuste os caminhos dos arquivos se necessário.


# Arquivo 1 (o que estava comentado no código original do usuário)
nome_arquivo1 = 'C:\\Users\\maiqu\\Documents\\Mestrado\\01 - Aprendizado de Maquina\\Trabalho final\\Dataset\\link_r7tjn68rmw-1\\r7tjn68rmw-1\\Dataset_OriginalComClass.csv'                # Define o nome do primeiro arquivo CSV.
# Arquivo 2 (o que NÃO estava comentado no código original do usuário)
nome_arquivo2 = 'C:\\Users\\maiqu\\Documents\\Mestrado\\01 - Aprendizado de Maquina\\Trabalho final\\Dataset\\link_r7tjn68rmw-1\\r7tjn68rmw-1\\Dataset_OriginalSinteticosComClass.csv'      # Define o nome do segundo arquivo CSV.

# Nomes descritivos para os datasets para usar nos títulos
nome_dataset1 = 'Dataset 1: Original'                   # Define um nome descritivo para o primeiro dataset.
nome_dataset2 = 'Dataset 2: Original + Sinteticos'      # Define um nome descritivo para o segundo dataset.


# ==============================================================================
# SEÇÃO 4: DEFINIÇÃO DA FUNÇÃO DE CARREGAMENTO E LIMPEZA DE DADOS
# ==============================================================================
# Esta seção contém a definição da função `carregar_e_limpar_dados`.

def carregar_e_limpar_dados(nome_arquivo, nome_dataset_str):
    """
    Carrega um arquivo CSV, realiza limpeza básica de dados e imprime informações iniciais.

    A limpeza inclui remoção de colunas e linhas totalmente nulas, remoção da coluna ID, e remoção de espaços
    em branco extras dos nomes das colunas. Informações como dimensões, primeiras linhas,
    tipos de dados, estatísticas descritivas e contagem de valores ausentes são impressas.

    Args:
        nome_arquivo (str): O caminho para o arquivo CSV a ser carregado.
        nome_dataset_str (str): Um nome descritivo para o dataset, usado em mensagens de log.

    Returns:
        pandas.DataFrame or None: O DataFrame processado se o carregamento e a limpeza
                                  forem bem-sucedidos e o DataFrame não estiver vazio,
                                  caso contrário, None.
    """
    print(
        f"\n--- CARREGANDO E PROCESSANDO: {nome_dataset_str} ({nome_arquivo}) ---")     # Imprime uma mensagem indicando o início do carregamento e processamento do dataset.
    try:                                                                                # Inicia um bloco try-except para tratamento de erros durante o carregamento do arquivo.
        df_original = pd.read_csv(nome_arquivo, sep=';',decimal=',')                    # Tenta ler o arquivo CSV usando ';' como separador e ',' como decimal.
        print(
            f"Dataset '{nome_dataset_str}' carregado com sucesso!")                     # Imprime uma mensagem de sucesso se o arquivo for carregado.
    except FileNotFoundError:                                                           # Captura o erro se o arquivo não for encontrado.
        print(
            f"Erro: O arquivo '{nome_arquivo}' não foi encontrado. Verifique o caminho.")   # Imprime uma mensagem de erro específica para arquivo não encontrado.
        return None                                                                         # Retorna None se o arquivo não for encontrado.
    except Exception as e:                                                                  # Captura qualquer outra exceção durante o carregamento.
        print(f"Ocorreu um erro ao carregar o CSV '{nome_arquivo}': {e}")                   # Imprime uma mensagem de erro genérica.
        return None                                                                         # Retorna None se ocorrer um erro no carregamento.

    print(
        f"\nDimensões originais do dataset '{nome_dataset_str}': {df_original.shape[0]} linhas, {df_original.shape[1]} colunas.")  # Imprime as dimensões originais (linhas, colunas) do DataFrame.

    # Subseção: Remoção de colunas totalmente nulas
    print(f"\nRemoção das colunas nulas e coluna(s) {coluna_a_remover}:")  # Imprime as colunas removidas.
    colunas_antes_remocao = df_original.columns.tolist()                # Obtém a lista de nomes das colunas antes da remoção.
    df = df_original.dropna(axis='columns', how='all') .copy()          # Remove colunas onde TODOS os valores são NaN (nulos).

    # Verifica se a coluna coluna_a_remover ("ID") existe no DataFrame 'df' antes de tentar removê-la
    if coluna_a_remover in df.columns:
        df = df.drop(columns=[coluna_a_remover])                                                # Remove a coluna "ID". O argumento inplace=True modifica o DataFrame 'df' diretamente.
    else:
        print(f"A coluna '{coluna_a_remover}' não foi encontrada no DataFrame para remoção.")   # Informa que a coluna "ID" não foi encontrada

    colunas_depois_remocao = df.columns.tolist()                    # Obtém a lista de nomes das colunas após a remoção.
    colunas_removidas = [col for col in colunas_antes_remocao if
                         col not in colunas_depois_remocao]         # Identifica as colunas que foram removidas.
    if colunas_removidas:                                           # Verifica se alguma coluna foi removida.
        print(
            f"Colunas que foram removidas: {colunas_removidas}")    # Imprime as colunas removidas.
    else:                                                           # Caso nenhuma coluna tenha sido removida.
        print(
            "Nenhuma coluna continha apenas valores nulos.")        # Informa que nenhuma coluna totalmente nula foi encontrada.
    print(
        f"Dimensões após remover colunas: {df.shape[0]} linhas, {df.shape[1]} colunas.")  # Imprime as dimensões do DataFrame após a remoção de colunas nulas.

    # Subseção: Remoção de linhas totalmente nulas
    num_linhas_totalmente_nulas_antes_remocao = df[df.isnull().all(axis=1)].shape[0]    # Conta o número de linhas onde todos os valores são NaN.
    if num_linhas_totalmente_nulas_antes_remocao > 0:                                   # Verifica se existem linhas totalmente nulas.
        print(f"Número de linhas completamente nulas encontradas: {num_linhas_totalmente_nulas_antes_remocao}")     # Informa o número de linhas totalmente nulas encontradas.
        df = df.dropna(how='all', axis=0)                                                                           # Remove linhas onde TODOS os valores são NaN.
        print("Linhas completamente nulas foram REMOVIDAS.")                                                        # Confirma a remoção das linhas.
        print(f"Dimensões após remover linhas totalmente nulas: {df.shape[0]} linhas, {df.shape[1]} colunas.")      # Imprime as dimensões após a remoção de linhas nulas.
    else:                                                                                                           # Caso nenhuma linha totalmente nula seja encontrada.
        print(  "Nenhuma linha completamente nula encontrada para remover.")                                        # Informa que não foram encontradas linhas totalmente nulas.

    # Subseção: Verificação de DataFrame vazio
    if df.empty:                                            # Verifica se o DataFrame está vazio após as remoções.
        print(f"O DataFrame '{nome_dataset_str}' está vazio após a limpeza inicial. Não é possível prosseguir com a análise para este dataset.")  # Informa que o DataFrame está vazio.
        return None                                         # Retorna None se o DataFrame estiver vazio.

    # Subseção: Limpeza dos nomes das colunas
    df.columns = df.columns.str.strip()                     # Remove espaços em branco extras do início e fim dos nomes das colunas.
    print("\nNomes das colunas após limpeza de espaços:")   # Imprime um título para os nomes das colunas limpos.
    print(df.columns)                                       # Imprime os nomes das colunas após a limpeza.

    # Subseção: Visualização inicial e informações do DataFrame
    print(f"\nPrimeiras 5 linhas do dataset '{nome_dataset_str}' (após tratamentos de nulos):")  # Imprime um título para a visualização das primeiras linhas.
    print(df.head())  # Exibe as primeiras 5 linhas do DataFrame.

    print(f"\nInformações gerais do dataset '{nome_dataset_str}' (após tratamentos de nulos):")  # Imprime um título para as informações gerais.
    df.info()  # Exibe informações gerais sobre o DataFrame (tipos de dados, valores não nulos, uso de memória).

    # Subseção: Estatísticas descritivas
    print(f"\nEstatísticas Descritivas para '{nome_dataset_str}' (colunas numéricas):")     # Imprime um título para as estatísticas descritivas.
    colunas_numericas_para_describe = df.select_dtypes(include=np.number).columns           # Seleciona apenas as colunas numéricas.
    if not colunas_numericas_para_describe.empty:                                           # Verifica se existem colunas numéricas.
        print(df[colunas_numericas_para_describe].describe())                               # Calcula e exibe estatísticas descritivas (média, desvio padrão, min, max, quartis) para colunas numéricas.
    else:                                                                                   # Caso não haja colunas numéricas.
        print("Nenhuma coluna numérica para descrever.")                                    # Informa que não há colunas numéricas.

    # Subseção: Análise de valores ausentes
    print(
        f"\nContagem de valores ausentes por coluna para '{nome_dataset_str}' (após tratamentos de nulos):")  # Imprime um título para a contagem de nulos.
    print(df.isnull().sum())  # Conta e exibe o número de valores ausentes (NaN) por coluna.
    print(
        f"\nPorcentagem de valores ausentes por coluna para '{nome_dataset_str}' (após tratamentos de nulos):")  # Imprime um título para a porcentagem de nulos.
    if len(df) > 0:  # Verifica se o DataFrame não está vazio para evitar divisão por zero.
        print((df.isnull().sum() / len(df)) * 100)  # Calcula e exibe a porcentagem de valores ausentes por coluna.
    else:  # Caso o DataFrame esteja vazio.
        print("DataFrame está vazio.")  # Informa que o DataFrame está vazio.
    print(
        f"--- FIM DO CARREGAMENTO E PROCESSAMENTO INICIAL: {nome_dataset_str} ---")  # Imprime uma mensagem indicando o fim do processamento para este dataset.
    return df  # Retorna o DataFrame processado.


# ==============================================================================
# SEÇÃO 5: CARREGAMENTO E PROCESSAMENTO DOS DATASETS
# ==============================================================================
# Esta seção chama a função `carregar_e_limpar_dados` para cada um dos arquivos CSV definidos,
# armazenando os DataFrames resultantes em df1 e df2.

df1 = carregar_e_limpar_dados(nome_arquivo1, nome_dataset1)  # Carrega e processa o primeiro dataset.
df2 = carregar_e_limpar_dados(nome_arquivo2, nome_dataset2)  # Carrega e processa o segundo dataset.

# ==============================================================================
# SEÇÃO 6: ANÁLISE COMPARATIVA DA COLUNA "Adequação MILHO"
# ==============================================================================
# Esta seção realiza uma análise comparativa da distribuição de classes na coluna
# "Adequação MILHO" entre os dois datasets. Inclui a impressão das porcentagens
# de cada classe e a geração de um gráfico de barras comparativo.

print("\n\n--- ANÁLISE COMPARATIVA DA COLUNA 'Adequação MILHO' ---")  # Imprime o título da seção de análise.
adequacao_column_name = "Adequação MILHO"  # Define o nome da coluna a ser analisada.
classes_ordem = ['Baixa', 'Média', 'Alta']  # Define a ordem desejada das classes para exibição e no gráfico.
adequacao_data_plot = []  # Inicializa uma lista vazia para armazenar dados para o gráfico de barras.

# Subseção: Análise para Dataset 1
if df1 is not None:  # Verifica se o primeiro DataFrame (df1) foi carregado corretamente.
    if adequacao_column_name in df1.columns:  # Verifica se a coluna 'Adequação MILHO' existe em df1.
        print(
            f"\nDistribuição percentual - {adequacao_column_name} ({nome_dataset1}):")  # Imprime um título para as porcentagens do dataset 1.
        counts1 = df1[adequacao_column_name].value_counts(normalize=True).mul(100).reindex(classes_ordem).fillna(
            0)  # Calcula a porcentagem de cada classe, reordena e preenche classes ausentes com 0.
        print(counts1)  # Imprime as porcentagens calculadas para o dataset 1.

        df_counts1_plot = counts1.rename(
            'Porcentagem').reset_index()  # Converte a série de contagens em um DataFrame e renomeia a coluna de valores para 'Porcentagem'.
        # A coluna do índice (classes) é nomeada com o nome da coluna original ('Adequação MILHO') por padrão no reset_index().
        df_counts1_plot.rename(columns={adequacao_column_name: 'Classe Adequação MILHO'},
                               inplace=True)  # Renomeia a coluna das classes para 'Classe Adequação MILHO'.
        df_counts1_plot['Dataset'] = nome_dataset1  # Adiciona uma coluna 'Dataset' para identificar a origem dos dados.
        adequacao_data_plot.append(df_counts1_plot)  # Adiciona o DataFrame processado à lista para o gráfico combinado.
    else:  # Caso a coluna 'Adequação MILHO' não exista em df1.
        print(
            f"\nColuna '{adequacao_column_name}' não encontrada em {nome_dataset1}.")  # Informa que a coluna não foi encontrada.
else:  # Caso df1 não tenha sido carregado.
    print(
        f"\n{nome_dataset1} não foi carregado. Pulando análise de '{adequacao_column_name}'.")  # Informa que o dataset não foi carregado.

# Subseção: Análise para Dataset 2
if df2 is not None:  # Verifica se o segundo DataFrame (df2) foi carregado corretamente.
    if adequacao_column_name in df2.columns:  # Verifica se a coluna 'Adequação MILHO' existe em df2.
        print(
            f"\nDistribuição percentual - {adequacao_column_name} ({nome_dataset2}):")  # Imprime um título para as porcentagens do dataset 2.
        counts2 = df2[adequacao_column_name].value_counts(normalize=True).mul(100).reindex(classes_ordem).fillna(
            0)  # Calcula a porcentagem de cada classe, reordena e preenche classes ausentes com 0.
        print(counts2)  # Imprime as porcentagens calculadas para o dataset 2.

        df_counts2_plot = counts2.rename(
            'Porcentagem').reset_index()  # Converte a série de contagens em um DataFrame e renomeia a coluna de valores para 'Porcentagem'.
        # A coluna do índice (classes) é nomeada com o nome da coluna original ('Adequação MILHO') por padrão no reset_index().
        df_counts2_plot.rename(columns={adequacao_column_name: 'Classe Adequação MILHO'},
                               inplace=True)  # Renomeia a coluna das classes para 'Classe Adequação MILHO'.
        df_counts2_plot['Dataset'] = nome_dataset2  # Adiciona uma coluna 'Dataset' para identificar a origem dos dados.
        adequacao_data_plot.append(df_counts2_plot)  # Adiciona o DataFrame processado à lista para o gráfico combinado.
    else:  # Caso a coluna 'Adequação MILHO' não exista em df2.
        print(
            f"\nColuna '{adequacao_column_name}' não encontrada em {nome_dataset2}.")  # Informa que a coluna não foi encontrada.
else:  # Caso df2 não tenha sido carregado.
    print(
        f"\n{nome_dataset2} não foi carregado. Pulando análise de '{adequacao_column_name}'.")  # Informa que o dataset não foi carregado.

# Subseção: Geração do gráfico de barras comparativo
if adequacao_data_plot:  # Verifica se há dados para plotar (se a lista não está vazia).
    df_adequacao_combined_plot = pd.concat(
        adequacao_data_plot)  # Concatena os DataFrames de contagem de df1 e df2 em um único DataFrame.

    # As linhas abaixo estão comentadas, mas servem para depurar as colunas do DataFrame combinado, se necessário.
    # print("\nColunas em df_adequacao_combined_plot antes do plot:") # Imprime texto informativo.
    # print(df_adequacao_combined_plot.columns) # Imprime os nomes das colunas do DataFrame combinado.
    # print(df_adequacao_combined_plot.head()) # Imprime as primeiras linhas do DataFrame combinado.

    plt.figure(figsize=(10, 7))  # Cria uma nova figura matplotlib com tamanho especificado para o gráfico.
    sns.barplot(x='Classe Adequação MILHO', y='Porcentagem', hue='Dataset', data=df_adequacao_combined_plot,
                palette=['skyblue', 'lightcoral'], order=classes_ordem)  # Cria um gráfico de barras usando seaborn.
    plt.title(f'Comparativo da Distribuição de Classes - {adequacao_column_name}',
              fontsize=15)  # Define o título do gráfico.
    plt.ylabel('Porcentagem (%)', fontsize=12)  # Define o rótulo do eixo Y.
    plt.xlabel(f'Classe de {adequacao_column_name}', fontsize=12)  # Define o rótulo do eixo X.
    plt.xticks(rotation=0, ha='center',
               fontsize=10)  # Configura os rótulos do eixo X (rotação, alinhamento, tamanho da fonte).
    plt.legend(title='Dataset', fontsize=10)  # Adiciona uma legenda ao gráfico.
    plt.grid(axis='y', linestyle='--')  # Adiciona uma grade horizontal pontilhada ao gráfico.
    plt.tight_layout()  # Ajusta o layout do gráfico para evitar sobreposições.
else:  # Caso não haja dados para o gráfico.
    print(
        f"\nNão foi possível gerar o gráfico comparativo para '{adequacao_column_name}' (sem dados ou coluna não encontrada).")  # Informa que o gráfico não pôde ser gerado.
print("--- FIM DA ANÁLISE DA COLUNA 'Adequação MILHO' ---")  # Imprime o fim da seção de análise.

# ==============================================================================
# SEÇÃO 7: ANÁLISE UNIVARIADA COMPARATIVA E DETECÇÃO DE OUTLIERS
# ==============================================================================
# Esta seção realiza uma análise univariada para cada coluna numérica comum aos dois datasets.
# Para cada coluna, gera histogramas e boxplots comparativos e calcula estatísticas de outliers.

print("\n\n--- INÍCIO DA ANÁLISE UNIVARIADA COMPARATIVA E DETECÇÃO DE OUTLIERS ---")  # Imprime o título da seção.
if df1 is not None and df2 is not None:  # Verifica se ambos os DataFrames foram carregados.
    colunas_atributos1 = df1.columns.drop(
        'ID') if 'ID' in df1.columns else df1.columns  # Seleciona colunas de atributos de df1, excluindo 'ID' se existir.
    colunas_atributos2 = df2.columns.drop(
        'ID') if 'ID' in df2.columns else df2.columns  # Seleciona colunas de atributos de df2, excluindo 'ID' se existir.
    numeric_cols1 = df1[colunas_atributos1].select_dtypes(
        include=np.number).columns  # Identifica colunas numéricas em df1.
    numeric_cols2 = df2[colunas_atributos2].select_dtypes(
        include=np.number).columns  # Identifica colunas numéricas em df2.
    common_numeric_cols = sorted(list(set(numeric_cols1) & set(
        numeric_cols2)))  # Encontra as colunas numéricas comuns a ambos os DataFrames e as ordena.

    if not common_numeric_cols:  # Verifica se não há colunas numéricas comuns.
        print(
            "Nenhuma coluna numérica em comum encontrada entre os dois datasets para análise univariada comparativa.")  # Informa a ausência de colunas comuns.
    else:  # Caso existam colunas numéricas comuns.
        print(
            f"Analisando colunas numéricas comuns: {common_numeric_cols}")  # Lista as colunas comuns que serão analisadas.
        for coluna in common_numeric_cols:  # Itera sobre cada coluna numérica comum.
            print(f"\nAnalisando comparativamente a coluna: {coluna}")  # Informa qual coluna está sendo analisada.
            plt.figure(figsize=(14, 10))  # Cria uma nova figura para os gráficos da coluna atual.

            # Subseção: Análise da Coluna para o Dataset 1 (gráficos superiores)
            ax1 = plt.subplot(2, 2, 1)  # Cria o primeiro subplot (linha 1, coluna 1) para o histograma de df1.
            if df1[coluna].notna().sum() > 0:  # Verifica se a coluna em df1 possui dados não nulos.
                sns.histplot(df1[coluna].dropna(), kde=True, bins=20,
                             ax=ax1)  # Plota o histograma com estimativa de densidade do kernel (KDE).
                ax1.set_title(f'Histograma de {coluna}\n({nome_dataset1})')  # Define o título do histograma.
                ax1.set_xlabel(coluna);
                ax1.set_ylabel('Frequência')  # Define os rótulos dos eixos X e Y.
            else:  # Caso a coluna não tenha dados válidos.
                ax1.text(0.5, 0.5, 'Sem dados válidos', ha='center', va='center',
                         transform=ax1.transAxes)  # Exibe mensagem de "sem dados".
                ax1.set_title(
                    f'Histograma de {coluna}\n({nome_dataset1}) - Sem dados')  # Define título indicando ausência de dados.

            ax2 = plt.subplot(2, 2, 2)  # Cria o segundo subplot (linha 1, coluna 2) para o boxplot de df1.
            if df1[coluna].notna().sum() > 0:  # Verifica se a coluna em df1 possui dados não nulos.
                sns.boxplot(x=df1[coluna].dropna(), ax=ax2)  # Plota o boxplot.
                ax2.set_title(f'Boxplot de {coluna}\n({nome_dataset1})')  # Define o título do boxplot.
                ax2.set_xlabel(coluna)  # Define o rótulo do eixo X.
                # Cálculo de outliers para df1
                Q1_df1, Q3_df1 = df1[coluna].quantile(0.25), df1[coluna].quantile(
                    0.75)  # Calcula o primeiro (Q1) e terceiro (Q3) quartis.
                IQR_df1 = Q3_df1 - Q1_df1  # Calcula o Intervalo Interquartil (IQR).
                lim_inf_df1, lim_sup_df1 = Q1_df1 - 1.5 * IQR_df1, Q3_df1 + 1.5 * IQR_df1  # Define os limites inferior e superior para detecção de outliers.
                out_df1 = df1[(df1[coluna] < lim_inf_df1) | (df1[coluna] > lim_sup_df1)]  # Identifica os outliers.
                num_out_df1, non_nan_df1 = len(out_df1), len(
                    df1[coluna].dropna())  # Conta o número de outliers e o número de valores não nulos.
                perc_out_df1 = (
                                           num_out_df1 / non_nan_df1) * 100 if non_nan_df1 > 0 else 0  # Calcula a porcentagem de outliers.
                print(
                    f"  Outliers '{coluna}' ({nome_dataset1}): Q1={Q1_df1:.2f}, Q3={Q3_df1:.2f}, IQR={IQR_df1:.2f}, LimInf={lim_inf_df1:.2f}, LimSup={lim_sup_df1:.2f}, N={num_out_df1} ({perc_out_df1:.2f}%)")  # Imprime as estatísticas de outliers.
            else:  # Caso a coluna não tenha dados válidos.
                ax2.text(0.5, 0.5, 'Sem dados válidos', ha='center', va='center',
                         transform=ax2.transAxes)  # Exibe mensagem de "sem dados".
                ax2.set_title(
                    f'Boxplot de {coluna}\n({nome_dataset1}) - Sem dados')  # Define título indicando ausência de dados.
                print(
                    f"  Outliers '{coluna}' ({nome_dataset1}): Coluna sem dados válidos.")  # Informa ausência de dados para cálculo de outliers.

            # Subseção: Análise da Coluna para o Dataset 2 (gráficos inferiores)
            ax3 = plt.subplot(2, 2, 3)  # Cria o terceiro subplot (linha 2, coluna 1) para o histograma de df2.
            if df2[coluna].notna().sum() > 0:  # Verifica se a coluna em df2 possui dados não nulos.
                sns.histplot(df2[coluna].dropna(), kde=True, bins=20, ax=ax3)  # Plota o histograma com KDE.
                ax3.set_title(f'Histograma de {coluna}\n({nome_dataset2})')  # Define o título do histograma.
                ax3.set_xlabel(coluna);
                ax3.set_ylabel('Frequência')  # Define os rótulos dos eixos X e Y.
            else:  # Caso a coluna não tenha dados válidos.
                ax3.text(0.5, 0.5, 'Sem dados válidos', ha='center', va='center',
                         transform=ax3.transAxes)  # Exibe mensagem de "sem dados".
                ax3.set_title(
                    f'Histograma de {coluna}\n({nome_dataset2}) - Sem dados')  # Define título indicando ausência de dados.

            ax4 = plt.subplot(2, 2, 4)  # Cria o quarto subplot (linha 2, coluna 2) para o boxplot de df2.
            if df2[coluna].notna().sum() > 0:  # Verifica se a coluna em df2 possui dados não nulos.
                sns.boxplot(x=df2[coluna].dropna(), ax=ax4)  # Plota o boxplot.
                ax4.set_title(f'Boxplot de {coluna}\n({nome_dataset2})')  # Define o título do boxplot.
                ax4.set_xlabel(coluna)  # Define o rótulo do eixo X.
                # Cálculo de outliers para df2
                Q1_df2, Q3_df2 = df2[coluna].quantile(0.25), df2[coluna].quantile(0.75)  # Calcula Q1 e Q3.
                IQR_df2 = Q3_df2 - Q1_df2  # Calcula IQR.
                lim_inf_df2, lim_sup_df2 = Q1_df2 - 1.5 * IQR_df2, Q3_df2 + 1.5 * IQR_df2  # Define limites para outliers.
                out_df2 = df2[(df2[coluna] < lim_inf_df2) | (df2[coluna] > lim_sup_df2)]  # Identifica outliers.
                num_out_df2, non_nan_df2 = len(out_df2), len(
                    df2[coluna].dropna())  # Conta outliers e valores não nulos.
                perc_out_df2 = (
                                           num_out_df2 / non_nan_df2) * 100 if non_nan_df2 > 0 else 0  # Calcula porcentagem de outliers.
                print(
                    f"  Outliers '{coluna}' ({nome_dataset2}): Q1={Q1_df2:.2f}, Q3={Q3_df2:.2f}, IQR={IQR_df2:.2f}, LimInf={lim_inf_df2:.2f}, LimSup={lim_sup_df2:.2f}, N={num_out_df2} ({perc_out_df2:.2f}%)")  # Imprime estatísticas de outliers.
            else:  # Caso a coluna não tenha dados válidos.
                ax4.text(0.5, 0.5, 'Sem dados válidos', ha='center', va='center',
                         transform=ax4.transAxes)  # Exibe mensagem de "sem dados".
                ax4.set_title(
                    f'Boxplot de {coluna}\n({nome_dataset2}) - Sem dados')  # Define título indicando ausência de dados.
                print(
                    f"  Outliers '{coluna}' ({nome_dataset2}): Coluna sem dados válidos.")  # Informa ausência de dados para cálculo de outliers.

            plt.suptitle(f'Análise Univariada Comparativa: {coluna}',
                         fontsize=16)  # Define um título principal para a figura (agrupando os 4 subplots).
            plt.tight_layout(
                rect=[0, 0.03, 1, 0.95])  # Ajusta o layout para evitar sobreposição e dar espaço ao suptitle.
elif df1 is None or df2 is None:  # Caso um ou ambos os DataFrames não tenham sido carregados.
    print(
        "Um ou ambos os DataFrames não puderam ser carregados. Análise univariada comparativa não pode prosseguir completamente.")  # Informa que a análise não pode prosseguir.
print("--- FIM DA ANÁLISE UNIVARIADA COMPARATIVA E DETECÇÃO DE OUTLIERS ---\n")  # Imprime o fim da seção.

# ==============================================================================
# SEÇÃO 8: ANÁLISE BIVARIADA COMPARATIVA (MATRIZ DE CORRELAÇÃO)
# ==============================================================================
# Esta seção calcula e visualiza as matrizes de correlação para as colunas numéricas
# de cada dataset, apresentando-as como heatmaps comparativos.

print("\n--- ANÁLISE BIVARIADA COMPARATIVA (MATRIZ DE CORRELAÇÃO) ---")  # Imprime o título da seção.
num_heatmaps = 0  # Inicializa um contador para o número de heatmaps a serem gerados.
if df1 is not None and not df1.select_dtypes(
    include=np.number).empty: num_heatmaps += 1  # Incrementa se df1 existe e tem colunas numéricas.
if df2 is not None and not df2.select_dtypes(
    include=np.number).empty: num_heatmaps += 1  # Incrementa se df2 existe e tem colunas numéricas.

if num_heatmaps > 0:  # Verifica se há pelo menos um dataset com dados numéricos para gerar heatmap.
    plt.figure(figsize=(20,
                        18 * num_heatmaps if num_heatmaps > 0 else 18))  # Cria uma nova figura, ajustando a altura com base no número de heatmaps.
    plot_index = 1  # Inicializa um índice para os subplots (caso haja mais de um heatmap).

    # Subseção: Heatmap para Dataset 1
    if df1 is not None:  # Verifica se df1 foi carregado.
        df1_numerico = df1.select_dtypes(include=np.number)  # Seleciona apenas colunas numéricas de df1.
        if not df1_numerico.empty:  # Verifica se existem colunas numéricas.
            print(f"\nMatriz de Correlação ({nome_dataset1}):")  # Imprime título para a matriz de correlação.
            correlation_matrix1 = df1_numerico.corr()  # Calcula a matriz de correlação.
            print(correlation_matrix1)  # Imprime a matriz de correlação.
            ax_hm1 = plt.subplot(num_heatmaps, 1, plot_index)  # Cria um subplot para o heatmap de df1.
            sns.heatmap(correlation_matrix1, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                        annot_kws={"size": 8}, ax=ax_hm1 )  # Gera o heatmap.
            ax_hm1.set_title(f'Heatmap da Matriz de Correlação\n({nome_dataset1})',
                             fontsize=14)  # Define o título do heatmap.
            plt.setp(ax_hm1.get_xticklabels(), rotation=45, ha='right');
            plt.setp(ax_hm1.get_yticklabels(), rotation=0)  # Configura rótulos dos eixos X e Y.

            # Sinalizar Valores Com Alta Correlação
            for text_annotation in ax_hm1.texts:
                try:
                    # Obtém o valor numérico da anotação de texto
                    value = float(text_annotation.get_text())

                    # Verifica se o valor está nos intervalos desejados
                    if (value >= valor_correlacao_deletar) or (value <= -valor_correlacao_deletar):
                        # Sinaliza o valor adicionando uma caixa delimitadora
                        text_annotation.set_bbox(
                            dict(facecolor='none', edgecolor='black', linewidth=1.5, boxstyle='round,pad=0.3'))
                except ValueError:
                    # Ignora anotações que não podem ser convertidas para float (raro com fmt=".2f")
                    continue

            plot_index += 1  # Incrementa o índice do subplot.
        else:
            print(
                f"Nenhuma coluna numérica em {nome_dataset1} para matriz de correlação.")  # Informa ausência de colunas numéricas.
    else:
        print(f"{nome_dataset1} não carregado. Pulando heatmap.")  # Informa que df1 não foi carregado.

    # Subseção: Heatmap para Dataset 2
    if df2 is not None:  # Verifica se df2 foi carregado.
        df2_numerico = df2.select_dtypes(include=np.number)  # Seleciona apenas colunas numéricas de df2.
        if not df2_numerico.empty:  # Verifica se existem colunas numéricas.
            print(f"\nMatriz de Correlação ({nome_dataset2}):")  # Imprime título para a matriz de correlação.
            correlation_matrix2 = df2_numerico.corr()  # Calcula a matriz de correlação.
            print(correlation_matrix2)  # Imprime a matriz de correlação.
            ax_hm2 = plt.subplot(num_heatmaps, 1, plot_index)  # Cria um subplot para o heatmap de df2.
            sns.heatmap(correlation_matrix2, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5,
                        annot_kws={"size": 8}, ax=ax_hm2 )  # Gera o heatmap.
            ax_hm2.set_title(f'Heatmap da Matriz de Correlação\n({nome_dataset2})',
                             fontsize=14)  # Define o título do heatmap.
            plt.setp(ax_hm2.get_xticklabels(), rotation=45, ha='right');
            plt.setp(ax_hm2.get_yticklabels(), rotation=0)  # Configura rótulos dos eixos X e Y.

            # Sinalizar Valores Com Alta Correlação
            for text_annotation in ax_hm2.texts:
                try:
                    # Obtém o valor numérico da anotação de texto
                    value = float(text_annotation.get_text())

                    # Verifica se o valor está nos intervalos desejados
                    if (value >= valor_correlacao_deletar2) or (value <= -valor_correlacao_deletar2):
                        # Sinaliza o valor adicionando uma caixa delimitadora
                        text_annotation.set_bbox(
                            dict(facecolor='none', edgecolor='black', linewidth=1.5, boxstyle='round,pad=0.3'))
                except ValueError:
                    # Ignora anotações que não podem ser convertidas para float (raro com fmt=".2f")
                    continue

            plot_index += 1  # Incrementa o índice do subplot.
        else:
            print(
                f"Nenhuma coluna numérica em {nome_dataset2} para matriz de correlação.")  # Informa ausência de colunas numéricas.
    else:
        print(f"{nome_dataset2} não carregado. Pulando heatmap.")  # Informa que df2 não foi carregado.

    if plot_index > 1:  # Verifica se pelo menos um heatmap foi efetivamente adicionado ao plot.
        plt.tight_layout(
            pad=25.0)  # Ajusta o layout da figura para evitar sobreposições, com um preenchimento (padding).
else:  # Caso nenhum dataset possua colunas numéricas.
    print(
        "Nenhum dos DataFrames possui colunas numéricas para gerar heatmaps de correlação.")  # Informa que não é possível gerar heatmaps.
print("\n--- FIM DA ANÁLISE EXPLORATÓRIA COMPARATIVA ---")  # Imprime o fim da seção de análise exploratória.

# ==============================================================================
# SEÇÃO 9: EXIBIÇÃO DOS GRÁFICOS
# ==============================================================================
# Esta seção verifica se alguma figura foi gerada e, em caso afirmativo, exibe todas
# as figuras de uma vez.

if plt.get_fignums():  # Verifica se existem números de figuras ativos (ou seja, se figuras foram criadas).
    print("\nExibindo todos os gráficos gerados...")  # Informa que os gráficos serão exibidos.
    plt.show()  # Exibe todas as figuras matplotlib geradas durante a execução do script. Bloqueia a execução até que as janelas dos gráficos sejam fechadas.
else:  # Caso nenhuma figura tenha sido criada.
    print("\nNenhum gráfico foi gerado para exibição.")  # Informa que não há gráficos para mostrar.

# ==============================================================================
# SEÇÃO 10: FINALIZAÇÃO DO SCRIPT
# ==============================================================================
# Imprime uma mensagem indicando que o script foi finalizado.
print("\nScript finalizado.")  # Imprime uma mensagem final no console.