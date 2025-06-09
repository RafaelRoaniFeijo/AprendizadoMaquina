# ==============================================================================
# SEÇÃO 0: IMPORTAÇÃO DE BIBLIOTECAS
# ==============================================================================
# Esta seção importa todas as bibliotecas Python necessárias para a execução do script.
# - pandas: Para manipulação e análise de dados tabulares (DataFrames).
# - numpy: Para operações numéricas eficientes, especialmente com arrays.
# - matplotlib.pyplot: Para criar gráficos estáticos, animados e interativos.
# - seaborn: Baseada no matplotlib, fornece uma interface de alto nível para desenhar gráficos estatísticos atraentes.
# - sklearn.preprocessing.MinMaxScaler: Para normalizar features escalando-as para um intervalo específico (geralmente [0, 1]).
# - os: Fornece uma maneira de usar funcionalidades dependentes do sistema operacional, como manipulação de caminhos de arquivo e diretórios.
# - tkinter: Para criar interfaces gráficas de usuário (GUI), usado aqui para exibir o DataFrame em uma janela.

import pandas as pd  # Importa a biblioteca pandas e a apelida de 'pd'.
import numpy as np  # Importa a biblioteca numpy e a apelida de 'np'.
import matplotlib.pyplot as plt  # Importa o submódulo pyplot da matplotlib e o apelida de 'plt'.
import seaborn as sns  # Importa a biblioteca seaborn e a apelida de 'sns'.
from sklearn.preprocessing import \
    MinMaxScaler  # Importa a classe MinMaxScaler do módulo de pré-processamento do scikit-learn.
import os  # Importa o módulo 'os' para interagir com o sistema operacional, como criar pastas.
import tkinter as tk  # Importa a biblioteca Tkinter para GUI
from tkinter import ttk  # Importa o themed Tkinter (melhor aparência dos widgets)

# ==============================================================================
# SEÇÃO 1: DEFINIÇÕES INICIAIS E CONFIGURAÇÕES
# ==============================================================================
# Nesta seção, definimos variáveis globais como caminhos de arquivo, nomes de colunas
# importantes, sufixos para nomes de arquivos de saída.

# --- 1. DEFINIÇÕES INICIAIS ---
# Caminho para o arquivo CSV (ajuste se necessário)
caminho_do_arquivo_leitura = 'C:\\Users\\maiqu\\Documents\\Mestrado\\01 - Aprendizado de Maquina\\Trabalho final\\Dataset\\link_r7tjn68rmw-1\\r7tjn68rmw-1\\'  # Define o diretório base onde o arquivo de leitura está localizado.
# nome_do_arquivo_leitura = 'Dataset_OriginalComClass.csv'        # Define o nome do arquivo CSV a ser lido.
nome_do_arquivo_leitura = 'Dataset_SinteticosComClass.csv'        # Define o nome do arquivo CSV a ser lido.
# nome_do_arquivo_leitura = 'Dataset_OriginalSinteticosComClass.csv'  # Define o nome do arquivo CSV a ser lido.
caminho_arquivo = caminho_do_arquivo_leitura + nome_do_arquivo_leitura  # Concatena o diretório e o nome do arquivo para formar o caminho completo.

coluna_alvo = "Adequação MILHO"  # Define o nome da coluna que será o alvo da análise/modelo (variável dependente).
coluna_id = "ID"  # Define o nome da coluna que serve como identificador único para as linhas.
# Nomes exatos das colunas de textura (confirmados como corretos pelo usuário)
col_areia = 'Sand %'  # Define o nome da coluna que representa a porcentagem de areia.
col_argila = 'Clay %'  # Define o nome da coluna que representa a porcentagem de argila.
col_silte = 'Silt %'  # Define o nome da coluna que representa a porcentagem de silte.
col_ph_nome = 'pH'  # Define o nome da coluna que representa o pH do solo.

# Definições para salvamento
sufixo_preprocessado_completo = "_PREPROCESSADO_COMPLETO"  # Sufixo para o nome do arquivo CSV completo após o pré-processamento.
nome_pasta_salvar_csv_FOLD = 'Saida_CSV\\'  # Nome da subpasta onde os arquivos CSV gerados serão salvos. # Mantido para salvar o dataset completo
caminho_arquivos_saida_CSV = caminho_do_arquivo_leitura + nome_pasta_salvar_csv_FOLD  # Define o caminho completo para a pasta de saída dos CSVs.

print(
    f"--- Iniciando pré-processamento do arquivo: {caminho_arquivo} ---")  # Imprime uma mensagem indicando o início do script e o arquivo que será processado.

# ==============================================================================
# SEÇÃO 2: CARREGAMENTO DO DATASET
# ==============================================================================
# Esta seção tenta carregar o arquivo CSV especificado em um DataFrame do pandas.
# Também lida com possíveis erros durante o carregamento, como arquivo não encontrado.

# --- 2. CARREGAMENTO DO DATASET ---
try:  # Inicia um bloco de tratamento de exceções para o carregamento do arquivo.
    df = pd.read_csv(caminho_arquivo, sep=';',
                     decimal=',')  # Tenta ler o arquivo CSV, especificando ';' como separador de colunas e ',' como separador decimal.
    print("\n✅ Dataset carregado com sucesso!")  # Imprime uma mensagem de sucesso se o carregamento for bem-sucedido.
    print(
        f"Dimensões originais: {df.shape[0]} linhas, {df.shape[1]} colunas.")  # Imprime as dimensões (número de linhas e colunas) do DataFrame original.
except FileNotFoundError:  # Captura o erro específico se o arquivo não for encontrado no caminho especificado.
    print(
        f"❌ Erro: O arquivo '{caminho_arquivo}' não foi encontrado. Verifique o caminho.")  # Imprime uma mensagem de erro.
    exit()  # Encerra o script se o arquivo não for encontrado.
except Exception as e:  # Captura qualquer outra exceção que possa ocorrer durante o carregamento.
    print(f"❌ Ocorreu um erro ao carregar o CSV: {e}")  # Imprime a mensagem de erro da exceção.
    exit()  # Encerra o script se ocorrer um erro.

# Remover espaços extras dos nomes das colunas (boa prática)
df.columns = df.columns.str.strip()  # Remove espaços em branco do início e do fim de cada nome de coluna.

# ==============================================================================
# SEÇÃO 3: REMOÇÃO DE LINHAS E COLUNAS TOTALMENTE EM BRANCO
# ==============================================================================
# Esta seção remove colunas e linhas que contêm apenas valores ausentes (NaN),
# o que ajuda a limpar o dataset de entradas completamente vazias.

# --- 3. REMOÇÃO DE LINHAS E COLUNAS TOTALMENTE EM BRANCO ---
print("\n--- Removendo linhas e colunas totalmente em branco ---")  # Imprime um título para esta seção.
colunas_originais = df.columns.tolist()  # Armazena a lista de nomes de colunas originais antes da remoção.
linhas_originais = df.shape[0]  # Armazena o número original de linhas antes da remoção.

df_sem_colunas_vazias = df.dropna(axis='columns',
                                  how='all')  # Remove colunas onde todos os seus valores são NaN. 'axis='columns'' especifica que a operação é por coluna, 'how='all'' especifica que só remove se todos os valores forem NaN.
colunas_depois_remocao_col = df_sem_colunas_vazias.columns.tolist()  # Obtém a lista de colunas após a remoção das colunas vazias.
colunas_removidas_vazias = [col for col in colunas_originais if
                            col not in colunas_depois_remocao_col]  # Identifica quais colunas foram efetivamente removidas.

if colunas_removidas_vazias:  # Verifica se alguma coluna foi removida.
    print(f"Colunas totalmente em branco removidas: {colunas_removidas_vazias}")  # Imprime a lista de colunas removidas.
else:  # Caso nenhuma coluna tenha sido removida.
    print(
        "Nenhuma coluna totalmente em branco foi encontrada para remover.")  # Informa que nenhuma coluna totalmente vazia foi encontrada.
print(
    f"Dimensões após remover colunas vazias: {df_sem_colunas_vazias.shape[0]} linhas, {df_sem_colunas_vazias.shape[1]} colunas.")  # Imprime as dimensões do DataFrame após esta etapa.

df_limpo = df_sem_colunas_vazias.dropna(axis='rows',
                                        how='all')  # Remove linhas onde todos os seus valores (nas colunas restantes) são NaN. 'axis='rows'' especifica a operação por linha.
linhas_depois_remocao_row = df_limpo.shape[0]  # Obtém o número de linhas após a remoção de linhas vazias.

if linhas_originais > linhas_depois_remocao_row:  # Compara o número de linhas antes e depois de todas as limpezas de nulos.
    print(
        f"Linhas que se tornaram totalmente em branco (ou já eram) foram removidas.")  # Informa sobre a remoção de linhas.
else:  # Caso nenhuma linha tenha sido removida nesta etapa (ou na combinação das etapas).
    print(
        "Nenhuma linha totalmente em branco foi encontrada para remover (após a possível remoção de colunas).")  # Informa que nenhuma linha totalmente vazia foi encontrada.
print(
    f"Dimensões após limpeza de nulos: {df_limpo.shape[0]} linhas, {df_limpo.shape[1]} colunas.")  # Imprime as dimensões finais do DataFrame após a limpeza.
print("✅ Remoção de linhas e colunas vazias concluída.")  # Sinaliza a conclusão desta etapa.

# ==============================================================================
# SEÇÃO 4: IDENTIFICAÇÃO DE OUTLIERS (TEXTURA E PH)
# ==============================================================================
# Esta seção verifica outliers com base em regras de domínio:
# 1. Soma das frações texturais do solo (Areia, Argila, Silte): identifica se a soma está fora do intervalo esperado de 99-101%.
# 2. pH: identifica se os valores de pH estão fora da faixa fisicamente plausível (0-14).

# --- 4. IDENTIFICAÇÃO DE OUTLIERS (TEXTURA E PH) ---
print(f"\n--- Identificando Outliers (Soma das Frações Texturais e pH) ---")  # Imprime o título principal da seção.

coluna_id_existe_em_df_limpo = True if coluna_id and isinstance(coluna_id,
                                                                str) and coluna_id in df_limpo.columns else False  # Verifica se a coluna ID (definida em 'coluna_id') existe no DataFrame 'df_limpo'.

# ------------------------------------------------------------------------------
# Subseção 4.1: Outliers na Soma das Frações Texturais
# ------------------------------------------------------------------------------
# Verifica se a soma das porcentagens de Areia, Argila e Silte está entre 99% e 101%.
# Valores fora dessa faixa são considerados outliers para a consistência da textura.

print(
    f"\n--- Verificando outliers na soma das frações texturais ('{col_areia}', '{col_argila}', '{col_silte}') ---")  # Título da subseção.
print("    (Outlier identificado se a soma < 99% ou > 101%)")  # Explica o critério de outlier para a soma.
cols_textura = [col_areia, col_argila, col_silte]  # Cria uma lista com os nomes das colunas de textura.
colunas_textura_presentes_no_df = all(col in df_limpo.columns for col in
                                      cols_textura)  # Verifica se todas as colunas de textura definidas existem no DataFrame 'df_limpo'.

if colunas_textura_presentes_no_df:  # Prossegue apenas se todas as colunas de textura existirem.
    for col_textura_check in cols_textura:  # Itera sobre cada coluna de textura.
        df_limpo[col_textura_check] = pd.to_numeric(df_limpo[col_textura_check],
                                                    errors='coerce')  # Converte a coluna para tipo numérico; valores não convertíveis se tornam NaN.

    df_limpo['Soma_Textura_Temp'] = df_limpo[col_areia] + df_limpo[col_argila] + df_limpo[
        col_silte]  # Calcula a soma das três frações texturais para cada linha e armazena em uma coluna temporária.
    linhas_soma_nan_textura = df_limpo[df_limpo[
        'Soma_Textura_Temp'].isna()]  # Identifica linhas onde a soma resultou em NaN (devido a NaN em um dos componentes).

    condicao_outlier_soma = (df_limpo['Soma_Textura_Temp'] < 99.0) | (df_limpo[
                                                                          'Soma_Textura_Temp'] > 101.0)  # Define a condição lógica para a soma ser considerada um outlier (menor que 99 OU maior que 101).
    outliers_soma_textura = df_limpo[df_limpo[
                                         'Soma_Textura_Temp'].notna() & condicao_outlier_soma]  # Filtra as linhas que atendem à condição de outlier e cuja soma não é NaN.

    num_total_linhas_df_limpo = len(
        df_limpo)  # Obtém o número total de linhas no DataFrame 'df_limpo' para cálculo de percentual.
    num_outliers_soma = len(outliers_soma_textura)  # Conta o número de linhas identificadas como outliers na soma.

    if num_outliers_soma > 0:  # Verifica se foram encontrados outliers na soma.
        print(
            f"Encontradas {num_outliers_soma} linha(s) com outliers na soma das frações texturais (soma fora de [99%, 101%]):")  # Informa o número de outliers encontrados.
        if coluna_id_existe_em_df_limpo:  # Se a coluna ID existe, usa-a para identificar as linhas.
            for index, row in outliers_soma_textura.iterrows():  # Itera sobre as linhas com outliers.
                if coluna_id in row: # Checa se coluna_id ainda existe na linha (pode ter sido removida por outros processos)
                    print(
                        f"  ID: {row[coluna_id]}, Soma Textura: {row['Soma_Textura_Temp']:.2f}%")  # Imprime o ID e o valor da soma.
                else:
                    print(
                        f"  Índice: {index}, Soma Textura: {row['Soma_Textura_Temp']:.2f}% (Coluna ID não encontrada na linha)")
        else:  # Caso a coluna ID não exista ou não tenha sido encontrada.
            print(
                "  (Coluna ID não disponível ou não encontrada no df_limpo, mostrando índices das linhas):")  # Informa que os índices serão mostrados.
            for index, row in outliers_soma_textura.iterrows():  # Itera sobre as linhas com outliers.
                print(
                    f"  Índice: {index}, Soma Textura: {row['Soma_Textura_Temp']:.2f}%")  # Imprime o índice da linha e o valor da soma.
    else:  # Caso nenhum outlier na soma tenha sido encontrado.
        print(
            "Nenhum outlier na soma das frações texturais encontrado (todas as somas válidas estão entre 99% e 101% inclusive).")  # Informa que não há outliers.

    if not linhas_soma_nan_textura.empty:  # Verifica se houve linhas onde a soma não pôde ser calculada.
        print(
            f"Adicionalmente, {len(linhas_soma_nan_textura)} linha(s) não puderam ter a soma de textura calculada (resultado NaN) devido a valores não numéricos/ausentes.")  # Informa sobre linhas com soma NaN.
        if coluna_id_existe_em_df_limpo and coluna_id in linhas_soma_nan_textura.columns:  # Se a coluna ID existe.
            print(
                f"  IDs dessas linhas com soma NaN na textura: {linhas_soma_nan_textura[coluna_id].tolist()}")  # Lista os IDs das linhas com soma NaN.
        else:  # Caso a coluna ID não exista.
            print(
                f"  Índices dessas linhas com soma NaN na textura: {linhas_soma_nan_textura.index.tolist()}")  # Lista os índices das linhas com soma NaN.

    if num_total_linhas_df_limpo > 0:  # Evita divisão por zero se o DataFrame estiver vazio.
        percentual_outliers_soma = (
                                           num_outliers_soma / num_total_linhas_df_limpo) * 100  # Calcula o percentual de linhas com outliers na soma em relação ao total de linhas.
        print(
            f"Percentual do total de linhas com outliers na soma das frações texturais: {percentual_outliers_soma:.2f}%")  # Imprime o percentual.
    else:  # Caso o DataFrame esteja vazio.
        print(
            "Não há linhas no DataFrame para calcular o percentual de outliers na soma de textura.")  # Informa que o cálculo não é possível.
    df_limpo.drop(columns=['Soma_Textura_Temp'],
                  inplace=True)  # Remove a coluna temporária 'Soma_Textura_Temp' do DataFrame.
else:  # Caso uma ou mais colunas de textura não sejam encontradas no DataFrame.
    print(
        f"⚠️ Uma ou mais colunas de textura ({', '.join(cols_textura)}) não encontradas. Verificação de outliers na soma pulada.")  # Informa que a verificação foi pulada.
print("✅ Verificação de outliers na soma das frações texturais concluída.")  # Sinaliza o fim da subseção.

# ------------------------------------------------------------------------------
# Subseção 4.2: Outliers na Coluna pH
# ------------------------------------------------------------------------------
# Verifica se os valores na coluna 'pH' estão dentro da faixa fisicamente plausível (0 a 14).
# Valores fora dessa faixa são considerados outliers.

print(f"\n--- Verificando outliers na coluna '{col_ph_nome}' ---")  # Título da subseção.
print(f"    (Outlier identificado se pH < 0 ou pH > 14)")  # Explica o critério de outlier para pH.
if col_ph_nome in df_limpo.columns:  # Verifica se a coluna de pH existe no DataFrame.
    df_limpo[col_ph_nome] = pd.to_numeric(df_limpo[col_ph_nome],
                                          errors='coerce')  # Converte a coluna de pH para tipo numérico; erros de conversão viram NaN.

    condicao_outlier_ph = (df_limpo[col_ph_nome] < 0) | (
            df_limpo[col_ph_nome] > 14)  # Define a condição lógica para um valor de pH ser considerado outlier.
    outliers_ph = df_limpo[df_limpo[
                               col_ph_nome].notna() & condicao_outlier_ph]  # Filtra as linhas que atendem à condição de outlier de pH e cujo pH não é NaN.

    linhas_ph_nan = df_limpo[
        df_limpo[col_ph_nome].isna()]  # Identifica linhas onde o pH é NaN (após a tentativa de conversão).
    num_outliers_ph = len(outliers_ph)  # Conta o número de linhas com outliers de pH.

    if num_outliers_ph > 0:  # Verifica se foram encontrados outliers de pH.
        print(
            f"Encontradas {num_outliers_ph} linha(s) com outliers de pH (valores < 0 ou > 14):")  # Informa o número de outliers de pH.
        if coluna_id_existe_em_df_limpo:  # Se a coluna ID existe.
            for index, row in outliers_ph.iterrows():  # Itera sobre as linhas com outliers de pH.
                if coluna_id in row:
                    print(f"  ID: {row[coluna_id]}, pH: {row[col_ph_nome]}")  # Imprime o ID e o valor de pH.
                else:
                    print(f"  Índice: {index}, pH: {row[col_ph_nome]} (Coluna ID não encontrada na linha)")
        else:  # Caso a coluna ID não exista.
            print(
                "  (Coluna ID não disponível ou não encontrada no df_limpo, mostrando índices das linhas):")  # Informa que os índices serão mostrados.
            for index, row in outliers_ph.iterrows():  # Itera sobre as linhas com outliers de pH.
                print(f"  Índice: {index}, pH: {row[col_ph_nome]}")  # Imprime o índice e o valor de pH.
    else:  # Caso nenhum outlier de pH tenha sido encontrado.
        print(
            "Nenhum outlier de pH (valores < 0 ou > 14) encontrado em linhas com dados de pH válidos.")  # Informa que não há outliers de pH.

    if not linhas_ph_nan.empty:  # Verifica se houve linhas com pH NaN.
        print(
            f"Adicionalmente, {len(linhas_ph_nan)} linha(s) continham valores não numéricos/ausentes na coluna '{col_ph_nome}'.")  # Informa sobre linhas com pH NaN.
        if coluna_id_existe_em_df_limpo and coluna_id in linhas_ph_nan.columns:  # Se a coluna ID existe.
            print(
                f"  IDs dessas linhas com pH NaN: {linhas_ph_nan[coluna_id].tolist()}")  # Lista os IDs das linhas com pH NaN.
        else:  # Caso a coluna ID não exista.
            print(
                f"  Índices dessas linhas com pH NaN: {linhas_ph_nan.index.tolist()}")  # Lista os índices das linhas com pH NaN.

    if num_total_linhas_df_limpo > 0:  # Evita divisão por zero se o DataFrame estiver vazio.
        percentual_ph_outliers = (
                                         num_outliers_ph / num_total_linhas_df_limpo) * 100  # Calcula o percentual de linhas com outliers de pH em relação ao total de linhas.
        print(
            f"Percentual do total de linhas com outliers de pH: {percentual_ph_outliers:.2f}%")  # Imprime o percentual.
    else:  # Caso o DataFrame esteja vazio.
        print(
            "DataFrame vazio, não é possível calcular percentual de outliers de pH.")  # Informa que o cálculo não é possível.
else:  # Caso a coluna de pH não seja encontrada.
    print(
        f"⚠️ Coluna '{col_ph_nome}' não encontrada no dataset. Verificação de outliers de pH pulada.")  # Informa que a verificação foi pulada.
print("✅ Verificação de outliers de pH concluída.")  # Sinaliza o fim da subseção.
print("✅ Identificação de Outliers (Textura e pH) concluída.")  # Sinaliza o fim da seção de identificação de outliers.

# ==============================================================================
# SEÇÃO 5: ANÁLISE DA COLUNA "Adequação MILHO" (PERCENTUAL E GRÁFICO)
# ==============================================================================
# Esta seção calcula a distribuição percentual das classes na coluna alvo ("Adequação MILHO")
# e exibe um gráfico de barras com esses percentuais, na ordem "Baixa", "Média", "Alta".

# --- 5. ANÁLISE DA COLUNA "Adequação MILHO" (PERCENTUAL E GRÁFICO) ---
print(f"\n--- Análise da coluna '{coluna_alvo}' ---")  # Título da seção.
if coluna_alvo in df_limpo.columns:  # Verifica se a coluna alvo existe no DataFrame.
    ordem_desejada_classes = ['Baixa', 'Média',
                              'Alta']  # Define a ordem desejada para as classes no gráfico e na listagem.
    percentual_classes = df_limpo[coluna_alvo].value_counts(normalize=True).mul(
        100)  # Calcula a frequência relativa (normalizada) de cada classe e multiplica por 100 para obter a porcentagem.
    percentual_classes = percentual_classes.reindex(ordem_desejada_classes).fillna(
        0)  # Reordena as classes conforme a lista 'ordem_desejada_classes' e preenche com 0 classes que não existirem nos dados.

    print("Porcentagem de cada classe (ordenado):")  # Imprime um título para a listagem das porcentagens.
    for classe, percent in percentual_classes.items():  # Itera sobre as classes e suas porcentagens (já ordenadas).
        print(f"  {classe}: {percent:.2f}%")  # Imprime cada classe e sua respectiva porcentagem formatada.

    plt.figure(figsize=(8, 5))  # Cria uma nova figura matplotlib para o gráfico com tamanho 8x5 polegadas.
    sns.barplot(  # Cria um gráfico de barras usando a biblioteca seaborn.
        x=percentual_classes.index,  # Define as categorias do eixo X (nomes das classes, já ordenados).
        y=percentual_classes.values,  # Define os valores do eixo Y (as porcentagens).
        hue=percentual_classes.index,
        # Usa as classes também para o parâmetro 'hue', permitindo aplicar a paleta por classe e resolver um FutureWarning do seaborn.
        palette="viridis",  # Define a paleta de cores "viridis" para o gráfico.
        legend=False,  # Desativa a legenda automática para 'hue', pois seria redundante com os rótulos do eixo X.
        order=ordem_desejada_classes  # Define explicitamente a ordem das barras no gráfico.
    )
    plt.title(f'Percentual de Classes - {coluna_alvo}')  # Define o título do gráfico.
    plt.ylabel('Porcentagem (%)')  # Define o rótulo do eixo Y.
    plt.xlabel('Classe')  # Define o rótulo do eixo X.
    plt.ylim(0,
             max(percentual_classes.values) + 10 if not percentual_classes.empty else 100)  # Ajusta o limite superior do eixo Y para melhor visualização dos valores anotados sobre as barras.
    for i, v in enumerate(
            percentual_classes.values):  # Itera sobre os valores das porcentagens para anotá-los no gráfico.
        plt.text(i, v + 1, f"{v:.1f}%", color='black',
                 ha='center')  # Adiciona o texto da porcentagem acima de cada barra.
    plt.tight_layout()  # Ajusta o layout do gráfico para evitar sobreposição de elementos.
    plt.show()  # Exibe o gráfico gerado.
    print(
        f"✅ Análise percentual e gráfico da coluna '{coluna_alvo}' concluídos com ordem personalizada.")  # Sinaliza a conclusão da etapa.
else:  # Caso a coluna alvo não seja encontrada.
    print(
        f"⚠️ Coluna '{coluna_alvo}' não encontrada no dataset. Esta etapa será pulada.")  # Informa que a etapa foi pulada.

# ==============================================================================
# SEÇÃO 6: MAPEAMENTO DA COLUNA "Adequação MILHO" PARA VALORES NUMÉRICOS
# ==============================================================================
# Esta seção converte os valores textuais da coluna "Adequação MILHO"
# ("Baixa", "Média", "Alta") para valores numéricos (0, 5, 10, respectivamente).

# --- 6. ALTERAÇÃO DA COLUNA "Adequação MILHO" PARA VALORES DECIMAIS ---
print(f"\n--- Mapeando valores da coluna '{coluna_alvo}' ---")  # Título da seção.
if coluna_alvo in df_limpo.columns:  # Verifica se a coluna alvo existe.
    mapeamento = {'Baixa': 0, 'Média': 5, 'Alta': 10}  # Define o dicionário de mapeamento.
    valores_unicos_antes = df_limpo[
        coluna_alvo].unique()  # Armazena os valores únicos da coluna antes do mapeamento para verificação.
    df_limpo[coluna_alvo] = df_limpo[coluna_alvo].map(
        mapeamento)  # Aplica o mapeamento à coluna. Valores não presentes no dicionário 'mapeamento' se tornarão NaN.

    if df_limpo[
        coluna_alvo].isnull().any():  # Verifica se algum valor resultou em NaN após o mapeamento (indicando que não estava no dicionário).
        print(
            f"⚠️ Atenção: Alguns valores na coluna '{coluna_alvo}' não foram mapeados e resultaram em NaN.")  # Alerta sobre valores não mapeados.
        print(
            f"   Valores únicos originais que podem não ter sido mapeados: {valores_unicos_antes}")  # Mostra os valores originais.
        print(
            f"   Valores após mapeamento (NaNs indicam falha): {df_limpo[coluna_alvo].unique()}")  # Mostra os valores após o mapeamento, incluindo NaNs.
    else:  # Caso todos os valores tenham sido mapeados com sucesso.
        print(
            f"Todos os valores da coluna '{coluna_alvo}' foram mapeados com sucesso.")  # Informa sucesso no mapeamento.
    print(f"✅ Mapeamento da coluna '{coluna_alvo}' concluído.")  # Sinaliza a conclusão da etapa.
else:  # Caso a coluna alvo não exista.
    print(
        f"⚠️ Coluna '{coluna_alvo}' não encontrada. Mapeamento não realizado.")  # Informa que o mapeamento foi pulado.

# ==============================================================================
# SEÇÃO 7: NORMALIZAÇÃO MIN-MAX DAS COLUNAS NUMÉRICAS (EXCETO ID INICIALMENTE)
# ==============================================================================
# Esta seção normaliza todas as colunas numéricas do DataFrame (a coluna 'ID'
# já foi tratada ou será removida antes do salvamento final). Os valores
# normalizados são arredondados para 5 casas decimais.

# --- 7. NORMALIZAÇÃO MIN-MAX DE TODAS AS COLUNAS ---
print("\n--- Normalizando colunas para o intervalo [0, 1] ---")  # Título da seção.

# Seleciona todas as colunas que são de tipo numérico para normalização.
# A coluna 'ID' (se ainda existir e for numérica) será normalizada aqui se não for explicitamente excluída.
# No entanto, a prática comum é não normalizar IDs. A remoção final do ID é feita na Seção 9.
colunas_para_normalizar = df_limpo.select_dtypes(include=np.number).columns.tolist()

# Lógica para excluir 'ID' da normalização se 'coluna_id' estiver definida e presente
if coluna_id and isinstance(coluna_id, str) and coluna_id in colunas_para_normalizar:
    print(f"Coluna '{coluna_id}' será explicitamente excluída da normalização nesta etapa.")
    colunas_para_normalizar.remove(coluna_id)
elif coluna_id and isinstance(coluna_id, str) and coluna_id not in df_limpo.columns:
     print(f"Coluna ID ('{coluna_id}') não encontrada no DataFrame neste ponto, portanto não será excluída da lista de normalização.")
elif not (coluna_id and isinstance(coluna_id, str)):
    print("Nenhuma coluna ID válida especificada para exclusão da normalização.")


if not colunas_para_normalizar:  # Verifica se há colunas restantes para normalizar.
    print(
        "⚠️ Nenhuma coluna numérica (restante) encontrada para normalizar.")
else:  # Caso haja colunas para normalizar.
    print(f"Colunas a serem normalizadas: {colunas_para_normalizar}")  # Lista as colunas que serão normalizadas.
    scaler = MinMaxScaler()  # Cria uma instância do normalizador MinMaxScaler.
    df_limpo[colunas_para_normalizar] = scaler.fit_transform(
        df_limpo[colunas_para_normalizar])  # Aplica a normalização (ajusta e transforma) às colunas selecionadas.
    df_limpo[colunas_para_normalizar] = df_limpo[colunas_para_normalizar].round(
        5)  # Arredonda os valores normalizados para 5 casas decimais.
    print(
        f"✅ Normalização Min-Max concluída para as colunas selecionadas com arredondamento para 5 casas decimais.")  # Sinaliza a conclusão.

# ==============================================================================
# SEÇÃO 8: EMBARALHAMENTO DAS LINHAS DO DATASET
# ==============================================================================
# Esta seção embaralha as linhas do DataFrame df_limpo de forma aleatória.
# Isso é útil para remover qualquer ordem preexistente nos dados que possa
# influenciar etapas subsequentes.
# Um 'random_state' é usado para garantir que o embaralhamento seja reprodutível.

# --- 8. EMBARALHAMENTO DAS LINHAS DO DATASET ---
print(f"\n--- Embaralhando as linhas do DataFrame ---")  # Título da seção.
if not df_limpo.empty:  # Verifica se o DataFrame não está vazio.
    df_limpo = df_limpo.sample(frac=1, random_state=42).reset_index(
        drop=True)  # Embaralha 100% das linhas (frac=1) usando um estado aleatório fixo (42) para reprodutibilidade, e reseta o índice do DataFrame resultante, descartando o índice antigo.
    print(
        f"✅ Linhas do DataFrame embaralhadas randomicamente (com random_state=42) e índice resetado.")  # Sinaliza a conclusão.
else:  # Caso o DataFrame esteja vazio.
    print("⚠️ DataFrame está vazio. Nenhuma linha para embaralhar.")  # Informa que não há linhas para embaralhar.

print("\n--- Pré-processamento de dados concluído. Iniciando salvamento. ---")  # Mensagem de transição.

# ==============================================================================
# SEÇÃO 9: REMOÇÃO DE COLUNAS ESPECÍFICAS E SALVAR DATAFRAME PRÉ-PROCESSADO
# ==============================================================================
# Esta seção primeiro remove as colunas 'ID', 'Fe ppm' e 'Mn ppm'.
# Em seguida, verifica se a pasta de saída para os CSVs existe e a cria
# caso não exista. Por fim, salva o DataFrame 'df_limpo' modificado
# em um único arquivo CSV.

# --- 9. REMOÇÃO DE COLUNAS E SALVAR DATAFRAME PRÉ-PROCESSADO COMPLETO ---

# Definir colunas a serem removidas antes de salvar
colunas_para_remover_antes_salvar = [coluna_id, 'Fe ppm', 'Mn ppm'] # Usa a variável coluna_id

if not df_limpo.empty:
    colunas_existentes_no_df = df_limpo.columns.tolist()
    colunas_a_remover_efetivamente = [col for col in colunas_para_remover_antes_salvar if col in colunas_existentes_no_df]

    if colunas_a_remover_efetivamente:
        df_limpo.drop(columns=colunas_a_remover_efetivamente, inplace=True, errors='ignore')
        print(f"\nColunas removidas antes de salvar: {colunas_a_remover_efetivamente}")
        print(f"Dimensões após remover colunas específicas: {df_limpo.shape[0]} linhas, {df_limpo.shape[1]} colunas.")
    else:
        print("\nNenhuma das colunas especificadas para remoção ('ID', 'Fe ppm', 'Mn ppm') foi encontrada no DataFrame antes de salvar.")
else:
    print("\nDataFrame 'df_limpo' está vazio, nenhuma coluna para remover.")


# Verificar se a pasta de saída não existe e criá-la
if not os.path.exists(caminho_arquivos_saida_CSV):  # Verifica se o caminho para a pasta de saída não existe.
    try:  # Tenta criar a pasta.
        os.makedirs(
            caminho_arquivos_saida_CSV)  # Cria a pasta de saída, incluindo quaisquer diretórios pais necessários.
        print(f"Pasta '{caminho_arquivos_saida_CSV}' criada com sucesso!")  # Informa que a pasta foi criada.
    except OSError as e:  # Captura erros que possam ocorrer durante a criação da pasta.
        print(f"Erro ao criar a pasta '{caminho_arquivos_saida_CSV}': {e}")  # Imprime a mensagem de erro.
else:  # Caso a pasta já exista.
    print(f"Pasta '{caminho_arquivos_saida_CSV}' já existe.")  # Informa que a pasta já existe.

print(f"\n--- Salvando o DataFrame pré-processado completo (sem as colunas removidas) ---")  # Título da operação de salvamento.
if not df_limpo.empty:  # Verifica se o DataFrame não está vazio.
    nome_base_arquivo_original = os.path.splitext(os.path.basename(caminho_arquivo))[
        0]  # Extrai o nome base do arquivo de leitura original (sem a extensão).
    nome_arquivo_completo_salvo = f"{nome_base_arquivo_original}{sufixo_preprocessado_completo}.csv"  # Constrói o nome do arquivo de saída para o dataset completo.
    try:  # Tenta salvar o DataFrame.
        df_limpo.to_csv(os.path.join(caminho_arquivos_saida_CSV, nome_arquivo_completo_salvo), index=False, sep=';',
                        decimal=',')  # Salva o DataFrame 'df_limpo' no caminho especificado, sem o índice, usando ';' como separador e ',' como decimal.
        print(
            f"✅ DataFrame pré-processado completo salvo em: {os.path.join(caminho_arquivos_saida_CSV, nome_arquivo_completo_salvo)}")  # Confirma o salvamento e mostra o caminho completo.
    except Exception as e:  # Captura erros durante o salvamento.
        print(f"❌ Erro ao salvar o DataFrame completo: {e}")  # Imprime a mensagem de erro.
else:  # Caso o DataFrame esteja vazio.
    print("⚠️ DataFrame está vazio. Nenhum arquivo completo para salvar.")  # Informa que o arquivo não foi salvo.


# ==============================================================================
# SEÇÃO 10: VISUALIZAÇÃO DA TABELA FINAL PRÉ-PROCESSADA EM NOVA JANELA
# ==============================================================================
# Esta seção exibe o DataFrame final ('df_limpo', já sem as colunas removidas)
# em uma nova janela GUI utilizando Tkinter.

# Função para exibir DataFrame em uma nova janela Tkinter
def exibir_dataframe_em_janela(df, titulo_janela="Visualização do DataFrame"):
    """
    Exibe um DataFrame pandas em uma nova janela Tkinter usando ttk.Treeview.
    """
    if df.empty:
        print("DataFrame está vazio. Nada para exibir na janela.")
        return

    janela = tk.Tk()
    janela.title(titulo_janela)
    janela.geometry("800x600")  # Tamanho inicial da janela

    frame = ttk.Frame(janela, padding="10")
    frame.pack(expand=True, fill='both')

    tree = ttk.Treeview(frame, show='headings')

    # Definir colunas
    tree["columns"] = list(df.columns)
    for col in df.columns:
        tree.column(col, anchor=tk.W, width=100, minwidth=50)  # Ajuste a largura conforme necessário
        tree.heading(col, text=col, anchor=tk.W)

    # Inserir dados
    for index, row in df.iterrows():
        tree.insert("", tk.END, values=list(row))

    # Barras de rolagem
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    vsb.pack(side='right', fill='y')
    tree.configure(yscrollcommand=vsb.set)

    hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    hsb.pack(side='bottom', fill='x')
    tree.configure(xscrollcommand=hsb.set)

    tree.pack(expand=True, fill='both')
    janela.mainloop()


# --- 10. VISUALIZAÇÃO DA TABELA FINAL PRÉ-PROCESSADA ---
print("\n\n--- Tabela Final Pré-processada (após remoção de colunas e embaralhamento) ---")  # Título da seção.
if not df_limpo.empty:  # Verifica se o DataFrame final não está vazio.
    # df_para_visualizar é uma cópia do df_limpo, que já teve colunas removidas na Seção 9
    df_para_visualizar = df_limpo.copy()

    # A coluna 'ID' especificada por 'coluna_id' provavelmente já foi removida.
    # A lógica abaixo tentará mover a coluna 'ID' se ela, por algum motivo, ainda existir.
    # Normalmente, após a Seção 9, 'coluna_id' não estará em df_para_visualizar.columns
    id_col_existe_no_df_final = False
    if coluna_id and isinstance(coluna_id, str) and coluna_id in df_para_visualizar.columns:
        id_col_existe_no_df_final = True

    if id_col_existe_no_df_final:
        print(f"Coluna '{coluna_id}' (se ainda presente) será movida para o início para visualização.")
        cols_ordenadas = [coluna_id] + [col for col in df_para_visualizar.columns if
                                        col != coluna_id]
        df_para_visualizar = df_para_visualizar[cols_ordenadas]
    elif coluna_id and isinstance(coluna_id, str):
        print(
            f"Informação: Coluna ID ('{coluna_id}') não foi encontrada no DataFrame final para visualização (provavelmente removida antes de salvar). A ordem das colunas será mantida.")
    else:
        print(
            "Nenhuma coluna ID especificada para mover para o início, ou ela já foi removida. A ordem das colunas será mantida.")

    print(
        f"Visualizando o DataFrame final com {len(df_para_visualizar)} linhas e {len(df_para_visualizar.columns)} colunas em uma nova janela.")

    try:
        exibir_dataframe_em_janela(df_para_visualizar, titulo_janela="DataFrame Pré-processado Final")
        print("✅ Janela de visualização do DataFrame foi aberta. Feche a janela para finalizar o script.")
    except Exception as e:
        print(f"❌ Erro ao tentar exibir o DataFrame na janela Tkinter: {e}")
        print("Como alternativa, exibindo as primeiras 10 linhas no console:")
        print(df_para_visualizar.head(10).to_string())

else:  # Caso o DataFrame final ('df_limpo') esteja vazio.
    print("O DataFrame final ('df_limpo') está vazio. Nada para visualizar.")

print("\n--- Script Finalizado ---")  # Mensagem final indicando que todo o script foi executado.