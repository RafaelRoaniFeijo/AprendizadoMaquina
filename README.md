# Classificação de Adequação do Solo para Cultivo de Milho

## 1. Objetivo do Projeto

Este projeto, desenvolvido para a disciplina de Aprendizado de Máquina, tem como objetivo criar e avaliar modelos preditivos para classificar a adequação do solo para o cultivo de milho em três categorias: 'Baixa', 'Média' e 'Alta'.

O foco principal do trabalho foi desenvolver uma metodologia robusta para lidar com o severo desbalanceamento de classes presente no dataset original, onde a classe 'Alta' era significativamente sub-representada.

## 2. Metodologia

O fluxo de trabalho foi estruturado para garantir a validade e a reprodutibilidade dos resultados, seguindo as melhores práticas da área.

### 2.1. Pré-processamento e Balanceamento

1.  **Normalização Inicial:** Todos os dados de atributos físico-químicos do solo passaram por uma etapa de normalização utilizando `MinMaxScaler`.
2.  **Aumento de Dados Sintético:** Para corrigir o desbalanceamento, foi implementada uma técnica de sobreamostragem (oversampling) customizada. Este método gerou iterativamente novas amostras sintéticas para as classes minoritárias, criando um dataset de treinamento final equilibrado.

### 2.2. Estrutura de Avaliação

A avaliação dos modelos foi realizada em duas configurações principais para medir o impacto do balanceamento:

* **Configuração *Baseline*:** Treinamento e teste realizados utilizando o dataset original desbalanceado, através de uma validação cruzada de 5 folds para estabelecer uma performance de referência.
* **Configuração *Proposta* (Principal):** Esta é a metodologia validada.
    1.  O dataset real foi inicialmente dividido, separando um conjunto de teste fixo e independente.
    2.  O conjunto de treino restante foi balanceado com a técnica de aumento de dados.
    3.  Os modelos foram treinados neste conjunto balanceado, usando uma validação cruzada de 5 folds para maior robustez.
    4.  A avaliação de cada um dos 5 modelos treinados foi feita no **conjunto de teste real e independente**, garantindo uma medição honesta da capacidade de generalização.

### 2.3. Modelos e Otimização

* **Modelos Testados:** Foram avaliados cinco algoritmos de classificação: k-Nearest Neighbors (k-NN), Árvore de Decisão, Random Forest, Support Vector Machine (SVM) e XGBoost.
* **Otimização de Hiperparâmetros:** Foi utilizado o `RandomizedSearchCV` para buscar a melhor combinação de hiperparâmetros para cada modelo, com o objetivo de maximizar a métrica F1-Score Macro.

## 3. Principais Resultados

A principal conclusão do estudo é que a estratégia de balanceamento de classes foi **crucial** para o sucesso da modelagem.

-   Na configuração *Baseline*, os modelos foram incapazes de identificar a classe minoritária 'Alta' (F1-score de 0.0).
-   Na configuração *Proposta*, o modelo **XGBoost** se destacou como o de melhor desempenho, elevando o F1-score da classe 'Alta' para **0.810**, provando a abordagem.
-   Análises de diagnóstico (curvas de aprendizado e calibração) confirmaram o modelo XGBoost como melhor.

## 4. Como Executar o Projeto

### 4.1. Execução

O notebook principal que contém todo o fluxo de trabalho, desde a carga dos dados até a geração das análises e figuras, é o `TreinoSinteticoReal_ValReal.ipynb`.

## 5. Estrutura do Repositório

-   `/04_Treinamento/TreinoSinteticoReal_ValReal.ipynb`: Notebook principal com a metodologia proposta e todas as análises.
-   `/04_Treinamento/TreinoReal_ValReal.ipynb`: Notebook secundário para realização do comparativo.
-   `/00_Datasets/`: Contém os datasets utilizados nos experimentos.
-   `Trabalho_Final_Aprendizagem_de_Máquina_UFRGS.pdf`: A versão final do artigo científico.
-   `README.md`: Este arquivo.
