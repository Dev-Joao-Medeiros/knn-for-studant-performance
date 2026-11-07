# Desempenho de Estudantes com Machine Learning: KNN vs. Árvore de Decisão

Classificação de estudantes como **aprovados** ou **reprovados** em Matemática usando o algoritmo **KNN (K-Nearest Neighbors)** e comparação com uma **Árvore de Decisão**, no dataset *Student Performance* da UCI.

## Contexto acadêmico

Este projeto é um trabalho da disciplina de **Inteligência Computacional**, do curso de Engenharia da Computação da **UFPA (Campus Castanhal)**.

O objetivo da atividade é **analisar e comparar modelos de classificação para descobrir qual deles é o melhor** em um mesmo problema. O trabalho foi feito em dupla:

- **KNN:** implementado neste repositório.
- **Árvore de Decisão:** implementada pelo colega de dupla, com o mesmo problema, os mesmos atributos e a mesma divisão de dados, para que a comparação seja justa.

## O que é o KNN

O **K-Nearest Neighbors (K vizinhos mais próximos)** é um algoritmo de aprendizado supervisionado baseado em **instâncias**. Ele não constrói um modelo explícito: guarda os dados de treino e, para classificar um novo exemplo, faz três passos:

1. Calcula a **distância** (aqui, euclidiana ou Manhattan) entre o novo exemplo e todos os exemplos de treino.
2. Seleciona os **K mais próximos**.
3. **Vota**: a classe mais frequente entre esses K vizinhos vira a previsão. Em regressão, usa-se a média dos valores.

Pontos de atenção:

- **K pequeno** tende a overfitting (sensível a ruído); **K grande** tende a favorecer a classe majoritária.
- Como usa distância, **exige padronização** das variáveis numéricas. Sem isso, uma coluna como `absences` (0 a 93) dominaria uma como `studytime` (1 a 4).
- Variáveis categóricas precisam virar números (**One-Hot Encoding**).
- Dá o **mesmo peso a todas as colunas**, então atributos irrelevantes prejudicam o resultado.

## Para que serve

O KNN é usado em **classificação** e **regressão**. Aplicações comuns:

- sistemas de recomendação;
- reconhecimento de padrões e de imagens;
- detecção de anomalias;
- diagnóstico e triagem a partir de casos semelhantes;
- previsão de desempenho, como neste trabalho: dado o perfil de um aluno, prever se ele será aprovado a partir de alunos parecidos.

É simples e fácil de interpretar, mas pode ser lento com muitos dados e sofre quando há muitas variáveis pouco informativas.

## Dataset

- **Fonte:** [UCI Machine Learning Repository, Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- **Arquivo usado:** `student-mat.csv` (disciplina de Matemática), separado por `;`
- **Tamanho:** 395 estudantes e 33 colunas originais (perfil, família, hábitos de estudo, faltas e as notas G1, G2 e G3, de 0 a 20)

### Definição do problema

- **Alvo:** `G3 >= 10` → **Aprovado (1)**; `G3 < 10` → **Reprovado (0)**
- **Distribuição:** 265 aprovados (67,1%) e 130 reprovados (32,9%). As classes são **desbalanceadas**.
- **Atributos de entrada:** 30. `G1`, `G2` e `G3` são removidos para evitar vazamento de informação (as notas anteriores quase entregam a nota final).
- **Classe positiva nas métricas:** **Reprovado** (`pos_label=0`).

Como um modelo que sempre prevê "aprovado" já acerta 67%, a acurácia sozinha engana. Por isso as métricas principais são a **revocação** e o **F1-score da classe Reprovado**.

## Metodologia

1. **Divisão:** 80% treino (316 alunos) e 20% teste (79 alunos), estratificada, com `random_state=42`.
2. **Pré-processamento:** One-Hot Encoding nas variáveis categóricas e `StandardScaler` nas numéricas (30 atributos viram 56 colunas), dentro de um `Pipeline` para evitar vazamento de dados.
3. **Busca de hiperparâmetros:** `GridSearchCV` com validação cruzada estratificada de 5 folds, otimizando o **F1 da classe Reprovado**:
   - `n_neighbors`: 1 a 30
   - `weights`: `uniform` e `distance`
   - `metric`: `euclidean` e `manhattan`
4. **Avaliação:** uma única vez no conjunto de teste, comparando com um `DummyClassifier` (sempre prevê "aprovado") e com a Árvore de Decisão.

Trecho principal:

```python
preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", StandardScaler(), num_cols),
])
pipe = Pipeline([("prep", preprocess), ("knn", KNeighborsClassifier())])

f1_reprovado = make_scorer(f1_score, pos_label=0)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    pipe,
    {
        "knn__n_neighbors": range(1, 31),
        "knn__weights": ["uniform", "distance"],
        "knn__metric": ["euclidean", "manhattan"],
    },
    cv=cv, scoring=f1_reprovado, n_jobs=-1,
)
grid.fit(X_train, y_train)
```

## Resultados (conjunto de teste, 79 alunos)

O GridSearchCV escolheu **K = 2, distância euclidiana e pesos uniformes** (F1 de Reprovado na validação cruzada: 0,4293).

| Modelo | Acurácia | Revocação (Reprovado) | F1-score (Reprovado) | AUC |
| --- | --- | --- | --- | --- |
| Dummy (sempre "aprovado") | 67,09% | 0% | 0% | 0,500 |
| **Árvore de Decisão** | **67,09%** | 57,69% | **53,57%** | **0,647** |
| KNN (K = 2, otimizado) | 54,43% | **76,92%** | 52,63% | 0,565 |
| KNN (K = 5, padrão do scikit-learn) | 58,23% | 19,23% | 23,26% | não calculada |

Detalhes do KNN no teste (26 reprovados e 53 aprovados):

- **K = 2:** encontra 20 dos 26 reprovados, mas marca 30 aprovados como reprovados. A precisão de Reprovado é 0,40 e a acurácia fica abaixo da do Dummy.
- **K = 5:** o comportamento se inverte. Ele marca poucos alunos como reprovados e acerta só 5 dos 26.
- A AUC de 0,565 do KNN com K = 2 está perto de um classificador aleatório (0,5).

Os valores do KNN com K = 2 foram obtidos a partir da matriz de confusão (20, 6, 30 e 23).

## Modelo campeão: Árvore de Decisão

A **Árvore de Decisão** foi o melhor modelo da comparação, pelos seguintes motivos:

- **Melhor equilíbrio:** mantém a acurácia do Dummy (67,09%) e ainda identifica 57,69% dos reprovados. O KNN com K = 2 tem revocação maior, mas ao custo de muitos falsos alarmes e de acurácia bem menor.
- **Maior AUC (0,647)**, contra 0,565 do KNN, o que indica melhor capacidade de separar as classes.
- **Seleciona sozinha os atributos relevantes:** usa `failures` já na raiz, enquanto o KNN trata todas as colunas com o mesmo peso.
- **Não depende de escala nem de outliers:** divide por limiares (por exemplo, `absences <= 21,5`), enquanto o KNN depende de distância.

## Por que o KNN foi pior neste dataset

- **Muitas colunas pouco informativas:** após o One-Hot, são 56 colunas e nenhuma tem correlação forte com o resultado (a maior é `failures`, cerca de -0,34). O ruído dilui os poucos atributos úteis na distância.
- **Poucos dados:** apenas 316 alunos no treino.
- **Classes desbalanceadas:** com K maior, a votação tende à classe majoritária (aprovados); com K muito pequeno, o resultado depende de empates.
- **Sensibilidade a outliers e escala:** a coluna `absences` tem cauda longa (até cerca de 75) e pode distorcer as distâncias.

## Limitações

- O conjunto de teste tem só **26 reprovados**. Mudar a previsão de um aluno altera a revocação em cerca de 3,8 pontos percentuais, então os números têm margem de erro.
- Foi usado **um único split** treino/teste. Repetir com várias sementes, ou com validação cruzada também na avaliação final, daria uma estimativa mais estável.
- Experimentos que podem melhorar o KNN: usar apenas K ímpares, otimizar `balanced_accuracy` ou `roc_auc`, ou reduzir os atributos antes do KNN.

## Como executar

O projeto foi desenvolvido no **Google Colab**.

1. Baixe o dataset no [site da UCI](https://archive.ics.uci.edu/dataset/320/student+performance) e extraia o `student-mat.csv`.
2. Abra o notebook no Colab e faça o upload do `student-mat.csv` (ícone de pasta, à esquerda).
3. Execute as células em ordem.

Para rodar localmente:

```bash
pip install pandas scikit-learn matplotlib
```

Bibliotecas usadas: `pandas`, `scikit-learn` e `matplotlib`.

## Autoria

Trabalho da disciplina de Inteligência Computacional, UFPA (Campus Castanhal), novembro de 2026.

- **KNN:** João Medeiros
- **Árvore de Decisão:** colega de dupla
