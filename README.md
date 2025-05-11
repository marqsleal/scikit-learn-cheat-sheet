# Machine Learning Cheat Sheet - Scikit-Learn
Cheat Sheet da biblioteca [Scikit-Learn](https://scikit-learn.org/stable/).

# Instalando

```bash
pip install scikit-learn
```

# 1. Data Pre-Processing - `sklearn.preprocessing`: 

## 1.1 `StandardScaler`:
Padroniza features numéricas removendo a média e escalando para a unidade de variância.  

**Código**:
```python
from sklearn.preprocessing import StandardScaler

data = [[0, 0], [0, 0], [1, 1], [1, 1]]

scaler = StandardScaler()

scaler.fit(data)

print(scaler.mean_)
# [0.5 0.5]

print(scaler.transform(data))
# [[-1. -1.]
#  [-1. -1.]
#  [ 1.  1.]
#  [ 1.  1.]]

print(scaler.transform([[2, 2]]))
#[[3. 3.]]
```

**Casos de Uso**:
 - Quando a feature possui uma distribuição aproximadamente Gaussiana (normal). 

## 1.2 `MinMaxScaler`:
Escala as features numéricas para um range específico. 

**Código**:
```python
from sklearn.preprocessing import MinMaxScaler

data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]

scaler = MinMaxScaler()

scaler.fit(data)

print(scaler.data_max_)
# [ 1. 18.]

print(scaler.transform(data))
# [[0.   0.  ]
#  [0.25 0.25]
#  [0.5  0.5 ]
#  [1.   1.  ]]

print(scaler.transform([[2, 2]]))
# [[1.5 0. ]]
```

**Casos de Uso**:
 - Quando a feature não possui uma distribuição normal.  

## 1.3 `Binarizer`:
`Binarizer` binariza os dados de acordo com o `threshold` escolhid.

```python
from sklearn.preprocessing import Binarizer

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]

transformer = Binarizer().fit(X)  # fit does nothing.

transformer.transform(X)
# array([[1., 0., 1.],
#        [1., 0., 0.],
#        [0., 1., 0.]])
```

**Casos de uso**:
 - Em NLP, quando se deseja saber se apenas um termo aparece ou não.
 - Em CV, se tratando de greyscaling.
 - Modelos como Naive Bayes Bernoulli requerem variáveis booleanas.
 - Criação de features booleanas.
 - Preparação de dados para modelos de classificação.

## 1.4: `LabelEncoder`:
Realiza o encoding da feature com valores entre 0 e n. classes - 1.
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])

le.transform(["tokyo", "tokyo", "paris"])
# array([2, 2, 1]...)

# Reverter o processo:
list(le.inverse_transform([2, 2, 1]))
# [np.str_('tokyo'), np.str_('tokyo'), np.str_('paris')]
```
**Casos de Uso**:
 - Quando as classes tem uma hierarquia ou ordem natural (“Baixo” < “Médio” < “Alto”)
 - Variáveis-Alvo (y)
 - Variáveis de Entrada (X), arvores de decisão e random forests podem lidar diretamente com inteiros sem supor ordenação
 - Alto número de classes

## 1.4 `OneHotEncoder`:
Realiza o encoding de uma feature categórica em array numérico one-hot 
```python
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown='ignore')
X = [['Male', 1], ['Female', 3], ['Female', 2]]
enc.fit(X)
# [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]

enc.transform([['Female', 1], ['Male', 4]]).toarray()
# array(
#   [[1., 0., 1., 0., 0.],
#   [0., 1., 0., 0., 0.]])

enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])
# array([['Male', 1],
#       [None, 2]], dtype=object)

enc.get_feature_names_out(['gender', 'group'])
# array(['gender_Female', 'gender_Male', 'group_1', 'group_2', 'group_3'], ...)
```
**Casos de Uso**:
 - Quando não há ordenação entre as categorias ("Azul", "Vermelho", "Amarelo")
 - Variáveis de Entrada (X), modelos lineares e SVMs esperam features ortogonais, 
 - Alto número de classes

## 1.5 `PolynomialFeatures`:
`PolynomialFeatures` é um pré-processador que gera uma nova matriz de características contendo todas as combinações polinomiais de variáveis de entrada até um grau especificado (`degree`).  

**Parâmetros Relevantes**:
 - `degree`: do tipo int ou tupla (`min_degree`, `max_degree`). Valor padrão: `2`.
 - `interaction_only`: tipo booleano, gera termos de interação de variáveis dinstintas apenas, excluindo potenciais maiores que 1 de uma mesma variável. Valor Padrão: `False` 
 - `include_bias`: tipo booleano, inclui uma coluna constante de valores. Valor Padrão: `True`

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)

X
# array([[0, 1],
#        [2, 3],
#        [4, 5]])
poly = PolynomialFeatures(2)

poly.fit_transform(X)
# array([[ 1.,  0.,  1.,  0.,  0.,  1.],
#        [ 1.,  2.,  3.,  4.,  6.,  9.],
#        [ 1.,  4.,  5., 16., 20., 25.]])

poly = PolynomialFeatures(interaction_only=True)

poly.fit_transform(X)
# array([[ 1.,  0.,  1.,  0.],
#        [ 1.,  2.,  3.,  6.],
#        [ 1.,  4.,  5., 20.]])
```
**Casos de Uso**:
 - Em casos mais simples de regressões lineares, usado para capturar curvaturas em dados univariados.
 - Problemas com multiplas variáveis independentes, expandindo características em cenários onde interações entre variáveis podem melhorar a predição.
 - Em tarefas de classificação não linear, transformar características via polinômios pode permitir a separação de classes que seriam inseparáveis em um espaço linear original.

**Ponto de Atenção**:
 - O número de colunas cresce rapidamente com o grau e o número de variáveis, podendo gerar problemas de performance e até de estouro de memória.
 - Potências elevadas podem induzir alta correlação entre colunas.
 - Escalar antes de elevar o grau evita dominância numérica de grandes valores.

# 2. Data Imputers - `sklearn.imputer`:

## 2.1 `SimpleImputer`:
`SimpleImputer` imputa aos campos ausentes nas colunas os dados referentes a estratégia (`strategy`) passada na construção do objeto.  

**Parâmetros Relevantes**:
 - `strategy`: tipo string, escolha da estratégia utilizada na imputação dos dados, sendo estes a média (`mean`), a mediana (`median`), a moda (`most_frequent`), etc. Valor Padrão: `mean` 
```python
import numpy as np
from sklearn.impute import SimpleImputer

imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

imp_mean.fit([[7, 2, 3], [4, np.nan, 6], [10, 5, 9]])

X = [[np.nan, 2, 3], [4, np.nan, 6], [10, np.nan, 9]]

print(imp_mean.transform(X))
# [[ 7.   2.   3. ]
#  [ 4.   3.5  6. ]
#  [10.   3.5  9. ]]
```

**Casos de Uso**:
 - `mean`: ideal para variáveis cuja distribuição seja aproximadamente simétrica e sem outliers significativos, pois a média é sensível a valores extremos.
 - `median`: recomendado quando os dados apresentam outliers ou distribuições assimétricas, pois a mediana é robusta a valores extremos.
 - `most_frequent`: adequado para variáveis categóricas ou para preservar as categorias mais comuns em variáveis numéricas com distribuição multimodal.
 - `constant`: indicado quando se conhece previamente o valor de substituição apropriado (por exemplo, 0, “Desconhecido” ou outro placeholder), ou quando se deseja sinalizar explicitamente as imputações.

## 2.2 `KNNImputer`:
`KNNImputer` implementa uma técnica de imputação multivariada que utiliza o valor médio (ou outro estatístico) dos *k* vizinhos mais próximos para preencher valores ausentes, preservando estruturas locais e relações entre características.  

**Parâmetros Relevantes**:
 - `n_neighbors`: tipo int, número de *k* vizinhos a considerar na imputação. Valor padrão: `5`.
 - `weights`: tipo string, critério de ponderação, `uniform` tratando todos como iguais, `distance` mais peso aos mais próximos. Valor Padrão: `uniform`.

```python
import numpy as np
from sklearn.impute import KNNImputer

X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]

imputer = KNNImputer(n_neighbors=2)

imputer.fit_transform(X)
# array([[1. , 2. , 4. ],
#        [3. , 4. , 3. ],
#        [5.5, 6. , 5. ],
#        [8. , 8. , 7. ]])
```

**Casos de Uso**:
 - Relações não lineares ou interdependências fortes: quando características correlacionadas podem guiar a imputação.
 - Preservação de padrões locais: útil em séries temporais ou dados espaciais, onde vizinhanças refletem similaridade real.
 - Proporção moderada de faltantes: funciona bem se a fração de valores ausentes não for muito alta, para garantir vizinhos suficientes


**Ponto de Atenção**:
 - Certifique-se de que os dados foram escalados préviamente (`StandardScaler`, `MinMaxScaler`) antes de aplicar `KNNImputer`.
 - Imputação em grandes bases pode afetar a performance da pipeline.
 - Se muitos valores estiverem ausentes, podem não haver vizinhos o suficiente.

#  3. Dimentionality Reduction:

## 3.1 PCA - `sklearn.decomposition.PCA`:  
`PCA` (Analise de Componentes Principais) é uma técnica de redução de dimensionalidade linear que projeta um conjunto de dados em um de menor dimensão, preservando ao máximo a variância original dos dados. Ele funciona centralizando as variáveis, calculando os autovetores e autovalores da matriz de covariância (ou equivalentes via SVD) e ordenando esses componentes pelo quanto explicam da variância total. 

**Parâmetros Relevantes**:
 - `n_components`: Número de componentes principais a reter; pode ser inteiro, fração de variância explicada ou `None` (retém todos). Valor padrão: `None`.
 - `svd_solver`: Estratégia para computar a SVD (`auto`, `full`, `arpack`, `randomized`) permitindo balanço entre precisão e performance. Valor padrão: `auto`.
 - `whiten`: Se `True`, divide cada componente pelo seu desvio padrão, garantindo variância unitária, útil em alguns pipelines de preprocessing. Valor padrão: `False`.

```python
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

pca = PCA(n_components=2)

pca.fit(X)

print(pca.explained_variance_ratio_)
# [0.9924... 0.0075...]

print(pca.singular_values_)
# [6.30061... 0.54980...]
```

**Como escolher o `n_componentes`**:  
Plotando a variância acumulada explicada em função do número de componentes e identificando o "joelho" da curva, escolhendo o menor *k*.
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Número de componentes')
plt.ylabel('Variância acumulada explicada')
plt.show()
```

**Como escolher o `svd_solver`**:  
 - Pequenos datasets: `full` ou `auto`.
 - Grandes datasets densos: `randomized`.
 - Dados Esparços: `arcpack`.
 - Melhor performance de execução: `randomized`.

**`GridSearchCV` para `PCA`**:
```python
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_components': [0.90, 0.95, 0.99],           # Em frações de variância
    'svd_solver': ['auto', 'full', 'randomized']
}
pca = PCA()
grid = GridSearchCV(pca, param_grid, cv=5, scoring='explained_variance')
grid.fit(X)
print("Melhor combo:", grid.best_params_)
```

**Casos de Uso**:
 - Quando há muitas variáveis correlacionadas, o PCA concentra a maior parte da informação nos primeiros componentes, reduzindo redundâncias.
 - Diminuir o número de recursos pode ajudar a simplificar o modelo e evitar ajustar ruídos, especialmente em casos de poucos dados relativos ao número de variáveis.
 - Para explorar e visualizar dados de alta dimensão em 2D ou 3D, mantendo o máximo de variância possível nas projeções iniciais.
 - Em pipelines de machine learning, reduzir dimensionalidade pode acelerar algoritmos posteriores (p.ex., SVM, redes neurais) e melhorar estabilidade numérica.

**Pontos de Atenção**:
 - PCA captura apenas relações lineares; se a estrutura dos dados for não linear, considere métodos como t-SNE ou UMAP.
 - É recomendável padronizar (z-score) as variáveis antes de aplicar PCA caso tenham escalas muito diferentes.
 - Componentes principais são combinações lineares de variáveis originais e podem ser menos interpretáveis em termos de significado físico, exigindo análise cuidadosa dos loadings.

## 3.2 TruncatedSVD - `sklearn.decomposition.TruncatedSVD`: 
`TruncatedSVD` é uma técnica de redução de dimensionalidade que aplica **Decomposição de Valores Singulares** de forma truncada, preservando apenas os maiores valores singulares para aproximar a matriz original com o menor posto. Ao contrário do `PCA`, o `TruncatedSVD` não centraliza os dados, o que o torna mais eficiente em matrizes esparsas. 

**Parâmetros Relevantes**:
 - `n_components`: número (inteiro) de componentes singulares a manter; controla a redução de dimensionalidade . Valor padrão: `2`.
 - `algorithm`: Estratégia para computar a SVD (`arpack`, `randomized`) permitindo balanço entre precisão e performance. Valor padrão: `randomized`.
 - `n_iter`: Número de iterações (inteiro) para um SVD randomizado resolver. Valor padrão: `5`.

```python
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import numpy as np

np.random.seed(0)

X_dense = np.random.rand(100, 100)

X_dense[:, 2 * np.arange(50)] = 0

X = csr_matrix(X_dense)

svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)

svd.fit(X)

print(svd.explained_variance_ratio_)
# [0.0157... 0.0512... 0.0499... 0.0479... 0.0453...]

print(svd.explained_variance_ratio_.sum())
# 0.2102...

print(svd.singular_values_)
# [35.2410...  4.5981...   4.5420...  4.4486...  4.3288...]
```

**Casos de Uso**:
 - O TruncatedSVD é a técnica subjacente ao Latent Semantic Analysis (LSA), usado para reduzir dimensões de matrizes de contagem de termos ou TF–IDF em tarefas de recuperação de informação e análise de texto.
 - Para datasets com milhares de variáveis e amostras, o algoritmo randomized acelera consideravelmente a decomposição, economizando tempo e memória em ambientes de produção ou pesquisa exploratória.
 - Se os dados não forem esparsos e puderem ser centralizados sem estourar memória, o PCA pode ser preferível por oferecer componentes interpretáveis em termos de variância total centrada.

## 3.3 t-SNE - `sklearn.manifold.TSNE`: 
`t-SNE` (t-distributed Stochastic Neighbor Embedding) é uma técnica não supervisionada de redução de dimensionalidade não linear para exploração de dados e visualização de dados de alta dimensão. A redução não linear da dimensionalidade significa que o algoritmo nos permite separar dados que não podem ser separados por uma linha reta. Devido à natureza não convexa de sua função de custo e à complexidade quadrática em número de amostras, o t-SNE costuma ser empregado em conjuntos de tamanho moderado (tipicamente até algumas dezenas de milhares de pontos) e **não é recomendado para pré-processamento de features em modelos de produção**, mas sim para exploração e apresentação de clusters e estruturas latentes nos dados. 

**Parâmetros Relevantes**:
 - `n_components`: número (inteiro) de dimensões de saída. Valor padrão: `2`.
 - `perplexity`:  Controla o número efetivo de vizinhos (float), tipicamente entre 5 e 50; afeta a granularidade do embutimento. Valor padrão: `30.0`.
 - `learning_rate`: passo do gradiente, deve ser da ordem de *O(n)* ou ajustado empiricamente; valores muito baixos ou muito altos prejudicam a convergência. Valor padrão: `auto`.
 - `n_iter`: número de iterações de otimização; recomenda-se ao menos 250 iterações de “queima” (early exaggeration) e 750 de ajuste fino. Valor padrão: `1000`.

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)
X_embedded = tsne.fit_transform(X)

plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, s=5)
plt.title("t-SNE embedding")
plt.xlabel("Dimensão 1")
plt.ylabel("Dimensão 2")
plt.show()
```

**Casos de Uso**:
 - Em uma visualização exploratória, revelar clusters e manifolds em dados de visão computacional, genômica, NLP ou outros domínios de alta dimensão.
 - Para validar features, verificando se embeddings (e.g., Word2Vec, autoencoders) capturam grupos semânticos ou estruturas latentes antes de modelos supervisados.
 - Para demonstrações e apresentações, gerando gráficos 2D/3D intuitivos que destacam seperações de classes ou agrupamentos naturais.

## 3.4 FeatureAgglomeration - `sklearn.manifold.FeatureAgglomeration`: 
`FeatureAgglomeration` é uma técnica de redução de dimensionalidade que aplica **clustering hierárquico** às features de um conjunto de dados, agrupando características que se comportam de forma semelhante em clusters, e agrega os valores de cada clusters por meio de uma função de pooling (por padrão, a média). 

**Parâmetros Relevantes**:
 - `n_clusters`: número de clusters finais de features (ou usar `distance_threshold` para determinar automaticamente). Valor Padrão: `2`. 
 - `linkage`: critério de fusão de clusters (“ward”, “complete”, “average”, “single”). Valor Padrão: `ward`. 
 - `metric`: métrica de distância entre features: “euclidean”, “manhattan”, “cosine” etc. Valor Padrão: `euclidean`. 

```python
import numpy as np
from sklearn import datasets, cluster

digits = datasets.load_digits()

images = digits.images

X = np.reshape(images, (len(images), -1))

agglo = cluster.FeatureAgglomeration(n_clusters=32)

agglo.fit(X)

X_reduced = agglo.transform(X)

X_reduced.shape
# (1797, 32)
```

**Escolhendo os melhores parametros**:
```python
import numpy as np
from sklearn.cluster import FeatureAgglomeration
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

# Definindo hiperparâmetros
param_grid = {
    'n_clusters': [2, 5, 10, 20],
    'linkage': ['ward', 'complete', 'average'],
    'metric': ['euclidean', 'manhattan', 'cosine'],
}

results = []
for params in ParameterGrid(param_grid):
    # Ajuste e transforma
    agglo = FeatureAgglomeration(
        n_clusters=params['n_clusters'],
        linkage=params['linkage'],
        metric=params['metric']
    )
    X_tr = agglo.fit_transform(X_train)
    X_te = agglo.transform(X_test)
    
    # Treinando classificador
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_tr, y_train)
    
    # Avaliando teste
    score = accuracy_score(y_test, clf.predict(X_te))
    results.append({**params, 'accuracy': score})

# Selecionando a melhor configuração
best = max(results, key=lambda x: x['accuracy'])
print("Melhor configuração:", best)
```

**Casos de Uso**:
 - Datasets com alta correlações de features, agrupando variáveis redundantes, reduzindo multicolinearidade e simplificando modelos.
 - Dados estruturados espacialmente, como imagens, sinais, utilizando a conectividade para agrupar pixels ou sensores próximos.
 - Em pré-processamento de pipelines para diminuir número de features para acelerar algorítimos custosos, como SVM ou redes neurais, sem perder padrões importantes.

# 4. Model Selection - `sklearn.model_selection`:

## 4.1 `train_test_split`
`train_test_split` divide arrays, matrizes ou dataframes em amostras aleatórias de train e teste. Utilizado em estimadores supervisionados.  

**Parâmetros Relevantes**:
 - `*arrays`: Arrays de input. Sendo lists, Arrays Numpy, Matrizes Scipy-Sparce ou Dataframes Pandas 
 - `test_size`: Porcentagem da amostra de testes, variando de `0.00` a `1.00`. Valor Padrão: `0.25` 
 - `train_size`: Porcentagem da amostra de treino, variando de `0.00` a `1.00`. Valor Padrão: complementar ao `test_size`
 - `random_state`: Controla o embaralhamento dos dados, permitindo reprodutibilidade
 - `shuffle`: Embaralhar ou não os dados antes de dividi-los. Valor Padrão: `True`
```python
import numpy as np
from sklearn.model_selection import train_test_split

X, y = np.arange(10).reshape((5, 2)), range(5) # Array de amostragem

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.33, 
    random_state=42
)
```
**Casos de Uso**:
 - Sempre que você estiver trabalhando com dados reais e precisa fazer a separação das features de treino e deste de modelos supervisionados.

## 4.2 `cross_val_score`:

## 4.3 `cross_val_predict`:

# 5. Model Metrics - `sklearn.metrics`:
Métricas dos modelos.

## 5.1 `accuracy_score`:

## 5.2 `confusion_matrix`:

## 5.3 `classification_report`:

## 5.4 `mean_squared_error`:

## 5.5 `r2_score`:

## 5.6 `roc_auc_score`:

## 5.7 `f1_score`:

## 5.8 `precision_score`:

## 5.9 `recall_score`:

# 6. `Pipeline`:
`Pipeline` permite que você aplique sequêncialmente uma lista de transformadores para pre-processar os dados. `Pipeline` pode ser usado como um estimador, evitando vazamento de dados entre as porções de teste e de treinamento.    

**Parâmetros Relevantes**:
 - `steps`: Uma lista de tuplas contendo uma string com o nome do passo, um estimador, podendo até conter as features.

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

numeric_features = ['age', 'income']
categorical_features = ['gender', 'city']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

pipeline = Pipeline(steps=[
    ('preproc', preprocessor),
    ('pca', PCA(n_components=3)),
    ('clf', LogisticRegression(random_state=0))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

y_prob = pipeline.predict_proba(X_test)[:, 1]
```

# 7. Classification Models

# 8. Regression Models

# 9. Clustering Models

