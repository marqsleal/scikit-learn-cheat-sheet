# Machine Learning Cheat Sheet - Scikit-Learn
Cheat Sheet da biblioteca Scikit-Learn.

# `sklearn.preprocessing` - Data Pre-Processing:
Pré-processamento dos dados.

## `StandardScaler`
Padroniza features numéricas removendo a média e escalando para a unidade de variância. É recomendando quando a feature possui uma distribuição aproximadamente Gaussiana (normal)  


**Código**:
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X_train)
X_train_standardized = scaler.transform(X_train)
X_test_standardized = scaler.transform(X_test)
```

## `MinMaxScaler`
Escala as features numéricas para um range específico. É recomendando quando a feature não possui uma distribuição normal  

**Código**:
```python
from sklearn.preprocessing import Normalizer
scaler = Normalizer().fit(X_train)
X_train_normalized = scaler.transform(X_train)
X_test_normalized = scaler.transform(X_test)
```

## Bonus: Checando se os dados são uma distribuição normal ou não
**Histograma**:
```python
import matplotlib.pyplot as plt

plt.hist(data, bins=30)
plt.title('Histograma dos dados')
plt.xlabel('Valor')
plt.ylabel('Frequência')
plt.show()
```
Caso o gráfico se assemelhe a um sino (Gaussiana), a distribuição dos dados é normal.

**Testes Estatísticos de normalidade**:
```python
from scipy import stats

stat, p = stats.shapiro(data)
print(f'Shapiro–Wilk: estatística={stat:.4f}, p-valor={p:.4f}')
```
 - Recomendado para pequenas amostras
 - [Shapiro-Wilk: É rejeitado caso p seja maior que `0.05`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html)

```python
from scipy import stats

res = stats.anderson(data, dist='norm')
print(f'Anderson–Darling: estatística={res.statistic:.4f}')
print('Valores críticos e níveis de significância:')
for sl, cv in zip(res.significance_level, res.critical_values):
    print(f'  {sl}% → {cv:.3f}')
```
 - [Anderson–Darling: Compara p com valores críticos](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html)

```python
from scipy import stats

stat, p = stats.normaltest(data)
print(f"D’Agostino–Pearson: estatística={stat:.4f}, p-valor={p:.4f}")
```
 - [D’Agostino–Pearson: Teste omnibus que combina skewness e kurtosis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html)

## `Binarizer`
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

## `LabelEncoder`
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

## `OneHotEncoder`
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

## `PolynomialFeatures`
`PolynomialFeatures` é um pré-processador que gera uma nova matriz de características contendo todas as combinações polinomiais de variáveis de entrada até um grau especificado (`degree`).
**Parametros Relevantes**:
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

# `sklearn.imputer` - Data Imputers:
Imputação de dados.

## `SimpleImputer`
`SimpleImputer` imputa aos campos ausentes nas colunas os dados referentes a estratégia (`strategy`) passada na construção do objeto.
**Parametros Relevantes**:
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

## `KNNImputer`
`KNNImputer` implementa uma técnica de imputação multivariada que utiliza o valor médio (ou outro estatístico) dos *k* vizinhos mais próximos para preencher valores ausentes, preservando estruturas locais e relações entre características.

**Parametros Relevantes**:
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

#  Dimentionality Reduction:

## `sklearn.decomposition.PCA` - PCA

## `sklearn.decomposition.TruncatedSVD` - TruncatedSVD

## `sklearn.manifold.TSNE` - t-SNE

## `sklearn.manifold.FeatureAgglomeration` - FeatureAgglomeration

# `sklearn.model_selection` - Model Selection:

## `train_test_split`
`train_test_split` divide arrays, matrizes ou dataframes em amostras aleatórias de train e teste. Utilizado em estimadores supervisionados.
**Parametros Relevantes**:
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

## `cross_val_score`

## `cross_val_predict`

# `sklearn.metrics` - Model Metrics:
Métricas dos modelos.

## `accuracy_score`

## `confusion_matrix`

## `classification_report`

## `mean_squared_error`

## `r2_score`

## `roc_auc_score`

## `f1_score`

## `precision_score`

## `recall_score`



# `Pipeline`
`Pipeline` permite que você aplique sequêncialmente uma lista de transformadores para pre-processar os dados. `Pipeline` pode ser usado como um estimador, evitando vazamento de dados entre as porções de teste e de treinamento.
**Parametros Relevantes**:
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

# Classification Models

# Regression Models

# Clustering Models

