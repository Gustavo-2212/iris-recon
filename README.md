# Projeto de Aprendizado de Máquina com Redes Multicamadas 🧠
---
### Autor
- **Nome:** Gustavo Alves de Oliveira
- **Matrícula:** 12311ECP026
- **Disciplina:** Aprendizagem de Máquina
- **Faculdade:** Universidade Federal de Uberlândia


🤖📊🔍

> Este projeto faz uso da base de dados [iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) para aprender a classificar flores em três tipos, sendo elas **Iris Setosa**, **Iris Versicolor** e **Iris Virginica**.\
As flores possuem quatro características principais que as diferenciam, que são: **comprimento da sépala**, **largura da sépala**, **comprimento da pétala** e **largura da pétala**.\
Foram utilizadas estratégias de otimização, como o cálculo do **momento** para evitar oscilações no modelo, ou seja, a função de erro tender para o mínimo global de modo mais veloz; usamos também a estratégia de **validação**, sendo uma possível ponto de parada do treinamento, ajudando a ganhar generalização no modelo e evitando o **overtraining**; **normalização** dos dados de entrada, para valores entre 1.0 e -1.0.\



### Redes Neurais Perceptrons de Multicamadas (MLP)

> As Redes Neurais Perceptrons de Multicamadas (MLPs) são uma classe de redes neurais artificiais que consistem em uma rede de nós (neurônios) organizados em camadas. As MLPs são compostas por uma camada de entrada, uma ou mais camadas ocultas e uma camada de saída.

### Funcionamento das MLPs:

1. **Camada de Entrada:**
   - Os nós na camada de entrada representam as características (features) dos dados de entrada. Cada nó nesta camada corresponde a uma característica específica.

2. **Camadas Ocultas:**
   - As camadas ocultas são compostas por neurônios que processam os dados de entrada e aprendem a representação dos padrões nos dados.
   - Cada neurônio em uma camada oculta recebe entradas das camadas anteriores, realiza uma combinação linear das entradas ponderadas por pesos, aplica uma função de ativação não linear e passa o resultado para as camadas posteriores.
   - A presença de camadas ocultas permite que as MLPs aprendam representações mais complexas e não lineares dos dados.

3. **Camada de Saída:**
   - A camada de saída produz a saída final da rede neural. A estrutura desta camada depende do tipo de problema que está sendo abordado, como classificação (por exemplo, softmax para classificação multiclasse) ou regressão (um único neurônio de saída para prever um valor contínuo).

>![MLP](https://www.researchgate.net/publication/293013889/figure/fig1/AS:335717596188674@1457052720824/Figura-1-Exemplo-simplificado-de-uma-rede-neural-multicamadas-HAYKIN-2001-Figure-1.png)

### Idealização e Desenvolvimento:

>As MLPs não têm um único idealizador. O desenvolvimento das MLPs foi uma evolução das redes neurais perceptrons originais propostas por Frank Rosenblatt em 1957. As MLPs foram introduzidas para superar a limitação das redes perceptrons de uma única camada, que só podiam resolver problemas linearmente separáveis. A ideia de adicionar camadas ocultas e usar algoritmos de treinamento como o backpropagation permitiu que as MLPs aprendessem a representar relações mais complexas nos dados.

### Função Sigmóide Bipolar como Função de Ativação:

>A função sigmóide bipolar, também conhecida como função de ativação bipolar, é uma função matemática usada em redes neurais como uma função de ativação. Sua fórmula é:

$
    f(x) = \frac{2}{1 + e^{-x}} - 1
$

>Esta função mapeia os valores de entrada para o intervalo [-1, 1]. Ela é suave e diferenciável em todos os pontos, o que a torna adequada para o treinamento de redes neurais utilizando algoritmos de otimização baseados em gradiente, como o backpropagation.\
A função sigmóide bipolar tem sido historicamente utilizada como função de ativação nos neurônios das camadas ocultas das MLPs. No entanto, devido a problemas como o desaparecimento do gradiente durante o treinamento profundo e a propagação do gradiente muito lenta em camadas profundas, funções de ativação alternativas, como ReLU (Rectified Linear Unit) e suas variantes, tornaram-se mais populares em redes neurais profundas modernas.

> ![Sigmóide Bipolar](https://www.researchgate.net/publication/331087209/figure/fig4/AS:726046831820800@1550114462005/Figura-54-Funcion-de-Activacion-Sigmoide-Bipolar.jpg)

### Estrutura do Projeto

O projeto está estruturado da seguinte forma:

- **./data**: Possui os dados para **treino** (105 | 35 de cada flor), **teste** (15 | 5 de cada flor) e para **validação** (30 | 10 de cada flor);
- **./src**: Contém os códigos-fonte;
- **./tmp**: Irá conter os arquivos de relatórios dos pesos iniciais, pesos finais, valores do erro conforme a rede aprende e um arquivo **.html** que plota o gráfico da média quadrática.

### Dependências do Projeto

Para executar o projeto, é necessário instalar a seguinte biblioteca Python:

- Bokeh: Biblioteca para criar visualizações interativas em navegadores da web.

```bash
pip install bokeh
```

### Como Executar o Projeto

1. Clone o repositório do projeto.
2. Instale as dependências listadas acima.
3. Execute, no diretório raiz do projeto, os comandos a seguir:

```console
foo@bar:~$ make clean
foo@bar:~$ make exec
```

### Conclusões

> Podemos ver que para reconhecer a flor Iris Setosa e a diferenciar das outras o modelo é quase que perfeito, porém entre as flores Iris Versicolor e Iris Virginica, o modelo não conseguiu apresentar uma alta taxa de "certeza" para todos os dados de teste (que são apresentados no console).\
Foi testado a rede neural com mais neurônios na camada interna, porém os resultados não apresentaram uma melhora significativa, ainda que o modelo demorasse mais para ajustar os pesos. Não foi testado outros valores para a taxa de momento e a taxa de aprendizagem utilizada.


🚀🔍💡

---
