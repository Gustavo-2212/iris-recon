#include <stdint.h>
#include <stdio.h>
#include <math.h>


/*
============================================================= 
                    Diretrizes e Constantes
============================================================= */
/* Dados gerais da rede neural */
#define QUANTIDADE_ENTRADAS_TREINO      (105)
#define QUANTIDADE_ENTRADAS_TESTE       (15)
#define QUANTIDADE_ENTRADAS_VALIDACAO   (30)
#define NEURONIOS_CAMADA_INTERNA        (6)
#define NEURONIOS_CAMADA_SAIDA          (3)
#define NUMERO_VARIAVEIS                (4)

/* Dados para a normalização dos dados de entrada */
#define X1_MAX (7.9)
#define X1_MIN (4.3)

#define X2_MAX (4.4)
#define X2_MIN (2.0)

#define X3_MAX (6.9)
#define X3_MIN (1.0)

#define X4_MAX (2.5)
#define X4_MIN (0.1)

#define Y_MAX   (1.0)
#define Y_MIN   (-1.0)


/*
============================================================= 
                    Variáveis Globais
============================================================= */
/* Variáveis de entradas usadas para treino da rede 
   neural. Serão montadas no arquivo "prepara_data.c" */
double entradas_treino[QUANTIDADE_ENTRADAS_TREINO][NUMERO_VARIAVEIS];

/* Variável de entrada usada para o teste da rede. */
double entradas_teste[QUANTIDADE_ENTRADAS_TESTE][NUMERO_VARIAVEIS];

/* Variável de entrada usada para a validação da rede. */
double entradas_validacao[QUANTIDADE_ENTRADAS_VALIDACAO][NUMERO_VARIAVEIS];

/* Declaração da variável que armazenará os pesos internos
   da rede neural. */
double w_int[NEURONIOS_CAMADA_INTERNA][NUMERO_VARIAVEIS];
double b_int[NEURONIOS_CAMADA_INTERNA];

/* Declaração da variável que armazenará os pesos da cama-
   da de saída da rede neural. */
double w[NEURONIOS_CAMADA_SAIDA][NEURONIOS_CAMADA_INTERNA];
double b[NEURONIOS_CAMADA_SAIDA];


/* Saída da rede neural sob os dados de teste */
double saida_teste[QUANTIDADE_ENTRADAS_TESTE][NEURONIOS_CAMADA_SAIDA];

/*
============================================================= 
                    Protótipo das Funções
============================================================= */
void montar_entradas(char *nome_arq, double entradas[][NUMERO_VARIAVEIS], int linhas, int colunas);
void visualizar_matriz(double matriz[][NEURONIOS_CAMADA_SAIDA], int linhas, int colunas);


/*
============================================================= 
                        Macros
============================================================= */
#define SIGMOIDE(x) ((2.0 / (1 + exp(-x))) - 1)
#define NORMALIZA(x, X_MAX, X_MIN) ( ((x - X_MIN) / (X_MAX - X_MIN)) * (Y_MAX - Y_MIN) + Y_MIN )



/*
============================================================= 
                        Saída Interessante
      ERRO_MINIMO: 0.103
      TAXA_APRENDIZAGEM: 0.003
      BETA: 0.2
      CICLOS_MAX: 1000000
      CICLOS_PARA_VALIDACAO: 100000

      PESOS INICIAIS:
         W:
            0.467000 -0.205000 -0.343000 -0.183000 0.148000 0.088000 
            -0.245000 -0.209000 -0.120000 -0.210000 0.075000 -0.106000 
            -0.258000 -0.494000 -0.163000 -0.018000 -0.389000 -0.208000 

         B:
            0.417000 -0.301000 -0.082000 

         W_int:
            -0.149000 -0.259000 0.497000 -0.147000 
            0.100000 0.106000 0.203000 -0.389000 
            0.432000 -0.487000 0.409000 0.166000 
            0.286000 0.388000 -0.396000 0.454000 
            0.126000 0.174000 -0.200000 -0.358000 
            -0.288000 0.326000 -0.222000 0.444000 

         B_int:
            -0.421000 0.183000 0.112000 0.084000 0.238000 0.415000 
============================================================= */
/* Fim do treinamento!
   Ciclos: 1000000

   ------------------------------------------------------------------
   |       Teste em 15 entradas da base de dados (5 de cada flor)   |
   ------------------------------------------------------------------
   [+] Iris-Setosa

           1.000000 -0.997485 -1.000000 


           1.000000 -0.997503 -1.000000 


           1.000000 -0.997459 -1.000000 


           1.000000 -0.997470 -1.000000 


           1.000000 -0.997491 -1.000000 

   [+] Iris-Versicolor

           -1.000000 1.000000 1.000000 


           -1.000000 1.000000 1.000000 


           -1.000000 1.000000 1.000000 


           -1.000000 1.000000 1.000000 


           -1.000000 1.000000 1.000000 
 
   [+] Iris-Virginica

           -1.000000 -1.000000 0.999999 


           -0.999999 -0.999980 0.998148 


           -1.000000 -1.000000 1.000000 


           -1.000000 -0.983465 0.999996 


           -1.000000 -1.000000 0.999211 


   ------------------------------------------------------------------
   |                       Fim dos Testes!                          |
   ------------------------------------------------------------------
*/
