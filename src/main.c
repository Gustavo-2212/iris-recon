#include "data.h"
#include <time.h>
#include <stdbool.h>
#include <stdlib.h>

/*
============================================================= 
                    Diretivas e Constantes
============================================================= */
#define CICLOS_MAX          (1000000)
#define TAXA_APRENDIZAGEM   (0.003)
#define ERRO_MINIMO         (0.105)
#define CICLOS_VALIDACAO    (CICLOS_MAX / (QUANTIDADE_ENTRADAS_VALIDACAO / NEURONIOS_CAMADA_SAIDA))   /* Porção de ciclos máximo para apresentar entrada validação */
#define BETA                (0.2) /* Taxa de consideração para o uso do momento no cálculo, valor entre 0 < BETA <= 1 */

double saidas[NEURONIOS_CAMADA_SAIDA][NEURONIOS_CAMADA_SAIDA] = { {1.0, -1.0, -1.0}, {-1.0, 1.0, -1.0}, {-1.0 ,-1.0, 1.0} };

/*
============================================================= 
                    Definição de Estruturas
============================================================= */
typedef struct erro_s {
    double *Erros;
    int ciclos;
} erro_t;


/*
============================================================= 
                    Protótipo das Funções
============================================================= */
struct erro_s treina_MPL(void);
void testa_MPL(void);
void salvar_pesos(const char nome_arq[20], double w[NEURONIOS_CAMADA_SAIDA][NEURONIOS_CAMADA_INTERNA], double b[NEURONIOS_CAMADA_SAIDA],
                    double w_int[NEURONIOS_CAMADA_INTERNA][NUMERO_VARIAVEIS], double b_int[NEURONIOS_CAMADA_INTERNA]);
void salvar_erros(const char nome_arq[], double erros[], int ciclos);
double calcula_erro(double Y[], int saida);
double *calcula_saida(double entrada[]);
bool testa_validacao(int i);


/*
============================================================= 
                    Função Main
============================================================= */
int main(void) {

    montar_entradas("./data/database_treino.txt", entradas_treino, QUANTIDADE_ENTRADAS_TREINO, NUMERO_VARIAVEIS);

    montar_entradas("./data/database_teste.txt", entradas_teste, QUANTIDADE_ENTRADAS_TESTE, NUMERO_VARIAVEIS);

    montar_entradas("./data/database_validacao.txt", entradas_validacao, QUANTIDADE_ENTRADAS_VALIDACAO, NUMERO_VARIAVEIS);

    erro_t rs = treina_MPL();
    testa_MPL();

    visualizar_matriz(saida_teste, QUANTIDADE_ENTRADAS_TESTE, NEURONIOS_CAMADA_SAIDA);

    salvar_erros("./tmp/erros.txt", rs.Erros, rs.ciclos);

    return 0;
}


/*
============================================================= 
                    Implementação das Funções
============================================================= */
erro_t treina_MPL(void) {
    srand(time(NULL));
    int ciclos = 0;

    /* Vetor para armazenar os erros conforme os ciclos se passam */
    double *Erros = (double *) calloc(CICLOS_MAX, sizeof(double));

    /* Matriz para cálculo do momento para otmização da rede neural */
    double momento[NEURONIOS_CAMADA_SAIDA][NEURONIOS_CAMADA_INTERNA];
    for(int i = 0; i < NEURONIOS_CAMADA_SAIDA; i++) {
        for(int j = 0; j < NEURONIOS_CAMADA_INTERNA; j++) {
            momento[i][j] = 0.0;
        }
    }

    /* Inicialização dos pesos dos neurônios da camada interna */
    for(int i = 0; i < NEURONIOS_CAMADA_INTERNA; i++) {
        for(int j = 0; j < NUMERO_VARIAVEIS; j++) {
            w_int[i][j] = (double) (rand() % 1000) / 1000 - 0.5;
        }
        b_int[i] = (double) (rand() % 1000) / 1000 - 0.5;
    }
    double delta_int[NEURONIOS_CAMADA_INTERNA][NEURONIOS_CAMADA_SAIDA];

    /* Inicialização dos pesos dos neurônios da camada de saída */
    for(int i = 0; i < NEURONIOS_CAMADA_SAIDA; i++) {
        for(int j = 0; j < NEURONIOS_CAMADA_INTERNA; j++) {
            w[i][j] = (double) (rand() % 1000) / 1000 - 0.5;
        }
        b[i] = (double) (rand() % 1000) / 1000 - 0.5;
    }

    /* Salvando os pesos iniciais */
    salvar_pesos("./tmp/pesos_iniciais.txt", w, b, w_int, b_int);

    /* Calculando o Erro inicial */
    double erro = 0;
    for(int i = 0; i < QUANTIDADE_ENTRADAS_TREINO; i++) {
        double *Yinit = calcula_saida(entradas_treino[i]);
        erro += calcula_erro(Yinit, (i / (QUANTIDADE_ENTRADAS_TREINO/3)));

        free(Yinit);
    }
    Erros[0] = (erro / (2 * QUANTIDADE_ENTRADAS_TREINO));


    /* Loop de treinamento */
    while(ciclos < CICLOS_MAX && erro > ERRO_MINIMO) {
        double E = 0;
        ciclos++;

        if ((ciclos % CICLOS_VALIDACAO) == 0 && testa_validacao(ciclos / CICLOS_VALIDACAO)) {
            break;
        }

        for(int flor = 0; flor < QUANTIDADE_ENTRADAS_TREINO; flor++) {
            double *entrada = entradas_treino[flor];
            double *saida = saidas[flor / (QUANTIDADE_ENTRADAS_TREINO / 3)];

            double Z[NEURONIOS_CAMADA_INTERNA];
            double Y[NEURONIOS_CAMADA_SAIDA];
            double soma;

            /* Fase de Feedfoward */
            /* Calculando a saída dos neurônios da camada interna */
            for(int neuronio = 0; neuronio < NEURONIOS_CAMADA_INTERNA; neuronio++) {
                soma = b_int[neuronio];
                for(int variavel = 0; variavel < NUMERO_VARIAVEIS; variavel++) {
                    soma += entrada[variavel] * w_int[neuronio][variavel];
                }
                Z[neuronio] = SIGMOIDE(soma);
            }

            /* Calcula a saída dos neurônios da camada de saída */
            for(int neuronio = 0; neuronio < NEURONIOS_CAMADA_SAIDA; neuronio++) {
                soma = b[neuronio];
                for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                    soma += Z[neuronio_int] * w[neuronio][neuronio_int];
                }
                Y[neuronio] = SIGMOIDE(soma);
            }
            /* 
                Até aqui foi gerado um vetor com as 3 saídas da rede neural. 
                Y[neuronio], 0 <= neuronio < 3
            */

           /* Cálculo da somatória do erro quadrático */
           E += calcula_erro(Y, (flor / (QUANTIDADE_ENTRADAS_TREINO / NEURONIOS_CAMADA_SAIDA)));

           /* Fase da Retropropagação */
           /* Cálculo da atualização dos pesos da camada dos neurônios de saída */
           double delta_w[NEURONIOS_CAMADA_SAIDA];
           for(int neuronio = 0; neuronio < NEURONIOS_CAMADA_SAIDA; neuronio++) {
            delta_w[neuronio] = (saida[neuronio] - Y[neuronio]) * 0.5 * (1 + Y[neuronio]) * (1 - Y[neuronio]);
           }

            double Delta_w[NEURONIOS_CAMADA_SAIDA][NEURONIOS_CAMADA_INTERNA];
            double Delta_b[NEURONIOS_CAMADA_SAIDA];

            for(int neuronio_out = 0; neuronio_out < NEURONIOS_CAMADA_SAIDA; neuronio_out++) {
                for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                    Delta_w[neuronio_out][neuronio_int] = TAXA_APRENDIZAGEM * delta_w[neuronio_out] * Z[neuronio_int];
                }
            }

            for(int neuronio_out = 0; neuronio_out < NEURONIOS_CAMADA_SAIDA; neuronio_out++) {
                Delta_b[neuronio_out] = TAXA_APRENDIZAGEM * delta_w[neuronio_out];
            }

            /* Cálculo para a atualização dos pesos dos neurônios da camada interna */
            double soma_delta_w = 0;
            for(int i = 0; i < NEURONIOS_CAMADA_SAIDA; i++) {
                soma_delta_w += delta_w[i];
            }

            for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                for(int neuronio_out = 0; neuronio_out < NEURONIOS_CAMADA_SAIDA; neuronio_out++) {
                    delta_int[neuronio_int][neuronio_out] = soma_delta_w * w[neuronio_out][neuronio_int] * 0.5 * (1 + Z[neuronio_int]) * (1 - Z[neuronio_int]);
                }
            }

            double soma_delta_int[NEURONIOS_CAMADA_INTERNA];
            for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                double soma = 0;
                for(int neuronio_out = 0; neuronio_out < NEURONIOS_CAMADA_SAIDA; neuronio_out++) {
                    soma += delta_int[neuronio_int][neuronio_out];
                }
                soma_delta_int[neuronio_int] = soma;
            }

            double Delta_w_int[NEURONIOS_CAMADA_INTERNA][NUMERO_VARIAVEIS];
            double Delta_b_int[NEURONIOS_CAMADA_INTERNA];

            for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                for(int variavel = 0; variavel < NUMERO_VARIAVEIS; variavel++) {
                    Delta_w_int[neuronio_int][variavel] = TAXA_APRENDIZAGEM * soma_delta_int[neuronio_int] * entrada[variavel];
                }
            }

            for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                Delta_b_int[neuronio_int] = TAXA_APRENDIZAGEM * soma_delta_int[neuronio_int];
            }


            /* Atualização dos pesos da camada de saída */
            for(int neuronio_out = 0; neuronio_out < NEURONIOS_CAMADA_SAIDA; neuronio_out++) {
                for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                    momento[neuronio_out][neuronio_int] = Delta_w[neuronio_out][neuronio_int] + BETA * momento[neuronio_out][neuronio_int];
                    w[neuronio_out][neuronio_int] += momento[neuronio_out][neuronio_int];
                }
            }

            for(int neuronio_out = 0; neuronio_out < NEURONIOS_CAMADA_SAIDA; neuronio_out++) {
                b[neuronio_out] += Delta_b[neuronio_out];
            }


            /* Atualização dos pesos da camada interna */
            for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                for(int variavel = 0; variavel < NUMERO_VARIAVEIS; variavel++) {
                    w_int[neuronio_int][variavel] += Delta_w_int[neuronio_int][variavel];
                }
            }

            for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                b_int[neuronio_int] += Delta_b_int[neuronio_int];
            }
        }

        erro = (E / (2 * QUANTIDADE_ENTRADAS_TREINO));
        Erros[ciclos] = erro;
    }

    printf("Fim do treinamento!\n");
    printf("Ciclos: %d\n", ciclos);
    salvar_pesos("./tmp/pesos_finais.txt", w, b, w_int, b_int);

    erro_t rs = {
        .Erros = Erros,
        .ciclos = ciclos,
    };

    return rs;
}

/*
        Função para fazer o cálculo do vetor de saída da rede neural com base em uma entrada e nos
        pesos calculados até o momento de chamada dessa função.
        Retorna o ponteiro de onde começa o vetor com os resultados dos 3 neurônios de saída
*/
double *calcula_saida(double entrada[]) {
    double Z[NEURONIOS_CAMADA_INTERNA];
    double *Y = (double*) malloc(sizeof(double) * NEURONIOS_CAMADA_SAIDA);
    double soma;

    for(int neuronio = 0; neuronio < NEURONIOS_CAMADA_INTERNA; neuronio++) {
        soma = b_int[neuronio];
        for(int variavel = 0; variavel < NUMERO_VARIAVEIS; variavel++) {
            soma += entrada[variavel] * w_int[neuronio][variavel];
        }
        Z[neuronio] = SIGMOIDE(soma);
    }

    /* Calcula a saída dos neurônios da camada de saída */
    for(int neuronio = 0; neuronio < NEURONIOS_CAMADA_SAIDA; neuronio++) {
        soma = b[neuronio];
        for(int saida_internos = 0; saida_internos < NEURONIOS_CAMADA_INTERNA; saida_internos++) {
            soma += Z[saida_internos] * w[neuronio][saida_internos];
        }
        Y[neuronio] = SIGMOIDE(soma);
    }

    return Y;
}

/*
        Calcula o erro médio quadrático, dado a resposta do modelo para uma determinada entrada e uma
        saída para ser comparada.
*/
double calcula_erro(double Y[], int saida) {
    double erro = 0;
    for(int i = 0; i < NEURONIOS_CAMADA_SAIDA; i++) {
        erro += pow(saidas[saida][i] - Y[i], 2);
    }

    return erro;
}

/*
        Cria um arquivo para salvar os pesos calculados até então.
*/
void salvar_pesos(const char nome_arq[20], double w[NEURONIOS_CAMADA_SAIDA][NEURONIOS_CAMADA_INTERNA], double b[NEURONIOS_CAMADA_SAIDA],
                    double w_int[NEURONIOS_CAMADA_INTERNA][NUMERO_VARIAVEIS], double b_int[NEURONIOS_CAMADA_INTERNA]) {
    FILE *arquivo = fopen(nome_arq, "a"); // Abre o arquivo para adicionar conteúdo (append)
    if (!arquivo) {
        printf("Erro ao abrir o arquivo pesos.txt.\n");
        return;
    }

    // Escreve a matriz1 no arquivo
    fprintf(arquivo, "W:\n");
    for (int i = 0; i < NEURONIOS_CAMADA_SAIDA; i++) {
        for (int j = 0; j < NEURONIOS_CAMADA_INTERNA; j++) {
            fprintf(arquivo, "%f ", w[i][j]);
        }
        fprintf(arquivo, "\n");
    }

    // Escreve o vetor1 no arquivo
    fprintf(arquivo, "\nB:\n");
    for (int i = 0; i < NEURONIOS_CAMADA_SAIDA; i++) {
        fprintf(arquivo, "%f ", b[i]);
    }
    fprintf(arquivo, "\n");

    // Escreve a matriz2 no arquivo
    fprintf(arquivo, "\nW_int:\n");
    for (int i = 0; i < NEURONIOS_CAMADA_INTERNA; i++) {
        for (int j = 0; j < NUMERO_VARIAVEIS; j++) {
            fprintf(arquivo, "%f ", w_int[i][j]);
        }
        fprintf(arquivo, "\n");
    }

    // Escreve o vetor2 no arquivo
    fprintf(arquivo, "\nB_int:\n");
    for (int i = 0; i < NEURONIOS_CAMADA_INTERNA; i++) {
        fprintf(arquivo, "%f ", b_int[i]);
    }
    fprintf(arquivo, "\n\n---------------------------------------------------------------------------------------------\n");

    fclose(arquivo); // Fecha o arquivo
}

/*
        Faz o teste da rede neural já treinada (pesos ajustados) para o conjunto de dados para teste.
*/
void testa_MPL(void) {
    double Z[NEURONIOS_CAMADA_INTERNA];
    double soma;

    for(int flor = 0; flor < QUANTIDADE_ENTRADAS_TESTE; flor++) {
        double *entrada = entradas_teste[flor];

        for(int neuronio = 0; neuronio < NEURONIOS_CAMADA_INTERNA; neuronio++) {
            soma = b_int[neuronio];
            for(int var = 0; var < NUMERO_VARIAVEIS; var++) {
                soma += entrada[var] * w_int[neuronio][var];
            }
            Z[neuronio] = SIGMOIDE(soma);
        }

        for(int neuronio = 0; neuronio < NEURONIOS_CAMADA_SAIDA; neuronio++) {
            soma = b[neuronio];
            for(int neuronio_int = 0; neuronio_int < NEURONIOS_CAMADA_INTERNA; neuronio_int++) {
                soma += Z[neuronio_int] * w[neuronio][neuronio_int];
            }
            saida_teste[flor][neuronio] = SIGMOIDE(soma);
        }
    }
}

/*
        Salva os erros em um arquivo para ser enviado ao arquivo "graphic.py" para
        que o gráfico do erro seja gerado.
*/
void salvar_erros(const char nome_arq[], double erros[], int ciclos) {
    FILE *arquivo = fopen(nome_arq, "w");
    if(!arquivo) {
        printf("Falha ao abrir o arquivo %s.\n", nome_arq);
        return;
    }

    fprintf(arquivo, "\t[+] Erros:\n");
    for(int i = 0; i < ciclos; i++) {
        fprintf(arquivo, "%f ", erros[i]);
    }

    fclose(arquivo);
}

/*
        Faz o teste dos pesos até então calculados, apresentando uma base de dados
        para validação (1 flor de cada é apresentada). Se o erro do modelo calculado
        sobre um das entradas dessa base de dados for satisfatório, paramos o treinamento.
*/
bool testa_validacao(int i) {
    double *entradas[] = {
        entradas_validacao[i],
        entradas_validacao[i + (QUANTIDADE_ENTRADAS_VALIDACAO / NEURONIOS_CAMADA_SAIDA)],
        entradas_validacao[i + 2 * (QUANTIDADE_ENTRADAS_VALIDACAO / NEURONIOS_CAMADA_SAIDA)]
    };

    double erro = 0;
    for(int amostra = 0; amostra < NEURONIOS_CAMADA_SAIDA; amostra++) {
        double *entrada = entradas[amostra];

        double *Y_saida = calcula_saida(entrada);

        erro += calcula_erro(Y_saida, amostra);

        free(Y_saida);
    }

    erro /= (2 * NEURONIOS_CAMADA_SAIDA);

    if (erro <= ERRO_MINIMO) {
        return true;
    }
    return false;
}