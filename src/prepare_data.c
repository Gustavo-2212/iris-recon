#include "data.h"

double X_MAX[NUMERO_VARIAVEIS] = {X1_MAX, X2_MAX, X3_MAX, X4_MAX};
double X_MIN[NUMERO_VARIAVEIS] = {X1_MIN, X2_MIN, X3_MIN, X4_MIN};

void montar_entradas(char *nome_arq, double entradas[][NUMERO_VARIAVEIS], int linhas, int colunas) {
    FILE *arq = fopen(nome_arq, "r");

    if(!arq) {
        printf("Falha ao abrir %s. Verifique o caminho e a existência desse arquivo.\n", nome_arq);
        return;
    }
    
    double valor;
    for(int i = 0; i < linhas; i++) {
        for(int j = 0; j < colunas; j++) {
            fscanf(arq, "%lf", &valor);
            entradas[i][j] = NORMALIZA(valor, X_MAX[j], X_MIN[j]);
        }
    }

    fclose(arq);
}

void visualizar_matriz(double matriz[][NEURONIOS_CAMADA_SAIDA], int linhas, int colunas) {
    char *flores[] = { "Iris-Setosa\0", "Iris-Versicolor\0", "Iris-Virginica\0" };
    printf("\n------------------------------------------------------------------\n");
    printf("|\tTeste em 15 entradas da base de dados (5 de cada flor)\t |");
    printf("\n------------------------------------------------------------------\n");
    for(int i = 0; i < linhas; i++) {
        
        if(i % (QUANTIDADE_ENTRADAS_TESTE / 3) == 0)
            printf("[+] %s\n\n\t", flores[i / (QUANTIDADE_ENTRADAS_TESTE / 3)]);
        else
            printf("\n\t");

        for(int j = 0; j < colunas; j++) {
            printf("%lf ", matriz[i][j]);
        }
        printf("\n\n");
    }
    printf("\n------------------------------------------------------------------\n");
    printf("|\t\t\tFim dos Testes!\t\t\t\t |");
    printf("\n------------------------------------------------------------------\n");
}