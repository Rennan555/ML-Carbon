PROJETO DE REDES NEURAIS PARA FIBRAS DE CARBONO - PETROBRÁS

1. Bibliotecas utilizadas:

1.1. TensorFlow (2.12.0v): Estruturação geral da rede
1.2. Numpy (1.23.5v): Operações matemáticas gerais
1.3. Sklearn (): Manipulação básica do dataset
1.4. Pickle (): Salvar acurárias dos modelos para comparação
1.5. Pandas (1.5.3v): Importação do dataset de formato .csv
1.6. OS (): Selecionar e verificar diretórios específicos
1.7. Mathplotlib (): Plotar gráficos de acurácia e loss

2. Propriedades do Dataset:

2.1. Contém 17.286 exemplos (m)
2.2. 2 fearures (f) principais - Campaign, Resistence e Ciclo
2.3. 3 targets (y) possíveis - Normal[1], Alerta[2], Anômalo[3]
2.4. Divisão de exemplos também em 15 ciclos (entra também como feature)

3. Modelos disponíveis:

3.1. Dense: RN apenas com otimizador
3.2. Norm Dense: RN com camada de normalização inicial e otimizador