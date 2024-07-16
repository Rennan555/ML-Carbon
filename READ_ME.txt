PROJETO DE REDES NEURAIS PARA FIBRAS DE CARBONO - PETROBRÁS

1. Bibliotecas utilizadas:

1.1. TensorFlow (2.12.0v): Estruturação geral da rede
1.2. Numpy (1.23.5v): Operações matemáticas gerais
1.3. Pandas (1.5.3v): Importação do dataset de formato .csv

2. Propriedades do Dataset:

2.1. Contém 17.286 exemplos (m)
2.2. 2 fearures (f) principais - Campaign, Resistence e Ciclo
2.3. 3 targets (y) possíveis - Normal[1], Alerta[2], Anômalo[3]
2.4. Divisão de exemplos também em 15 ciclos (entra também como feature)