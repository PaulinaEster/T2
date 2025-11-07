### Matrix Multiplication Strassen
O algoritmo de **Strassen** divide as matrizes em submatrizes e resolve as partes menores de forma recursiva, quando os resultados da multiplicação das submatrizes for recebido, então são aplicadas subtrações e adições para obter o valor esperado. Esse método foi projetado para ser assintoticamente mais eficiente, fazendo a redução de multiplicações de forma recursiva, utilizando divisão e conquista. Ele troca a multiplicação de matrizes por uma constante de adições/subtrações de matrizes, assim ao invés de realizar oito multiplicações recursivas de submatrizes de tamanho n/2 x n/2, o algoritmo de **Strassen** realiza apenas sete multiplicações recursivas.

## Run Serial Version:
2. Go to the directory of the desired application and perform the following command:
    ```
    make APPLICATION_NAME WORKLOAD=_WORKLOAD_VALUE
    ```
    `_WORKLOAD_VALUE` are:
        A, B, C, D, E, ... \ 

3. Command example:
    ```
    cd strassen
    make WORKLOAD=A DEBUG=ON TIMER=ON
    ./strassen.A.exe


## Run Parallel Version:
2. Go to the directory of the desired application and perform the following command:
    ```
    make APPLICATION_NAME WORKLOAD=_WORKLOAD_VALUE
    ```
    `_WORKLOAD_VALUE` are:
        A, B, C, D, E, ... \ 

3. Command example:
    ```
    cd strassen-mpi
    make WORKLOAD=A DEBUG=ON TIMER=ON
    ./strassen_mpi.A.exe
