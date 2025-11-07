#!/bin/bash

# Declaração associativa: cada chave é o diretório, o valor é um array de comandos make (alvo + parâmetros)
# !!ATENÇÃO!!
# Apenas execute esse cara se você já estar em maquinas alocadas, caso não tenha alocado recursos, segue comandos uteis:
# ladinfo -- para listar as maquinas que podem ser alocadas, visaulizar quantas maquinas estão livres
# PARA ALOCAR MAQUINAS: 
# -n numero de maquinas
# -t tempo que será alocado
# -e exclusivo ou -s shared
# ladalloc -n 4 -t 60 -e

declare -A mpiruns

# BENCHMARKS QUE SERÃO EXECUTADOS EM CADA PASTA

# quantidade de nodos que será utilizada na execução
numnodes=("16" "32" "64" "96")
#numnodes=("4" "2")
# classes que serão executadas
# classes=("S" "W" "A" "C")
classes=("128" "256" "512" "1024")

# qual o diretorio dos benchmarks que serão executados e quais os benchmarks.
mpirun="strassen_mpi"
cpprun="strassen"

# Arquivo que terá a saida do resultado.
resultado="./resultados/resultado.log"
# Arquivo de log temporario
logfile="./resultados/execucao.log"

# Limpa/cria arquivos de log
> "$logfile"
> "$resultado"

echo "---------- GERANDO EXECUTAVEIS DE $dir -----------"
make 
gcc strassen.c -o strassen

for workload in "${classes[@]}"
do
    for nodes in "${numnodes[@]}"
    do 
        echo "==== $count EXECUÇÃO PARALELA: $workload NODES $nodes"
        (mpirun -np $nodes --oversubscribe $mpirun $workload) >> "$logfile" 
        if [ $? -ne 0 ]; then
            echo "Erro na execução do mpirun -np $nodes '$workload' em $dir. Abortando." | tee -a "$logfile"
            exit 1
        fi 
        
        
    done
    echo "==== $count EXECUÇÃO SEQUENCIAL: $workload "
    echo " ENTRADA:  $workload " | tee -a >> "$resultado" 
    (./strassen $workload) >> "$resultado" 2>&1
    echo "================ " >> "$resultado"
    if [ $? -ne 0 ]; then
        echo "Erro na execução do SERIAL. Abortando." | tee -a "$resultado"
        exit 1
    fi
done
