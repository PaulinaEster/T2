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
numnodes=("4" "8" "16")
#numnodes=("4" "2")
# classes que serão executadas
classes=("A" "B" "C" "D" "E") 
# classes=("A" "B" "C") 

# qual o diretorio dos benchmarks que serão executados e quais os benchmarks.
mpirun="strassen_mpi"
cpprun="strassen"

HOME=$(pwd)

# Arquivo que terá a saida do resultado.
sequencial=$HOME"/resultados/sequencial.log"
# Arquivo de log temporario
mpi=$HOME"/resultados/mpi.log"

# Limpa/cria arquivos de log
#> "$mpi"
> "$sequencial"
echo "---------- GERANDO EXECUTAVEIS DE $dir -----------"

echo "==== $count EXECUÇÃO SEQUENCIAL:"
for workload in "${classes[@]}"
do
    echo " ENTRADA:  $workload "
    cd $HOME/strassen
    make WORKLOAD=$workload
    "./strassen.$workload.exe" >> "$sequencial" 2>&1
    echo " ================ " >> "$sequencial"
    if [ $? -ne 0 ]; then
        echo "Erro na execução do SERIAL. Abortando." | tee -a "$sequencial"
        exit 1
    fi
done


for workload in "${classes[@]}"
do
    cd $HOME/strassen-mpi
    make WORKLOAD=$workload
    for nodes in "${numnodes[@]}"
    do 
        echo "==== $count EXECUÇÃO PARALELA: $workload NODES $nodes"
        (mpirun -np $nodes -oversubscribe $mpirun.$workload.exe) >> "$mpi" 
        if [ $? -ne 0 ]; then
            echo "Erro na execução do mpirun -np $nodes '$workload' em $dir. Abortando." | tee -a "$mpi"
            exit 1
        fi 
    done
done
