#!/bin/bash

# Asegurarse de usar el máximo de hilos
export OMP_NUM_THREADS=$(nproc)
echo "Evaluando con $OMP_NUM_THREADS hilos físicos/lógicos..."
echo "--------------------------------------------------------"

# Lista de configuraciones a probar
configs=("static" "dynamic,1" "dynamic,10" "dynamic,64" "dynamic,256" "guided,1" "guided,10" "guided,64" "guided,256")

for conf in "${configs[@]}"; do
    export OMP_SCHEDULE="$conf"
    echo "Prueba con OMP_SCHEDULE=$conf"
    
    # Ejecutamos el programa, pero filtramos la salida para ver solo la Tarea A
    ./project | grep "Tarea A"
    echo "---"
done