import subprocess
import os
import matplotlib.pyplot as plt
import re

# Mi MV maneja 4 hilos lógicos, evaluaremos hasta el doble
max_threads = 8 
threads_list = list(range(1, max_threads + 1))
tiempos = []

print("Iniciando pruebas de rendimiento intensivas...")

for t in threads_list:
    os.environ['OMP_NUM_THREADS'] = str(t)
    # COrremos el código del proyecto
    resultado = subprocess.run(['./project'], capture_output=True, text=True)
    
    # Buscamos la línea del tiempo total usando expresiones regulares
    match = re.search(r"Tiempo total de computo:\s+([0-9.]+)", resultado.stdout)
    if match:
        tiempo = float(match.group(1))
        tiempos.append(tiempo)
        print(f"Hilos: {t} | Tiempo: {tiempo:.4f} s")
    else:
        print(f"Error procesando con {t} hilos. Revisa la salida de tu C++.")
        tiempos.append(0)

# Calcular Speedup
t1 = tiempos[0]
speedups = [t1 / tn for tn in tiempos]

# Gráfica 1: Tiempo vs Hilos
plt.figure(figsize=(10, 5))
plt.plot(threads_list, tiempos, marker='o', color='b', linewidth=2)
plt.title('Tiempo de Ejecución vs. Número de Hilos')
plt.xlabel('Número de Hilos (OpenMP)')
plt.ylabel('Tiempo Total (Segundos)')
plt.xticks(threads_list)
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('grafica_tiempo.png')
plt.close()

# Gráfica 2: Speedup vs Hilos
plt.figure(figsize=(10, 5))
plt.plot(threads_list, speedups, marker='o', color='g', linewidth=2, label='Speedup Real')
plt.plot(threads_list, threads_list, linestyle='--', color='r', label='Speedup Ideal')
plt.title('Aceleración (Speedup) vs. Número de Hilos')
plt.xlabel('Número de Hilos (OpenMP)')
plt.ylabel('Speedup (T1 / Tn)')
plt.xticks(threads_list)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.savefig('grafica_speedup.png')
plt.close()

print("Gráficas generadas: 'grafica_tiempo.png' y 'grafica_speedup.png'")