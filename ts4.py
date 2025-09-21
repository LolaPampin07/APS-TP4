# -*- coding: utf-8 -*-
# %%librerias + variables


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft

# declaracion de varibles

N = 1000 #cantidad de muestras
fs = N #frecuencia de muestreo
df = fs/N # Resolucion temporal
a0 = 2 #amplitud
realizaciones = 200 # Sirve para parametrizar la cantidad de realizaciones de sampling ->muestras que vamos a tomar de la frecuencia
omega_0 = np.pi / 2 # fs/4 -> mitad de banda digital
fr = np.random.uniform(-2,2) #variable aleatoria de distribucion normal para la frecuencia
omega_1 = omega_0 + fr * 2 * np.pi / N
nn = np.arange(N) # Vector dimensional de muestras
ff = np.arange(N) # Vector en frecuencia al escalar las muestras por la resolucion espectral

#signal to noise ratio en dB segun pide la consigna
SNR3=3
SNR10 = 10


# %% FUNCION SENOIDAL
def mi_funcion_sen(frecuencia, nn, amplitud = 1, offset = 0, fase = 0, fs = 2):   

    N = np.arange(nn)
    
    t = N / fs

    x = amplitud * np.sin(2 * np.pi * frecuencia * t + fase) + offset

    return t, x

t1,s1 = mi_funcion_sen(frecuencia = omega_1, nn = N, fs = fs, amplitud = a0) # Funcion senoidal con frecuencia aleatoria

# %%RUIDO

pot_ruido3 = a0*2 / (2*10*(SNR3/10))
print(f"Potencia del SNR 3dB -> {pot_ruido3:.3f}")
ruido3 = np.random.normal(0, np.sqrt(pot_ruido3), N) # Vector
var_ruido3 = np.var(ruido3)
print(f"Potencia de ruido 3dB -> {var_ruido3:.3f}")

pot_ruido10 = a0*2 / (2*10*(SNR10/10))
print(f"Potencia del SNR 10dB-> {pot_ruido10:.3f}")
ruido10 = np.random.normal(0, np.sqrt(pot_ruido10), N) # Vector
var_ruido10 = np.var(ruido10)
print(f"Potencia de ruido 10dB -> {var_ruido10:.3f}")


# Modelo de señal --> señal limpia + ruido
x1 = s1 + ruido3  
x2 = s1 + ruido10  

plt.figure()
plt.plot(x1,'x',label='senal + 3dB ruido')
plt.plot(x2,'o',label='senal + 10dB ruido')
plt.legend()
plt.show()

# %%FFT

X1 = (1/N)*fft(x1) # Multiplico por 1/N para calibrarlo --> llevar el piso de ruido a cero

X2 = (1/N)*fft(x2)# Multiplico por 1/N para calibrarlo --> llevar el piso de ruido a cero


# GRAFICO
plt.figure(figsize=(20,20))

# Grafico X1 en db

plt.title("Densidades espectrales de potencia (PDS) en db")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.xlim([0, fs/2]) # En este caso fs = N, pero pongo fs para saber que va eso y no va siempre N
# plt.plot(ff, np.log10(np.abs(S1)**2) * 10, label = 'S1') # En este caso es un db de tension
# plt.plot(ff, np.log10(np.abs(R)**2) * 10, label = 'Ruido')
plt.plot(ff, np.log10(2*np.abs(X1)**2 * 10), label = 'X1')  # Densidad espectral de potencia
plt.plot(ff, np.log10(2*np.abs(X2)**2* 10), label = 'X2')
plt.legend()
plt.show()

# En ruido es poco entonces se "tapa", no me juega en la suma
# Estar 250dB por debajo, es estar 25 ordenes de potencia por debajo
# Todo el piso de ruido es tapado por la energia, al ser esta tan tan grande (casi infinitamente mas grande), lo tapa al ruido.