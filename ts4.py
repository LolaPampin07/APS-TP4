# -*- coding: utf-8 -*-
# %%librerias + variables


import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from numpy.fft import fft, fftshift
from scipy.signal import windows

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

# %% Calculo las potencias para ver que machean

# SNR = 3dB
pot_ruido3 = a0**2 / (2*10**(SNR3/10))
print(f"Potencia del SNR = 3dB -> {pot_ruido3:.3f}")
ruido3 = np.random.normal(0, np.sqrt(pot_ruido3), N) # Vector
var_ruido3 = np.var(ruido3)
print(f"Potencia de ruido con SNR = 3dB -> {var_ruido3:.3f}")

x1_snr3 = s1 + ruido3  # Modelo de señal --> señal limpia + ruido

# SNR = 10dB
pot_ruido10 = a0**2 / (2*10**(SNR10/10))
print(f"Potencia del SNR = 10dB -> {pot_ruido10:.3f}")
ruido10 = np.random.normal(0, np.sqrt(pot_ruido10), N) # Vector
var_ruido10 = np.var(ruido10)
print(f"Potencia de ruido con SNR = 10dB -> {var_ruido10:.3f}")

x1_snr10 = s1 + ruido10  # Modelo de señal --> señal limpia + ruido

# %% CALCULO LAS DFT

S1 = (1/N)*fft(s1)
# modulo_S1 = np.abs(S1)**2

# SNR = 3dB
RUIDO3 = (1/N)*fft(ruido3)

# Calculo la FFT
X1_snr3 = (1/N)*fft(x1_snr3) # Multiplico por 1/N para calibrarlo --> llevar el piso de ruido a cero

# SNR = 10dB
RUIDO10 = (1/N)*fft(ruido10)

# Calculo la FFT
X1_snr10 = (1/N)*fft(x1_snr10) # Multiplico por 1/N para calibrarlo --> llevar el piso de ruido a cero

# %% GRAFICO
plt.figure()  # Tamaño de la figura (ancho, alto)

# Grafico X1 con SNR = 3db

plt.subplot(1,2,1)
plt.title("Densidades espectrales de potencia (PDS) con SNR = 3db")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.xlim([-fs/2, fs/2]) # En este caso fs = N, pero pongo fs para saber que va eso y no va siempre N
# plt.plot(ff, np.log10(np.abs(S1)**2) * 10, label = 'S1') # En este caso es un db de tension
# plt.plot(ff, np.log10(np.abs(R)**2) * 10, label = 'Ruido')
plt.plot(ff, np.log10(2*np.abs(X1_snr3)**2) * 10, label = 'X1 con SNR = 3dB')  # Densidad espectral de potencia
plt.legend()

plt.subplot(1,2,2)
plt.title("Densidades espectrales de potencia (PDS) con SNR = 10db")
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.xlim([0, fs/2]) # En este caso fs = N, pero pongo fs para saber que va eso y no va siempre N
# plt.plot(ff, np.log10(np.abs(S1)**2) * 10, label = 'S1') # En este caso es un db de tension
# plt.plot(ff, np.log10(np.abs(R)**2) * 10, label = 'Ruido')
plt.plot(ff, np.log10(2*np.abs(X1_snr10)**2) * 10, label = 'X1 con SNR = 10dB')  # Densidad espectral de potencia
plt.legend()

plt.show()

# En ruido es poco entonces se "tapa", no me juega en la suma
# Estar 250dB por debajo, es estar 25 ordenes de potencia por debajo
# Todo el piso de ruido es tapado por la energia, al ser esta tan tan grande (casi infinitamente mas grande), lo tapa al ruido.

# %% Vamos a hacer una funcion seno para poder pasarle matrices
k0 = N / 4
t = np.arange(N).reshape(-1,1) / fs # reshape para que las columnas sean tiempo
t_mat = np.tile(t, (1, realizaciones)) # (1000, 200)


# Repetir fr en filas (mismo valor de frecuencias por columna)
frecuencias = (k0 + fr) * df # en Hz
f_mat = np.tile(frecuencias, (N, 1))  # (1000, 200)


# Matriz de senoidales
s_mat = a0 * np.sin(2 * np.pi * f_mat * t_mat) # (1000, 200)

# RUIDO con SNR = 3dB
pot_ruido3 = a0**2 / (2 * 10**(SNR3 / 10))
ruido_mat3 = np.random.normal(0, np.sqrt(pot_ruido3), size = (N, realizaciones))  # (1000, 1)

x_mat3 = s_mat + ruido_mat3

# RUIDO con SNR = 10dB
pot_ruido10 = a0**2 / (2 * 10**(SNR10 / 10))
ruido_mat10 = np.random.normal(0, np.sqrt(pot_ruido10), size = (N, realizaciones))  # (1000, 1)

x_mat10 = s_mat + ruido_mat10

# Calculo la FFT normalizada a lo largo del eje del tiempo (filas)
X_mat3 = (1/N) * fft(x_mat3, axis=0)
X_mat10 = (1/N) * fft(x_mat10, axis=0)

plt.figure()

plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat3)**2) * 10, label = 'SNR = 3dB')  # Densidad espectral de potencia
plt.plot(ff, np.log10(2*np.abs(X_mat10)**2) * 10, label = 'SNR = 10dB')
plt.xlim([0, fs/2])

plt.show()


# %% Señales ventaneadas

# SNR = 3dB
x_vent_fla3 = ruido_mat3 * (windows.flattop(N).reshape(-1,1))
x_vent_BM3 = ruido_mat3 * (windows.blackman(N).reshape(-1,1))
x_vent_R3 = ruido_mat3 * (windows.boxcar(N).reshape(-1,1))
x_vent_H3 = ruido_mat3 * (windows.hamming(N).reshape(-1,1))

# Calculo la FFT normalizada a lo largo del eje del tiempo (filas)
X_mat_ft3 = (1/N) * fft(x_vent_fla3, axis=0)
X_mat_BM3 = (1/N) * fft(x_vent_BM3, axis=0)
X_mat_R3 = (1/N) * fft(x_vent_R3, axis=0)
X_mat_H3 = (1/N) * fft(x_vent_H3, axis=0)

# Graficos de la transformada de senales ventanadas con ruido
plt.figure()

plt.subplot(2,2,1)
plt.title('BLACKMAN (SNR = 3dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_BM3)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,2)
plt.title('RECTANGULAR (SNR = 3dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_R3)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,3)
plt.title('HAMMING (SNR = 3dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_H3)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,4)
plt.title('FLATOP (SNR = 3dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_ft3)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.tight_layout()
plt.show()

# SNR = 10dB
x_vent_fla10 = ruido_mat10 * (windows.flattop(N).reshape(-1,1))
x_vent_BM10 = ruido_mat10 * (windows.blackman(N).reshape(-1,1))
x_vent_R10 = ruido_mat10 * (windows.boxcar(N).reshape(-1,1))
x_vent_H10 = ruido_mat10 * (windows.hamming(N).reshape(-1,1))

# Calculo la FFT normalizada a lo largo del eje del tiempo (filas)
X_mat_ft10 = (1/N) * fft(x_vent_fla10, axis=0)
X_mat_BM10 = (1/N) * fft(x_vent_BM10, axis=0)
X_mat_R10 = (1/N) * fft(x_vent_R10, axis=0)
X_mat_H10 = (1/N) * fft(x_vent_H10, axis=0)

# Graficos de la transformada de senales ventanadas con ruido
plt.figure()

plt.subplot(2,2,1)
plt.title('BLACKMAN (SNR = 10dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_BM10)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,2)
plt.title('RECTANGULAR (SNR = 10dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_R10)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,3)
plt.title('HAMMING (SNR = 10dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_H10)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,4)
plt.title('FLATOP (SNR = 10dB)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_ft10)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.tight_layout()
plt.show()
# %% Estimador de energia

trans = 0.35
bins = 10

# SNR = 3dB
estimador_a_FT_3= 10*np.log10(2*(np.abs(X_mat_ft3[N//4,:])**2))
estimador_a_BM_3= 10*np.log10(2*(np.abs(X_mat_BM3[N//4,:])**2))
estimador_a_R_3= 10*np.log10(2*(np.abs(X_mat_R3[N//4,:])**2))
estimador_a_H_3= 10*np.log10(2*(np.abs(X_mat_H3[N//4,:])**2))

plt.figure()
plt.title("Histograma de la estimación de energía (SNR = 3dB)")
plt.hist(estimador_a_BM_3, label = 'Blackman', alpha = trans, bins = bins)
plt.hist(estimador_a_R_3,label = 'Rectangular', alpha = trans, bins = bins)
plt.hist(estimador_a_H_3,label = 'Hamming', alpha = trans, bins = bins)
plt.hist(estimador_a_FT_3,label = 'Flatop', alpha = trans, bins = bins)
plt.legend()
plt.show()


# SNR = 10dB
estimador_a_FT_10= 10*np.log10(2*(np.abs(X_mat_ft10[N//4,:])**2))
estimador_a_BM_10= 10*np.log10(2*(np.abs(X_mat_BM10[N//4,:])**2))
estimador_a_R_10= 10*np.log10(2*(np.abs(X_mat_R10[N//4,:])**2))
estimador_a_H_10= 10*np.log10(2*(np.abs(X_mat_H10[N//4,:])**2))

plt.figure()
plt.title("Histograma de la estimación de energía (SNR = 10dB)")
plt.hist(estimador_a_BM_10, label = 'Blackman', alpha = trans, bins = bins)
plt.hist(estimador_a_R_10,label = 'Rectangular', alpha = trans, bins = bins)
plt.hist(estimador_a_H_10,label = 'Hamming', alpha = trans, bins = bins)
plt.hist(estimador_a_FT_10,label = 'Flatop', alpha = trans, bins = bins)
plt.legend()
plt.show()

# %% Estimador de frecuencia               

plt.figure()
plt.hist(estimador_frec, bins=20, alpha=0.7)
plt.xlabel("Frecuencia estimada [Hz]")
plt.ylabel("Cantidad de ocurrencias")
plt.title("Histograma del estimador de frecuencia")
plt.show()


