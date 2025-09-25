# %% Librerias + variables
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft
from scipy.signal import windows

# Declaracion de varibles

N = 1000 # Cantidad de muestras
fs = N # Frecuencia de muestreo
df = fs/N # Resolucion temporal
a0 = 2 # Amplitud
realizaciones = 200 # Sirve para parametrizar la cantidad de realizaciones de sampling ->muestras que vamos a tomar de la frecuencia
omega_0 = fs / 4 # fs/4 -> mitad de banda digital
fr = np.random.uniform(-2,2) # Variable aleatoria de distribucion normal para la frecuencia
omega_1 = omega_0 + fr * df
nn = np.arange(N) # Vector dimensional de muestras
ff = np.arange(N) # Vector en frecuencia al escalar las muestras por la resolucion espectral

# Signal to noise ratio en dB segun pide la consigna
SNR3=3
SNR10 = 10

# %% FUNCION SENOIDAL
def mi_funcion_sen(frecuencia, nn, amplitud = 1, offset = 0, fase = 0, fs = 2):   

    N = np.arange(nn)
    
    t = N / fs

    x = amplitud * np.sin(2 * np.pi * frecuencia * t + fase) + offset

    return t, x

t1,s1 = mi_funcion_sen(frecuencia = omega_1*df, nn = N, fs = fs, amplitud = a0) # Funcion senoidal con frecuencia aleatoria

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

# %% GRAFICO senal + ruido
plt.figure()  # Tamaño de la figura (ancho, alto)
plt.title("Densidades espectrales de potencia (PDS) con w= " + str(omega_1*df))
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.xlim([0, fs/2]) # En este caso fs = N, pero pongo fs para saber que va eso y no va siempre N

plt.plot(ff, np.log10(2*np.abs(X1_snr10)**2) * 10,color='b', label = 'X1 con SNR = 10dB')  # Densidad espectral de potencia
plt.plot(ff, np.log10(2*np.abs(X1_snr3)**2) * 10, color='y',label = 'X1 con SNR = 3dB')  # Densidad espectral de potencia
plt.legend()
plt.show()
# En ruido es poco entonces se "tapa", no me juega en la suma
# Estar 250dB por debajo, es estar 25 ordenes de potencia por debajo
# Todo el piso de ruido es tapado por la energia, al ser esta tan tan grande (casi infinitamente mas grande), lo tapa al ruido.

# %% Realizo las 200 realizaciones
k0 = omega_1
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

# Grafico señal + ruido
plt.figure()
plt.title('FFT señal + ruido')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat3)**2) * 10)  # Densidad espectral de potencia
plt.plot(ff, np.log10(2*np.abs(X_mat10)**2) * 10)
plt.xlim([0, fs/2])
plt.show()

# %%#Ventaneo y graficos de la senales con ruido

# SNR = 3dB
x_vent_fla3 = x_mat3 * (windows.flattop(N).reshape(-1,1))
x_vent_BM3 = x_mat3 * (windows.blackman(N).reshape(-1,1))
x_vent_R3 = x_mat3 * (windows.boxcar(N).reshape(-1,1))
x_vent_H3 = x_mat3 * (windows.hamming(N).reshape(-1,1))

# Calculo la FFT normalizada a lo largo del eje del tiempo (filas)
X_mat_ft3 = (1/N) * fft(x_vent_fla3, axis=0)
X_mat_BM3 = (1/N) * fft(x_vent_BM3, axis=0)
X_mat_R3 = (1/N) * fft(x_vent_R3, axis=0)
X_mat_H3 = (1/N) * fft(x_vent_H3, axis=0)

# Graficos de la transformada de senales ventanadas con ruido
plt.figure()
plt.suptitle("Señal con ruido 3dB ventaneada")

plt.subplot(2,2,1)
plt.title('BLACKMAN')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_BM3)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,2)
plt.title('RECTANGULAR')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_R3)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,3)
plt.title('HAMMING')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_H3)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,4)
plt.title('FLAT-TOP')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_ft3)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.tight_layout()
plt.show()

# SNR = 10dB
x_vent_fla10 = x_mat10 * (windows.flattop(N).reshape(-1,1))
x_vent_BM10 = x_mat10 * (windows.blackman(N).reshape(-1,1))
x_vent_R10 = x_mat10 * (windows.boxcar(N).reshape(-1,1))
x_vent_H10 = x_mat10 * (windows.hamming(N).reshape(-1,1))

# Calculo la FFT normalizada a lo largo del eje del tiempo (filas)
X_mat_ft10 = (1/N) * fft(x_vent_fla10, axis=0)
X_mat_BM10 = (1/N) * fft(x_vent_BM10, axis=0)
X_mat_R10 = (1/N) * fft(x_vent_R10, axis=0)
X_mat_H10 = (1/N) * fft(x_vent_H10, axis=0)

# Graficos de la transformada de senales ventanadas con ruido
plt.figure()
plt.suptitle("Señal con ruido 10dB ventaneada")

plt.subplot(2,2,1)
plt.title('BLACKMAN')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_BM10)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,2)
plt.title('RECTANGULAR')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_R10)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,3)
plt.title('HAMMING')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('PDS [db]')
plt.plot(ff, np.log10(2*np.abs(X_mat_H10)**2) * 10)  # Densidad espectral de potencia
plt.xlim([0, fs/2])

plt.subplot(2,2,4)
plt.title('FLATOP')
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

# SNR = 10dB
estimador_a_FT_10= 10*np.log10(2*(np.abs(X_mat_ft10[N//4,:])**2))
estimador_a_BM_10= 10*np.log10(2*(np.abs(X_mat_BM10[N//4,:])**2))
estimador_a_R_10= 10*np.log10(2*(np.abs(X_mat_R10[N//4,:])**2))
estimador_a_H_10= 10*np.log10(2*(np.abs(X_mat_H10[N//4,:])**2))


plt.figure()
plt.suptitle("Histograma de la estimación de energía")

plt.subplot(1,2,1)
plt.title("SNR = 3dB")
plt.hist(estimador_a_BM_3, label = 'Blackman', alpha = trans, bins = bins)
plt.hist(estimador_a_R_3,label = 'Rectangular', alpha = trans, bins = bins)
plt.hist(estimador_a_H_3,label = 'Hamming', alpha = trans, bins = bins)
plt.hist(estimador_a_FT_3,label = 'Flatop', alpha = trans, bins = bins)
plt.xlabel('PDS [db]')
plt.ylabel('#Cantidad de ocurrencias')
plt.legend()

plt.subplot(1,2,2)
plt.title("SNR = 10dB")
plt.hist(estimador_a_BM_10, label = 'Blackman', alpha = trans, bins = bins)
plt.hist(estimador_a_R_10,label = 'Rectangular', alpha = trans, bins = bins)
plt.hist(estimador_a_H_10,label = 'Hamming', alpha = trans, bins = bins)
plt.hist(estimador_a_FT_10,label = 'Flatop', alpha = trans, bins = bins)
plt.xlabel('PDS [db]')
plt.ylabel('#Cantidad de ocurrencias')
plt.legend()

plt.show()

# %% Estimador de frecuencia

# Defino rango de frecuencias (en Hz)
freqs = np.fft.fftfreq(N, 1/fs)  # Eje de frecuencias
freqs = freqs[:N//2] 

# SNR = 3 dB
mag_FT3 = np.abs(X_mat_ft3[:N//2, :])
idx_FT3 = np.argmax(mag_FT3, axis=0)
est_frec_FT_3 = idx_FT3 * df

mag_BM3 = np.abs(X_mat_BM3[:N//2, :])
idx_BM3 = np.argmax(mag_BM3, axis=0)
est_frec_BM_3 = idx_BM3 * df

mag_R3 = np.abs(X_mat_R3[:N//2, :])
idx_R3 = np.argmax(mag_R3, axis=0)
est_frec_R_3 = idx_R3 * df

mag_H3 = np.abs(X_mat_H3[:N//2, :])
idx_H3 = np.argmax(mag_H3, axis=0)
est_frec_H_3 = idx_H3 * df


# SNR = 10 dB 
mag_FT10 = np.abs(X_mat_ft10[:N//2, :])
idx_FT10 = np.argmax(mag_FT10, axis=0)
est_frec_FT_10 = idx_FT10 * df

mag_BM10 = np.abs(X_mat_BM10[:N//2, :])
idx_BM10 = np.argmax(mag_BM10, axis=0)
est_frec_BM_10 = idx_BM10 * df

mag_R10 = np.abs(X_mat_R10[:N//2, :])
idx_R10 = np.argmax(mag_R10, axis=0)
est_frec_R_10 = idx_R10 * df

mag_H10 = np.abs(X_mat_H10[:N//2, :])
idx_H10 = np.argmax(mag_H10, axis=0)
est_frec_H_10 = idx_H10 * df

# Grafico los histogramas
plt.figure()
plt.suptitle("Histograma del estimador de frecuencia")

plt.subplot(1,2,1)
plt.title("SNR = 3 dB")
plt.hist(est_frec_BM_3, label = 'Blackman', alpha = trans, bins = 30)
plt.hist(est_frec_R_3, label = 'Rectangular', alpha = trans, bins = 30)
plt.hist(est_frec_FT_3, label = 'Flattop', alpha = trans, bins = 30)
plt.hist(est_frec_H_3, label = 'Hamming', alpha = trans, bins = 30)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("#Cantidad de ocurrencias")
plt.legend()

plt.subplot(1,2,2)
plt.title("SNR = 10 dB")
plt.hist(est_frec_BM_10, label = 'Blackman', alpha = trans, bins = 30)
plt.hist(est_frec_R_10, label = 'Rectangular', alpha = trans, bins = 30)
plt.hist(est_frec_FT_10, label = 'Flattop', alpha = trans, bins = 30)
plt.hist(est_frec_H_10, label = 'Hamming', alpha = trans, bins = 30)
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("#Cantidad de ocurrencias")
plt.legend()

plt.tight_layout()
plt.show()

# %% SESGO Y VARIANZA

# Valor real de referencia en dB para amplitud
amplitud_referencia_dB = 20 * np.log10(a0 / np.sqrt(2)) # Valor RMS pasando a dB

# Sesgo de los estimadores de amplitud (SNR = 3dB)
sesgo_amp_rectangular3 = np.median(estimador_a_R_3) - amplitud_referencia_dB
sesgo_amp_blackman3 = np.median(estimador_a_BM_3) - amplitud_referencia_dB
sesgo_amp_flattop3 = np.median(estimador_a_FT_3) - amplitud_referencia_dB
sesgo_amp_hamming3 = np.median(estimador_a_H_3) - amplitud_referencia_dB

# Varianza de los estimadores de amplitus (SNR = 3dB)
var_amp_rectangular3 = np.var(estimador_a_R_3, ddof = 1)
var_amp_blackman3 = np.var(estimador_a_BM_3, ddof = 1)
var_amp_flattop3 = np.var(estimador_a_FT_3, ddof = 1)
var_amp_hamming3 = np.var(estimador_a_H_3, ddof = 1)

print("\n===== Amplitud (dB) =====")
print("SNR = 3dB")
print(f"Rectangular: sesgo = {sesgo_amp_rectangular3:.4f}, varianza = {var_amp_rectangular3:.4f}")
print(f"Blackman: sesgo = {sesgo_amp_blackman3:.4f}, varianza = {var_amp_blackman3:.4f}")
print(f"Flat-top: sesgo = {sesgo_amp_flattop3:.4f}, varianza = {var_amp_flattop3:.4f}")
print(f"Hamming: sesgo = {sesgo_amp_hamming3:.4f}, varianza = {var_amp_hamming3:.4f}")

# Sesgo de los estimadores de amplitud (SNR = 10 dB)
sesgo_amp_rectangular10 = np.median(estimador_a_R_10) - amplitud_referencia_dB
sesgo_amp_blackman10 = np.median(estimador_a_BM_10) - amplitud_referencia_dB
sesgo_amp_flattop10 = np.median(estimador_a_FT_10) - amplitud_referencia_dB
sesgo_amp_hamming10 = np.median(estimador_a_H_10) - amplitud_referencia_dB

## Varianza de los estimadores de amplitud (SNR = 10 dB)
var_amp_rectangular10 = np.var(estimador_a_R_10, ddof=1)
var_amp_blackman10 = np.var(estimador_a_BM_10, ddof=1)
var_amp_flattop10 = np.var(estimador_a_FT_10, ddof=1)
var_amp_hamming10 = np.var(estimador_a_H_10, ddof=1)

print("\nSNR = 10dB")
print(f"Rectangular: sesgo = {sesgo_amp_rectangular10:.4f}, varianza = {var_amp_rectangular10:.4f}")
print(f"Blackman: sesgo = {sesgo_amp_blackman10:.4f}, varianza = {var_amp_blackman10:.4f}")
print(f"Flat-top: sesgo = {sesgo_amp_flattop10:.4f}, varianza = {var_amp_flattop10:.4f}")
print(f"Hamming: sesgo = {sesgo_amp_hamming10:.4f}, varianza = {var_amp_hamming10:.4f}")

# Valor real de la frecuencia en Hz
f_referencia = omega_1 * df

# Sesgo de los estimadores de frecuencia (SNR = 3 dB)
sesgo_frec_rectangular3 = np.median(est_frec_R_3) - f_referencia
sesgo_frec_blackman3 = np.median(est_frec_BM_3) - f_referencia
sesgo_frec_flattop3 = np.median(est_frec_FT_3) - f_referencia
sesgo_frec_hamming3 = np.median(est_frec_H_3) - f_referencia

# Varianza de los estimadores de frecuencia (SNR = 3 dB)
var_frec_rectangular3 = np.var(est_frec_R_3, ddof=1)
var_frec_blackman3 = np.var(est_frec_BM_3, ddof=1)
var_frec_flattop3 = np.var(est_frec_FT_3, ddof=1)
var_frec_hamming3 = np.var(est_frec_H_3, ddof=1)

print("\n===== Frecuencia (Hz) =====")
print("\nSNR = 3dB")
print(f"Rectangular: sesgo = {sesgo_frec_rectangular3:.4f}, varianza = {var_frec_rectangular3:.4f}")
print(f"Blackman: sesgo = {sesgo_frec_blackman3:.4f}, varianza = {var_frec_blackman3:.4f}")
print(f"Flat-top: sesgo = {sesgo_frec_flattop3:.4f}, varianza = {var_frec_flattop3:.4f}")
print(f"Hamming: sesgo = {sesgo_frec_hamming3:.4f}, varianza = {var_frec_hamming3:.4f}")


# Sesgo de los estimadores de frecuencia (SNR = 10 dB)
sesgo_frec_rectangular10 = np.median(est_frec_R_10) - f_referencia
sesgo_frec_blackman10 = np.median(est_frec_BM_10) - f_referencia
sesgo_frec_flattop10 = np.median(est_frec_FT_10) - f_referencia
sesgo_frec_hamming10 = np.median(est_frec_H_10) - f_referencia

# Varianza de los estimadores de frecuencia (SNR = 10 dB)
var_frec_rectangular10 = np.var(est_frec_R_10, ddof=1)
var_frec_blackman10 = np.var(est_frec_BM_10, ddof=1)
var_frec_flattop10 = np.var(est_frec_FT_10, ddof=1)
var_frec_hamming10 = np.var(est_frec_H_10, ddof=1)

print("\nSNR = 10dB")
print(f"Rectangular: sesgo = {sesgo_frec_rectangular10:.4f}, varianza = {var_frec_rectangular10:.4f}")
print(f"Blackman: sesgo = {sesgo_frec_blackman10:.4f}, varianza = {var_frec_blackman10:.4f}")
print(f"Flat-top: sesgo = {sesgo_frec_flattop10:.4f}, varianza = {var_frec_flattop10:.4f}")
print(f"Hamming: sesgo = {sesgo_frec_hamming10:.4f}, varianza = {var_frec_hamming10:.4f}")









