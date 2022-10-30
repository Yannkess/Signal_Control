import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control
import scipy

###############################################################################################
#Dane
r1 = [4, -4, 2]
p1 = [0, -1, 2]
k1 = 0
num, den = signal.invres(r1, p1, k1)
den = den[:-1]

zeros = [1,-4]
poles = [2,-1]
gain = [2]

#num1 = [-8,2,12]
#den1 = [1,0,-4]

num1 = [1, -1, 1]
den1 = [1, 0, 0]


A = np.array([[-3, 4,-4], 
              [-3, 4,-6],
              [-1, 1,-3]])
B = np.array([[-4],
              [-4],
              [-1]])
C = np.array([0, -1,2])
D = np.array([-1])



U = np.array([[-2],
              [-2],
              [-1]])

Y = np.array([[-3],
              [-2],
              [-25]
              ])

###############################################################################################
## Wyliczanie

#sys = signal.dlti(zeros,poles,gain)
sys = signal.dlti(num1,den1)
#sys = signal.StateSpace(A,B,C,D)
tf = signal.TransferFunction(sys)
ss = signal.StateSpace(sys)
r,p,k = signal.residue(tf.num,np.pad(tf.den,(0,1)))
zpk = signal.ZerosPolesGain(sys)

################################################################################################
## Czytanie

print('Odpowiedz impulsowa z tego liczymy odwrotną transformate. Patrz tablice')
print(r,p,k)
print('Macierze Stanu')
print(ss.A)
print(ss.B)
print(ss.C)
print(ss.D)
print('Zera i Bieguny')
print(zpk.zeros)
print(zpk.poles)
print(zpk.gain)
print('Transmitancja')
print(tf.num, tf.den)

##################################################################################################
## Sprawdzenie 

sys1 = signal.dlti(zpk.zeros,zpk.poles,zpk.gain)
sys2 = signal.dlti(tf.num,tf.den)
sys3 = signal.StateSpace(ss.A,ss.B,ss.C,ss.D)

t, g1 = signal.dimpulse(sys1)
print('1. Z odpowiedzi impulsowej policz recznie g[n] i porównaj czy sie zgadza')
print('2. Policz lim G(z) dla z dążącego do nieskończoności wynik powinien być rónwy g[0]')
print('Do obliczenia pochodnej https://www.wolframalpha.com/input?i=limit+calculator')
print('Odpowiedz dla zer i biegunów')
print(g1[0][:5])
t, g2 = signal.dimpulse(sys2)
print('Odpowiedz dla transmitancji')
print(g2[0][:5])

print('Odpowiedz dla macierzy')
gss = np.zeros((5))

for i in range(5):
  if i == 0:
    gss[i] = ss.D
  else:
    gss[i] = ss.C.dot(np.linalg.matrix_power(ss.A, i-1)).dot(ss.B)

print(gss)

#######################################################################################################
## Stabilnosc

poles = zpk.poles
print(poles)
print(np.abs(poles))
zeros = zpk.zeros
n = np.arange(0, 2 * np.pi, 0.01)
x = np.sin(n)
y = np.cos(n)

plt.figure(0)
plt.title("Stability")
plt.plot(x, y)
plt.axis('equal')
plt.scatter(poles.real, poles.imag, color='r')
plt.show()
#plt.scatter(zeros.real, zeros.imag, color='b')

#######################################################################################################
## Pobudzenie U
u = [1, -1, 1, 0]

t,y = signal.dlsim(sys, u)
print('Odpowiedz na pobudzenie U')
print(y)
######################################################################################################
u = np.array([3, 0, 0,2])
g = np.array([0,-2,0,0])

y = np.convolve(u, g)

plt.figure(1,figsize=(10, 15))
plt.subplot(311)
plt.title("Impulse U")
plt.stem(u)
plt.subplot(312)
plt.stem(g)
plt.subplot(313)
plt.stem(y)
plt.show()

print(y)

######################################################################################################
## Diagonalizacja
w, P = np.linalg.eig(A)
print('Macierz Permutacji P')
print(P)
print('')
invP = np.linalg.inv(P)

Ad = invP.dot(A).dot(P)
Bd = invP.dot(B)
Cd = C.dot(P)
Dd = D
print('Macierz A')
print(Ad)
print('')
print('Macierz B')
print(Bd)
print('')
print('Macierz C')
print(Cd)
print('')
print('Macierz D')
print(Dd)
######################################################################################################
## Sterowalnosc

x0 = np.array([[2], [2],[1]])        # w przyadku wyższych macierzy wektory będą dłuższe
x3 = np.array([[52], [33],[41]])


H = control.ctrb(A, B) # zawsze poprawnie wyliczy macierz sterowalnosci lepiej z tego korzystac
invH = np.linalg.inv(H)

A2 = A.dot(A) # To samo co niżej
A3 = np.linalg.matrix_power(A, 3) # To samo co wyżej, w przypadku macierzy 3x3 zmieniamy ostanią cyfre na 3 


U = invH.dot((x3 - A3.dot(x0)))  # U = invH.dot((xn - An.dot(x0))) wzór na pobudzenie U dla wyższych rzędów macierzy
print('Wektor U[n]')
print(U)
u0 = U[2][0]
u1 = U[1][0]
u2 = U[0][0]
print('Wektor U[0]')
print(u0)
print('Wektor U[1]')
print(u1)
print('Wektor U[2]')
print(u2)

#####################################################################################################################
## Obserwowalnosc

Po = np.array([C, C.dot(A), C.dot(A).dot(A)]) # ten sposob nie zawsze dziala gdyz macierze mogą zmieniać rozmiar
Po = control.obsv(A, C)  #specjalna funkcja zawsze działa,  Macierz Obserwowalności G
invPo = np.linalg.inv(Po)

fD = float(D)
fCB = float(C.dot(B))
fCAB = float(C.dot(A).dot(B))
# w zaleznosci czy jest D == 0 lub D ~=0 to fD może być nie używane
# 
# Dla Macierzy 3x3
K = np.array([[fD, 0, 0],           # ([[0, 0, 0],
               [fCB, fD, 0],         #  [fCB, 0, 0],
               [fCAB, fCB, fD]])     #  [fCAB, fCB, 0]])

#K = np.array([[fD,0],
#             [fCB,fD]])
Z = Y-K.dot(U)

u2 = U[2][0]
u1 = U[1][0]
u0 = U[0][0]

x0 = invPo.dot(Z)
x1 = A.dot(x0) + B.dot(u0)
x2 = A.dot(x1) + B.dot(u1)
x3 = A.dot(x2) + B.dot(u2)

print('Wektor x[0]')
print(x0)
print('Wektor x[3]')
print(x3)

##########################################################################################################################
## Charakterystka Częstotliwościowa

#Bode
w, H = signal.dfreqresp(sys)
plt.figure(2,figsize=(12, 8))
plt.subplot(211)
plt.title("Bode Diagram")
plt.plot(w, np.abs(H))
plt.grid()
plt.subplot(212)
plt.plot(w, np.angle(H) * 180 / np.pi)
plt.grid()
plt.show()

#Pobudzenie sygnałem U sprawdzając wykresy z odpowiedzi czestotliwosciowej
n = np.arange(20)
u = np.cos( 2/3 *np.pi * n)



n, y = signal.dlsim(sys, u)

plt.figure(3,figsize=(12, 8))
plt.title("Impulse U V2")
plt.plot(n, u)
plt.stem(n, u)
plt.plot(n, y)
plt.stem(n, y)
plt.grid()
plt.show()


# Ogolne Pobudzenie

o = [1, -1, 1, 0]
u = 5*o


n, y = signal.dlsim(sys, u)

plt.figure(4,figsize=(12, 8))
plt.title("Common Impulse")
plt.plot(n, u)
plt.stem(n, u,)
plt.plot(n, y)
plt.stem(n, y,'r')
plt.grid()
plt.show()