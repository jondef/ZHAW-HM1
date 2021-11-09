import sympy as sy
from decimal import Decimal
import matplotlib.pyplot as plt
import math

"""Für Aufgaben der Art
'Das Polynom vierten Grades
f(x) = 230x^4 + 18x^3 + 9x^2 − 221x − 9
besitzt zwei reelle Nullstellen, die erste x1 im Intervall [−1, 0] und die zweite x2 im Intervall [0, 1].
a) Versuchen Sie, diese Nullstellen mit einer Fixpunktiteration xn+1 = F(xn) bis auf 10^−6 genau zu bestimmen.
Stellen Sie dafür die entsprechende Fixpunktgleichung F(x) = x auf und wählen Sie geeignete Startwerte gemäss
der Abbildung. Was stellen Sie bzgl. der Nullstelle in [0, 1] fest? Weshalb?

b) F(x) erreicht auf dem Intervall [−0.5, 0.5] sein Minimum für x = 0 und sein Maximum für x = 0.5. |F'(x)| wird
maximal für x = 0.5. Zeigen Sie, dass für den ersten Fixpunkt x1 auf dem Intervall [−0.5, 0.5] die Bedingungen
des Banachschen Fixpunktsatzes erfüllt sind und bestimmen Sie α.

c) Wie häufig müssten Sie gemäss der a-priori Fehlerabschätzung iterieren, damit der absolute Fehler für x1 kleiner
als 10^−9 wird?'

@version: 1.0, 23.01.2021
@author: zahlesev
"""

x = sy.symbols('x')

"""==================== INPUT ===================="""
# ACHTUNG: Für Funktionen wie cos usw die Sympy-Befehle brauchen, z.B. sy.cos(), sy.exp() usw.
F = (1.0/221.0) * (230.0 * x ** 4.0 + 18.0 * x ** 3.0 + 9.0 * x ** 2.0 - 9.0)  # Linke Seite der Fixpunktgleichung F(x) = x
x0 = 0  # Startwert der Iteration. (Setze x0 = 1 für divergierenden Fixpunkt in [0, 1])
precision = 1e-9  # Wie genau die Lösung sein soll. Bei 10^-6 = 1e-6
min_iterations = 5  # Minimale Anzahl Iterationen bei divergierenden Folgen, bevor abgebrochen wird.

# Ob die Lipschitz-Konstante α berechnet werden soll, um die a-priori
# und a-posteriori Abschätzungen zu berechnen, sowie zu prüfen, ob der
# Banach'sche Fixpunktsatz erfüllt ist.
check_banach_and_calculate_estimations = True

# Intervall [a, b], in welchem sich der gesuchte Fixpunkt befindet (nur benötigt wenn Lipschitz-Konstante
# berechnet werden soll) a muss kleiner als b sein.
a = -0.5; b = 0.5
max_a_priori_error = 1e-9

# Ob die Fixpunktgleichung geplottet werden soll
show_plots = True
ap = -1; bp = 1  # Intervall, über welchem geplottet werden soll.
"""==============================================="""

def check_banach(F, x, a, b, show_plots):
    print("Prüfen, ob der Banach'sche Fixpunktsatz erfüllt ist:")
    print("1. Bedingung: F(x) bildet [" + str(a) + ", " + str(b) + "] auf [" + str(a) + ", " + str(b) + "] ab.")
    print("2. Bedingung: F(x) ist Lipschitz-stetig, d.h. |F(x) - F(y)| <= α * |x - y| ∀ x, y ∈ [" + str(a) + ", " + str(b) + "]\n")

    steps = 1000
    d = (Decimal(b) - Decimal(a)) / steps
    print("Prüfen der ersten Bedingung: Finde maximalen und minimalen Wert von F(x) im Intervall [" + str(a) + ", " + str(b) + "], mit Python, indem " + str(steps + 1) + " äquidistante Werte im Intervall berechnet werden.\n")

    xvalues = []
    yvalues = []

    xmax = a
    xmin = a
    ymax = F.subs(x, a)
    ymin = F.subs(x, a)

    for i in range(steps + 1):
        xvalue = Decimal(a) + i * d
        yvalue = F.subs(x, xvalue)

        xvalues.append(xvalue)
        yvalues.append(yvalue)

        if yvalue < ymin: ymin = yvalue; xmin = xvalue
        if yvalue > ymax: ymax = yvalue; xmax = xvalue

    print("Minimaler Wert von F(x) in [" + str(a) + ", " + str(b) + "]: F(" + str(xmin) + ") = " + str(ymin))
    print("Maximaler Wert von F(x) in [" + str(a) + ", " + str(b) + "]: F(" + str(xmax) + ") = " + str(ymax))
    print("=> F(x) bildet [" + str(a) + ", " + str(b) + "] auf [" + str(ymin) + ", " + str(ymax) + "] ab.\n")

    if ymin >= Decimal(a) and ymax <= Decimal(b):
        print("==> Da [" + str(ymin) + ", " + str(ymax) + "] ⊂ [" + str(a) + ", " + str(b) + "] ist die erste Bedingung erfüllt.\n")
    else:
        print("==> Da [" + str(ymin) + ", " + str(ymax) + "] ⊄ [" + str(a) + ", " + str(b) + "] ist die erste Bedingung NICHT erfüllt.\n")
        return 0

    if show_plots:
        plt.figure(2)
        plt.plot(xvalues, yvalues)
        plt.axvline(xmin, color="blue", linewidth="4")
        plt.axvline(xmax, color="red", linewidth="4")
        plt.xlim(a, b)
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("F(x)")
        plt.title("Min/Max von F(x) über [" + str(a) + ", " + str(b) + "]")
        plt.show()

    print("Prüfen der zweiten Bedingung: Bestimme α (Lipschitz-Konstante) als α = max(|F'(x0)|), mit " + str(a) + " <= x0 <= " + str(b) + " mit Python.")

    dF = sy.diff(F, x)
    print("F'(x) = " + str(dF))

    xvalues = []
    yvalues = []

    xmax = a
    ymax = dF.subs(x, a)

    for i in range(steps + 1):
        xvalue = Decimal(a) + i * d
        yvalue = dF.subs(x, xvalue)

        xvalues.append(xvalue)
        yvalues.append(yvalue)

        if abs(yvalue) > ymax: ymax = yvalue; xmax = xvalue

    alpha = ymax

    print("Maximaler Wert von F'(x0) in [" + str(a) + ", " + str(b) + "]: F'(" + str(xmax) + ") = α = " + str(alpha.evalf()))

    if show_plots:
        plt.figure(2)
        plt.plot(xvalues, yvalues)
        plt.axvline(xmax, color="red", linewidth="4")
        plt.xlim(a, b)
        plt.grid()
        plt.xlabel("x")
        plt.ylabel("F(x)")
        plt.title("F'(x) über [" + str(a) + ", " + str(b) + "]")
        plt.show()

    print("Überprüfung der Lipschitz-Stetigkeit |F(x) - F(y)| <= α * |x - y| ∀ x, y ∈ [" + str(a) + ", " + str(b) + "], entweder anschaulich durch Plot, oder numerisch grobe Prüfung mit Python.\n")

    steps = int(math.sqrt(1000))
    d = (Decimal(b) - Decimal(a)) / steps
    lipschitzOK = True

    for i in range(steps + 1):
        x1value = Decimal(a) + i * d
        y1value = F.subs(x, x1value)

        for j in range(steps + 1):
            x2value = Decimal(a) + i * d
            y2value = F.subs(x, x2value)

            dy = abs(y1value - y2value)  # |F(x) - F(y)|
            dx = abs(x1value - x2value)  # |x - y|

            if dy > alpha * dx:
                print("==> Lipschitz-Stetigkeit ist NICHT erfüllt für x = " + str(x1value) + " und y = " + str(x2value) + "! |F(x) - F(y)| = " + str(dy) + ", α * |x - y| = " + str(alpha * dx))
                lipschitzOK = False
                return 0

    if lipschitzOK:
        print("==> Lipschitz-Stetigkeit ist erfüllt, für alle geprüften x, y\n")
        print("===>>> Banach'scher Fixpunktsatz ist erfüllt!\n\n")
        return alpha

"""
MAIN PROGRAM
"""
# Plot F(x) and x
if show_plots:
    steps = 1000
    d = (Decimal(bp) - Decimal(ap)) / steps

    xvalues = []
    yvalues = []

    for i in range(steps + 1):
        xvalue = Decimal(ap) + i * d
        yvalue = F.subs(x, xvalue)

        xvalues.append(xvalue)
        yvalues.append(yvalue)

    plt.figure(1)
    plt.plot(xvalues, yvalues, label="F(x)")
    plt.plot(xvalues, xvalues, label="x")
    plt.xlim(ap, bp)
    plt.grid()
    plt.xlabel("x")
    plt.ylabel("F(x)")
    plt.title("Fixpunktiteration F(x) = x")
    plt.legend()
    plt.show()

# Check banach and calculate a priori estimation
alpha = 0
if check_banach_and_calculate_estimations:
    alpha = check_banach(F, x, a, b, show_plots)

    if alpha > 0:
        # a-priori estimation
        print("A-Priori-Abschätzung: Finde n, so dass (α^n / (1 - α)) * |x1 - x0| <= a-priori-Fehler")

        n_apriori = math.ceil((math.log((max_a_priori_error * (1 - alpha)) / (abs(F.subs(x, x0) - x0)))) / (math.log(alpha)))
        print("=> Aus A-Priori-Abschätzung folgt, dass absoluter Fehler < " + str(max_a_priori_error) + " für alle n >= " + str(n_apriori) + ".\n\n")



print("FIXPUNKTITERATION")
print("=================")
xn = [x0, F.subs(x, x0)]

print("Δ := Differenz zwischen den letzten zwei Resultaten.")

if check_banach_and_calculate_estimations:
    print("Δ_a_posteriori := a-posteriori-Abschätzung")

print("")

print("n = 0: x0 = " + str(xn[0]))
print("n = 1: x1 = F(x0) = " + str(xn[1]))

n = 1
while abs(xn[n] - xn[n-1]) > precision:
    n += 1

    xn.append(F.subs(x, xn[n-1]))

    if n > min_iterations and xn[n] > xn[0]:
        print("Folge divergiert! Kein Fixpunkt!")
        break

    if check_banach_and_calculate_estimations and alpha > 0:
        dx_a_posteriori = (alpha / (1 - alpha)) * abs(xn[n] - xn[n - 1])
        print("n = " + str(n) + ": x" + str(n) + " = F(x" + str(n - 1) + ") = " + str(xn[n]) + ", Δ = " + str(abs(xn[n] - xn[n - 1])) + ", Δ_a_posteriori ≤ " + str(dx_a_posteriori.evalf()))
    else:
        print("n = " + str(n) + ": x" + str(n) + " = F(x" + str(n - 1) + ") = " + str(xn[n]) + ", Δ = " + str(abs(xn[n] - xn[n - 1])))

