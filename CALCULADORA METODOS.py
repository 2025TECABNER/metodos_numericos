
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import ast
from sympy import symbols, expand, lambdify
# ABNER MOISES SARMIENTO VALENCIA S4B BY METODOS NUMERICOS
# Métodos Cerrados
def biseccion(f, a, b, tol):
    pasos = []
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe cambiar de signo en el intervalo")
    while (b - a) / 2 > tol:
        c = (a + b) / 2
        pasos.append(round(c, 5))
        if f(c) == 0:
            break
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return round(c, 5), pasos

def falsa_posicion(f, a, b, tol):
    pasos = []
    if f(a) * f(b) >= 0:
        raise ValueError("La función debe cambiar de signo en el intervalo")
    c = a
    while abs(f(c)) > tol:
        c = b - (f(b) * (a - b)) / (f(a) - f(b))
        pasos.append(round(c, 5))
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return round(c, 5), pasos

# Métodos Abiertos
def newton_raphson(f, df, x0, tol):
    pasos = [round(x0, 5)]
    while abs(f(x0)) > tol:
        x0 = x0 - f(x0) / df(x0)
        pasos.append(round(x0, 5))
    return round(x0, 5), pasos

def secante(f, x0, x1, tol):
    pasos = [round(x0, 5), round(x1, 5)]
    while abs(x1 - x0) > tol:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        x0, x1 = x1, x2
        pasos.append(round(x2, 5))
    return round(x1, 5), pasos

# Gauss-Seidel
def gauss_seidel(A, b, x0, tol, max_iter=100):
    n = len(A)
    x = x0.copy()
    historial = []
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            sum1 = sum(A[i][j] * x_new[j] for j in range(i))
            sum2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - sum1 - sum2) / A[i][i]
        historial.append([round(val, 5) for val in x_new])
        if np.linalg.norm(np.array(x_new) - np.array(x), ord=np.inf) < tol:
            break
        x = x_new
    return historial



def interpolacion_lagrange(x, y):
    poly = lagrange(x, y)
    coef = np.poly1d(poly).coefficients
    coef = [0 if abs(c) < 1e-10 else round(c, 5) for c in coef]

    grados = list(range(len(coef) - 1, -1, -1))
    terminos = []

    for c, g in zip(coef, grados):
        if c == 0:
            continue
        if g == 0:
            terminos.append(f"{c}")
        elif g == 1:
            if c == 1:
                terminos.append("x")
            elif c == -1:
                terminos.append("-x")
            else:
                terminos.append(f"{c}x")
        else:
            if c == 1:
                terminos.append(f"x^{g}")
            elif c == -1:
                terminos.append(f"-x^{g}")
            else:
                terminos.append(f"{c}x^{g}")

    formula = " + ".join(terminos).replace("+ -", "- ")
    poly.formula = formula
    return poly

def interpolacion_newton(x, y):
    x = np.array(x, dtype=float)
    coef = np.array(y, dtype=float)
    n = len(x)

    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j - 1]) / (x[j:n] - x[j - 1])

    def p(t):
        total = coef[0]
        for i in range(1, n):
            total += coef[i] * np.prod([t - x[j] for j in range(i)])
        return total

    
    poly_expr = np.poly1d([0.0])
    for i in range(n):
        term = np.poly1d([1.0])
        for j in range(i):
            term *= np.poly1d([1.0, -x[j]])  
        term *= coef[i]
        poly_expr += term

    
    coef = [0 if abs(c) < 1e-10 else round(c, 5) for c in poly_expr.coefficients]
    grados = list(range(len(coef) - 1, -1, -1))
    terminos = []

    for c, g in zip(coef, grados):
        if c == 0:
            continue
        if g == 0:
            terminos.append(f"{c}")
        elif g == 1:
            if c == 1:
                terminos.append("x")
            elif c == -1:
                terminos.append("-x")
            else:
                terminos.append(f"{c}x")
        else:
            if c == 1:
                terminos.append(f"x^{g}")
            elif c == -1:
                terminos.append(f"-x^{g}")
            else:
                terminos.append(f"{c}x^{g}")

    formula = " + ".join(terminos).replace("+ -", "- ")
    p.formula = formula
    return p


# Runge-Kutta 2do orden
def runge_kutta_2(f, x0, y0, h, n):
    x = x0
    y = y0
    puntos = [(round(x, 5), round(y, 5))]
    for _ in range(n):
        k1 = h * f(x, y)
        k2 = h * f(x + h, y + k1)
        y = y + 0.5 * (k1 + k2)
        x = x + h
        puntos.append((round(x, 5), round(y, 5)))
    return puntos

# Interfaz Gráfica
class CalculadoraNumerica(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Calculadora de Métodos Numéricos")
        self.geometry("1000x700")
        self.configure(bg="white")

        self.metodo_var = tk.StringVar()
        self.entries = {}

        self.show_welcome_screen()

    def show_welcome_screen(self):
        self.welcome_frame = tk.Frame(self, bg="white")
        self.welcome_frame.pack(expand=True, fill="both")

        tk.Label(self.welcome_frame, text="Bienvenido a tu calculadora de Métodos Numéricos",
                 font=("Arial", 24), bg="white").pack(pady=40)

        tk.Label(self.welcome_frame, text="by Abner Moisés Sarmiento Valencia - S4B",
                 font=("Arial", 12), bg="white").pack(pady=10)

        tk.Button(self.welcome_frame, text="Iniciar", font=("Arial", 14),
                  command=self.start_calculator).pack(pady=30)

    def start_calculator(self):
        self.welcome_frame.destroy()
        self.create_widgets()

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 12), background="white")
        style.configure("TButton", font=("Arial", 12))
        style.configure("TCombobox", font=("Arial", 12))

        ttk.Label(self, text="Seleccione el método:").pack(pady=10)
        metodo_menu = ttk.Combobox(self, textvariable=self.metodo_var, state="readonly")
        metodo_menu['values'] = [
            "Bisección", "Falsa Posición", "Newton-Raphson", "Secante",
            "Gauss-Seidel", "Interpolación de Lagrange", "Interpolación de Newton", "Runge-Kutta 2º orden"
        ]
        metodo_menu.pack()
        metodo_menu.bind("<<ComboboxSelected>>", self.display_inputs)

        self.input_frame = ttk.Frame(self)
        self.input_frame.pack(pady=10)

        self.result_text = tk.Text(self, height=12, font=("Consolas", 11))
        self.result_text.pack(pady=10, fill=tk.X)

        self.plot_frame = ttk.Frame(self)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

    def clear_inputs(self):
        for widget in self.input_frame.winfo_children():
            widget.destroy()
        self.entries = {}

    def display_inputs(self, event):
        self.clear_inputs()
        metodo = self.metodo_var.get()
        campos = []

        if metodo in ["Bisección", "Falsa Posición"]:
            campos = ["f(x)", "a", "b", "tol"]
        elif metodo == "Newton-Raphson":
            campos = ["f(x)", "f'(x)", "x0", "tol"]
        elif metodo == "Secante":
            campos = ["f(x)", "x0", "x1", "tol"]
        elif metodo == "Gauss-Seidel":
            campos = ["Matriz A", "Vector b", "x0", "tol"]
        elif metodo in ["Interpolación de Lagrange", "Interpolación de Newton"]:
            campos = ["x", "y"]
        elif metodo == "Runge-Kutta 2º orden":
            campos = ["f(x, y)", "x0", "y0", "h", "n"]

        for campo in campos:
            ttk.Label(self.input_frame, text=campo + ":").pack()
            entry = ttk.Entry(self.input_frame)
            entry.pack()
            self.entries[campo] = entry

        ttk.Button(self.input_frame, text="Ejecutar", command=self.run_method).pack(pady=10)

    def run_method(self):
        self.result_text.delete("1.0", tk.END)
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        metodo = self.metodo_var.get()

        try:
            if metodo in ["Bisección", "Falsa Posición"]:
                f = eval("lambda x: " + self.entries["f(x)"].get())
                a = float(self.entries["a"].get())
                b = float(self.entries["b"].get())
                tol = float(self.entries["tol"].get())
                metodo_func = biseccion if metodo == "Bisección" else falsa_posicion
                raiz, pasos = metodo_func(f, a, b, tol)
                self.result_text.insert(tk.END, f"Raíz: {raiz}\nPasos: {pasos}")
                self.plot_function(f, pasos)

            elif metodo == "Newton-Raphson":
                f = eval("lambda x: " + self.entries["f(x)"].get())
                df = eval("lambda x: " + self.entries["f'(x)"].get())
                x0 = float(self.entries["x0"].get())
                tol = float(self.entries["tol"].get())
                raiz, pasos = newton_raphson(f, df, x0, tol)
                self.result_text.insert(tk.END, f"Raíz: {raiz}\nPasos: {pasos}")
                self.plot_function(f, pasos)

            elif metodo == "Secante":
                f = eval("lambda x: " + self.entries["f(x)"].get())
                x0 = float(self.entries["x0"].get())
                x1 = float(self.entries["x1"].get())
                tol = float(self.entries["tol"].get())
                raiz, pasos = secante(f, x0, x1, tol)
                self.result_text.insert(tk.END, f"Raíz: {raiz}\nPasos: {pasos}")
                self.plot_function(f, pasos)

            elif metodo == "Gauss-Seidel":
                A = np.array(ast.literal_eval(self.entries["Matriz A"].get()), dtype=float)
                b = np.array(ast.literal_eval(self.entries["Vector b"].get()), dtype=float)
                x0 = np.array(ast.literal_eval(self.entries["x0"].get()), dtype=float)
                tol = float(self.entries["tol"].get())
                historial = gauss_seidel(A, b, x0, tol)
                for i, vec in enumerate(historial):
                    limpio = [round(float(v), 5) for v in vec]
                final = [round(float(v), 5) for v in historial[-1]]
                self.result_text.insert(tk.END, f"Resultado final: {final}\n")
                self.plot_xy(list(range(1, len(historial) + 1)), [v[0] for v in historial])

            elif metodo == "Interpolación de Lagrange":
                x = np.array(ast.literal_eval(self.entries["x"].get()), dtype=float)
                y = np.array(ast.literal_eval(self.entries["y"].get()), dtype=float)
                poly = interpolacion_lagrange(x, y)
                self.result_text.insert(tk.END, f"Polinomio generado:\n{poly.formula}\n")
                self.plot_function(poly, x)

            elif metodo == "Interpolación de Newton":
                x = np.array(ast.literal_eval(self.entries["x"].get()), dtype=float)
                y = np.array(ast.literal_eval(self.entries["y"].get()), dtype=float)
                poly = interpolacion_newton(x, y)
                self.result_text.insert(tk.END, f"Polinomio generado (expandido):\n{poly.formula}\n")
                self.plot_function(poly, x)

            elif metodo == "Runge-Kutta 2º orden":
                f = eval("lambda x, y: " + self.entries["f(x, y)"].get())
                x0 = float(self.entries["x0"].get())
                y0 = float(self.entries["y0"].get())
                h = float(self.entries["h"].get())
                n = int(self.entries["n"].get())
                puntos = runge_kutta_2(f, x0, y0, h, n)
                xs, ys = zip(*puntos)
                self.result_text.insert(tk.END, f"Puntos: {puntos}")
                self.plot_xy(xs, ys)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def plot_function(self, f, pasos):
        fig, ax = plt.subplots()
        x_vals = np.linspace(min(pasos) - 1, max(pasos) + 1, 400)
        y_vals = [f(x) for x in x_vals]
        ax.plot(x_vals, y_vals, label='f(x)')
        ax.plot(pasos, [f(x) for x in pasos], 'ro-', label='Iteraciones')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.legend()
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    def plot_xy(self, x, y):
        fig, ax = plt.subplots()
        ax.plot(x, y, marker='o')
        ax.grid(True)
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

if __name__ == '__main__':
    app = CalculadoraNumerica()
    app.mainloop()