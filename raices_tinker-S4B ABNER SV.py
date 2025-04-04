import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class RootFindingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cálculo de Raíces: Secante y Newton-Raphson")
        self.root.geometry("900x820")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)

        self.metodo_actual = tk.StringVar(value="Esperando selección...")
        self.setup_styles()
        self.setup_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background="#f0f0f0")
        style.configure("TLabelFrame", background="#ffffff", foreground="#333333")
        style.configure("TLabel", background="#ffffff", font=("Arial", 10))
        style.configure("TButton", background="#e0e0e0", relief="flat", font=("Arial", 10))
        style.configure("TEntry", font=("Arial", 10))

    def setup_ui(self):
        input_frame = ttk.LabelFrame(self.root, text="Parámetros de entrada", padding=10)
        input_frame.pack(pady=10, padx=10, fill='x')

        labels = ["Función f(x):", "x0:", "x1 (Secante):", "Tolerancia:"]
        entries = []
        for i, text in enumerate(labels):
            ttk.Label(input_frame, text=text).grid(row=i, column=0, sticky='e', pady=5, padx=5)
            entry = ttk.Entry(input_frame, width=30)
            entry.grid(row=i, column=1, pady=5, padx=5, sticky='w')
            entries.append(entry)

        self.func_entry, self.x0_entry, self.x1_entry, self.tol_entry = entries

        metodo_frame = ttk.Frame(self.root)
        metodo_frame.pack(pady=5)
        ttk.Label(metodo_frame, text="Método actual:", font=("Arial", 11, "bold")).pack(side="left")
        ttk.Label(metodo_frame, textvariable=self.metodo_actual, font=("Arial", 11)).pack(side="left", padx=10)

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)
        ttk.Button(button_frame, text="Método Secante", command=self.metodo_secante).grid(row=0, column=0, padx=20)
        ttk.Button(button_frame, text="Newton-Raphson", command=self.metodo_newton).grid(row=0, column=1, padx=20)
        ttk.Button(button_frame, text="Guardar Gráfica", command=self.guardar_grafica).grid(row=1, column=0, pady=5)
        ttk.Button(button_frame, text="Guardar Resultados", command=self.guardar_resultados).grid(row=1, column=1, pady=5)

        self.result_label = ttk.Label(self.root, text="Resultado: ", font=("Arial", 12, "bold"), background="#f0f0f0")
        self.result_label.pack(pady=5)

        self.text_area = scrolledtext.ScrolledText(self.root, width=90, height=12, font=("Courier New", 10))
        self.text_area.pack(pady=10)

        graph_frame = ttk.LabelFrame(self.root, text="Visualización de la función", padding=10)
        graph_frame.pack(pady=10, padx=10, fill='both', expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 4.5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def validar_entrada(self, metodo):
        try:
            f_expr = self.func_entry.get()
            if not f_expr:
                raise ValueError("Ingrese una función válida.")
            x = sp.symbols('x')
            f = sp.sympify(f_expr)

            float(self.x0_entry.get())
            if metodo == "secante":
                if not self.x1_entry.get():
                    raise ValueError("Para el método de la Secante, debe ingresar x1.")
                float(self.x1_entry.get())

            float(self.tol_entry.get())
            return f
        except Exception as e:
            messagebox.showerror("Error de entrada", f"Revise los campos ingresados:\n{str(e)}")
            return None

    def plot_function_with_points(self, f_lambd, points, raiz_aprox):
        self.ax.clear()
        x_vals = np.linspace(min(points)-1, max(points)+1, 400)
        y_vals = f_lambd(x_vals)

        self.ax.plot(x_vals, y_vals, label='f(x)', linewidth=2)
        self.ax.axhline(0, color='gray', linestyle='--')
        self.ax.plot(points, [f_lambd(p) for p in points], 'ro', label='Iteraciones')
        self.ax.plot(raiz_aprox, f_lambd(raiz_aprox), 'go', label=f'Raíz ≈ {raiz_aprox:.5f}')
        self.ax.set_title('Aproximación de la raíz')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('f(x)')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()

    def metodo_secante(self):
        self.metodo_actual.set("Método Secante")
        f = self.validar_entrada("secante")
        if f is None:
            return

        try:
            x = sp.symbols('x')
            f_lambd = sp.lambdify(x, f, 'numpy')
            x0 = float(self.x0_entry.get())
            x1 = float(self.x1_entry.get())
            tol = float(self.tol_entry.get())

            if abs(f_lambd(x0) - f_lambd(x1)) < 1e-12:
                messagebox.showwarning("Advertencia", "f(x0) y f(x1) son demasiado similares.")
                return

            iteraciones = 0
            puntos = [x0, x1]
            self.text_area.delete('1.0', tk.END)
            self.text_area.insert(tk.END, f"{'Iter':<5}{'x':>15}{'f(x)':>15}{'Error':>15}\n")
            self.text_area.insert(tk.END, "-"*50 + "\n")

            while abs(x1 - x0) > tol and iteraciones < 100:
                fx0, fx1 = f_lambd(x0), f_lambd(x1)
                denominador = fx1 - fx0
                if abs(denominador) < 1e-12:
                    messagebox.showerror("Error", "División por cero en la fórmula.")
                    return
                x_new = x1 - fx1 * (x1 - x0) / denominador
                error = abs(x1 - x0)
                self.text_area.insert(tk.END, f"{iteraciones:<5}{x1:>15.6f}{fx1:>15.6f}{error:>15.6f}\n")
                x0, x1 = x1, x_new
                puntos.append(x1)
                iteraciones += 1

            self.result_label.config(text=f"Raíz estimada: {x1:.6f}")
            self.plot_function_with_points(f_lambd, puntos, x1)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def metodo_newton(self):
        self.metodo_actual.set("Método Newton-Raphson")
        f = self.validar_entrada("newton")
        if f is None:
            return

        try:
            x = sp.symbols('x')
            df = sp.diff(f, x)
            f_lambd = sp.lambdify(x, f, 'numpy')
            df_lambd = sp.lambdify(x, df, 'numpy')
            x0 = float(self.x0_entry.get())
            tol = float(self.tol_entry.get())

            iteraciones = 0
            puntos = [x0]
            self.text_area.delete('1.0', tk.END)
            self.text_area.insert(tk.END, f"{'Iter':<5}{'x':>15}{'f(x)':>15}{'Error':>15}\n")
            self.text_area.insert(tk.END, "-"*50 + "\n")

            while abs(f_lambd(x0)) > tol and iteraciones < 100:
                fx = f_lambd(x0)
                dfx = df_lambd(x0)
                if abs(dfx) < 1e-12:
                    messagebox.showerror("Error", "La derivada es cero.")
                    return
                x_new = x0 - fx / dfx
                error = abs(x_new - x0)
                self.text_area.insert(tk.END, f"{iteraciones:<5}{x0:>15.6f}{fx:>15.6f}{error:>15.6f}\n")
                x0 = x_new
                puntos.append(x0)
                iteraciones += 1

            self.result_label.config(text=f"Raíz estimada: {x0:.6f}")
            self.plot_function_with_points(f_lambd, puntos, x0)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def guardar_grafica(self):
        try:
            self.fig.savefig("grafica_raiz.png")
            messagebox.showinfo("Éxito", "Gráfica guardada como 'grafica_raiz.png'.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar la gráfica.\n{e}")

    def guardar_resultados(self):
        try:
            texto = self.text_area.get("1.0", tk.END)
            with open("iteraciones.txt", "w") as file:
                file.write(texto)
            messagebox.showinfo("Éxito", "Resultados guardados en 'iteraciones.txt'.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar el archivo.\n{e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RootFindingApp(root)
    root.mainloop()
