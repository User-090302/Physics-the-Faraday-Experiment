#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import random
import json
from datetime import datetime
import math
import os

# try sklearn, fallback to numpy polyfit
try:
    from sklearn.linear_model import LinearRegression
    SKLEARN_AVAILABLE = True
except Exception:
    import numpy as np
    SKLEARN_AVAILABLE = False

# ------------------- genData (исправленная версия) -------------------
def genData(start, end, kolvo):
    dataI = [start["i"], end["i"]]
    dataB = [start["b"], end["b"]]

    for _ in range(max(0, kolvo - 2)):
        idx1, idx2 = random.sample(range(len(dataI)), 2)
        newValI = (dataI[idx1] + dataI[idx2]) / 2
        newValB = (dataB[idx1] + dataB[idx2]) / 2
        noiseI = random.uniform(-0.01, 0.01) * newValI
        noiseB = random.uniform(-0.01, 0.01) * newValB
        dataI.append(newValI + noiseI)
        dataB.append(newValB + noiseB)

    pairs = sorted(zip(dataI, dataB), key=lambda x: x[0])
    dataISorted, dataBSorted = zip(*pairs)
    return {"i": list(dataISorted), "b": list(dataBSorted)}

# ------------------- КОНСТАНТЫ -------------------
BG_MAIN = "#1e1e1e"
BG_PANEL = "#252526"
FG_TEXT = "#d4d4d4"
ACCENT = "#3c7dd9"
ENTRY_BG = "#2d2d2d"
GRID_COLOR = "#444444"
RED_TEXT = "#cc3333"

L_CONST = 0.02  # константа L (между наконечниками)
EXPERIMENT_ROWS = 9  # количество строк экспериментов

# ------------------- Утилиты -------------------
def parse_calib_json(path):
    """
    Загружает JSON и собирает все пары (I, B) из всех блоков вроде '20mm', '40mm'.
    Возвращает два списка: Ilist, Blist.
    """
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    Ilist = []
    Blist = []

    for key in obj:
        block = obj[key]
        if not isinstance(block, dict):
            continue
        i_array = block.get("i", [])
        b_array = block.get("b", [])

        if len(i_array) != len(b_array):
            raise ValueError(f"В блоке '{key}' количество I и B не совпадает.")
        Ilist.extend([float(v) for v in i_array])
        Blist.extend([float(v) for v in b_array])

    if len(Ilist) < 2:
        raise ValueError("В файле не найдено достаточно пар I/B.")

    return Ilist, Blist

def train_model(Ilist, Blist):
    """
    Тренируем модель I -> B. Возвращаем предиктор function(I_array)->B_array
    Если sklearn доступен, используем LinearRegression (fit_intercept=True),
    иначе применяем np.polyfit(1) и возвращаем функцию предсказания.
    """
    if not Ilist or not Blist or len(Ilist) < 2:
        raise ValueError("Недостаточно данных для обучения модели (нужно >=2 пар I/B).")

    X = [[float(x)] for x in Ilist]
    y = [float(b) for b in Blist]

    if SKLEARN_AVAILABLE:
        model = LinearRegression()
        model.fit(X, y)
        def predictor(arr):
            import numpy as _np
            arr2 = _np.array(arr).reshape(-1, 1)
            return model.predict(arr2).tolist()
        return predictor, model
    else:
        arrX = np.array(Ilist, dtype=float)
        arrY = np.array(Blist, dtype=float)
        coef = np.polyfit(arrX, arrY, 1)  # [a, b] y = a*x + b
        a, b = coef[0], coef[1]
        def predictor(arr):
            return [a * float(v) + b for v in arr]
        # create a dummy model-like object with coef for reporting
        class Dummy:
            def __init__(self, a, b):
                self.coef_ = [a]
                self.intercept_ = b
        return predictor, Dummy(a, b)

# ------------------- GUI -------------------
class VisualizeHard:

    def _setup_keyboard_navigation(self):
        """Универсальная настройка навигации по таблицам (Tab, Enter, стрелки)"""
        
        # Собираем все поля ввода в одну плоскую последовательность
        all_entries = []
        
        # Вторая программа (3 колонки: I, A1, A2)
        for i in range(EXPERIMENT_ROWS):
            all_entries.append(self.exEntries[i][0])  # I
            all_entries.append(self.exEntries[i][1])  # A1
            all_entries.append(self.exEntries[i][2])  # A2
            
        cols_per_row = 3  # I, A1, A2
        
        def find_current_index():
            """Найти индекс текущего активного поля"""
            current = self.master.focus_get()
            if current in all_entries:
                return all_entries.index(current)
            return -1
        
        def move_focus(delta):
            """Переместить фокус на delta позиций"""
            idx = find_current_index()
            if idx >= 0:
                new_idx = (idx + delta) % len(all_entries)
                all_entries[new_idx].focus_set()
        
        def move_vertical(delta_rows):
            """Переместить фокус вертикально на delta_rows строк"""
            idx = find_current_index()
            if idx >= 0:
                # Определяем сколько полей в строке
                current_row = idx // cols_per_row
                current_col = idx % cols_per_row
                new_row = current_row + delta_rows
                total_rows = len(all_entries) // cols_per_row
                
                if 0 <= new_row < total_rows:
                    new_idx = new_row * cols_per_row + current_col
                    all_entries[new_idx].focus_set()
        
        def move_horizontal(delta_cols):
            """Переместить фокус горизонтально между колонками"""
            idx = find_current_index()
            if idx >= 0:
                current_row = idx // cols_per_row
                current_col = idx % cols_per_row
                new_col = current_col + delta_cols
                
                # Проверяем границы колонок
                if 0 <= new_col < cols_per_row:
                    new_idx = current_row * cols_per_row + new_col
                    all_entries[new_idx].focus_set()
                # Если вышли за правую границу, переходим на следующую строку
                elif new_col >= cols_per_row and delta_cols > 0:
                    move_vertical(1)
                    # Переходим к первой колонке
                    new_idx = (current_row + 1) * cols_per_row
                    if new_idx < len(all_entries):
                        all_entries[new_idx].focus_set()
                # Если вышли за левую границу, переходим на предыдущую строку
                elif new_col < 0 and delta_cols < 0:
                    move_vertical(-1)
                    # Переходим к последней колонке
                    new_idx = (current_row - 1) * cols_per_row + (cols_per_row - 1)
                    if new_idx >= 0 and new_idx < len(all_entries):
                        all_entries[new_idx].focus_set()
        
        # Привязываем клавиши ко всем полям ввода
        for entry in all_entries:
            # Tab/Shift+Tab - следующее/предыдущее поле
            entry.bind('<Tab>', lambda e: move_focus(1) or "break")
            entry.bind('<Shift-Tab>', lambda e: move_focus(-1) or "break")
            
            # Enter - как Tab
            entry.bind('<Return>', lambda e: move_focus(1) or "break")
            entry.bind('<KP_Enter>', lambda e: move_focus(1) or "break")
            
            # Стрелки
            entry.bind('<Up>', lambda e: move_vertical(-1) or "break")
            entry.bind('<Down>', lambda e: move_vertical(1) or "break")
            entry.bind('<Left>', lambda e: move_horizontal(-1) or "break")
            entry.bind('<Right>', lambda e: move_horizontal(1) or "break")
            
            # Ctrl+Enter для быстрого построения/вычисления
            entry.bind('<Control-Return>', lambda e: self.on_compute_plot())
            entry.bind('<Control-KP_Enter>', lambda e: self.on_compute_plot())
        
        # Автофокус на первое поле при запуске
        if all_entries:
            all_entries[0].focus_set()

    def __init__(self, master):
        self.master = master
        self.master.title("Calibration Visualizer - Hard Mode")
        self.master.geometry("1280x1080")
        self.master.configure(bg=BG_MAIN)

        self.calibPath = None
        self.predictor = None
        self.modelObj = None

        self._style_ttk()
        self._build_layout()

    def _style_ttk(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background=BG_PANEL, fieldbackground=BG_PANEL, foreground=FG_TEXT, rowheight=20)
        style.configure("Treeview.Heading", background="#333333", foreground=FG_TEXT, relief="flat")

    def _build_layout(self):
        # grid layout
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_rowconfigure(2, weight=0)
        self.master.grid_columnconfigure(0, weight=2)
        self.master.grid_columnconfigure(1, weight=3)

        # panels
        self.frameTopLeft = self._panel(0, 0)
        self.frameTopRight = self._panel(0, 1)
        self.frameBottomLeft = self._panel(1, 0)
        self.frameBottomRight = self._panel(1, 1)

        self._build_top_left()
        self._build_top_right()
        self._build_bottom_tables()
        self._build_footer()

    def _panel(self, r, c):
        f = tk.Frame(self.master, bg=BG_PANEL, highlightbackground="#3a3a3a", highlightthickness=1)
        f.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
        return f

    # ---------- Верхняя левая: ввод EXPERIMENT_ROWS строк (I, A1, A2) + выбор JSON ------------
    def _build_top_left(self):
        tk.Label(self.frameTopLeft, text=f"Ввод экспериментов ({EXPERIMENT_ROWS} строк):", 
                bg=BG_PANEL, fg=FG_TEXT, font=("TkDefaultFont", 11, "bold")).pack(anchor="w", padx=8, pady=6)

        hdr = tk.Frame(self.frameTopLeft, bg=BG_PANEL)
        hdr.pack(anchor="w", padx=8)
        tk.Label(hdr, text="№", width=3, bg=BG_PANEL, fg=FG_TEXT).grid(row=0, column=0)
        tk.Label(hdr, text="I", width=12, bg=BG_PANEL, fg=FG_TEXT).grid(row=0, column=1)
        tk.Label(hdr, text="A_1 (deg)", width=12, bg=BG_PANEL, fg=FG_TEXT).grid(row=0, column=2)
        tk.Label(hdr, text="A_2 (deg)", width=12, bg=BG_PANEL, fg=FG_TEXT).grid(row=0, column=3)

        self.exEntries = []  # list of tuples (eI, eA1, eA2)
        rowsFrame = tk.Frame(self.frameTopLeft, bg=BG_PANEL)
        rowsFrame.pack(anchor="w", padx=8, pady=4)
        for i in range(EXPERIMENT_ROWS):
            row = tk.Frame(rowsFrame, bg=BG_PANEL)
            row.grid(row=i, column=0, sticky="w", pady=2)
            tk.Label(row, text=str(i+1), width=3, bg=BG_PANEL, fg=FG_TEXT).grid(row=0, column=0)
            eI = tk.Entry(row, width=12, bg=ENTRY_BG, fg=FG_TEXT, insertbackground=FG_TEXT, relief="flat")
            eA1 = tk.Entry(row, width=12, bg=ENTRY_BG, fg=FG_TEXT, insertbackground=FG_TEXT, relief="flat")
            eA2 = tk.Entry(row, width=12, bg=ENTRY_BG, fg=FG_TEXT, insertbackground=FG_TEXT, relief="flat")
            eI.grid(row=0, column=1, padx=4)
            eA1.grid(row=0, column=2, padx=4)
            eA2.grid(row=0, column=3, padx=4)
            self.exEntries.append((eI, eA1, eA2))

        # controls: load calib json, kolvo (for genData if needed), run
        controls = tk.Frame(self.frameTopLeft, bg=BG_PANEL)
        controls.pack(anchor="w", padx=8, pady=8, fill="x")

        tk.Button(controls, text="Загрузить калибровочный JSON", command=self.on_load_calib, bg=ACCENT, fg="white", relief="flat").pack(side="left", padx=4)
        self.calibLabel = tk.Label(controls, text="Файл не выбран", bg=BG_PANEL, fg=FG_TEXT)
        self.calibLabel.pack(side="left", padx=8)

        tk.Label(controls, text="kolvo (для genData):", bg=BG_PANEL, fg=FG_TEXT).pack(side="left", padx=(12,4))
        self.entryKolvo = tk.Entry(controls, width=6, bg=ENTRY_BG, fg=FG_TEXT, insertbackground=FG_TEXT, relief="flat")
        self.entryKolvo.insert(0, "50")
        self.entryKolvo.pack(side="left", padx=4)

        tk.Button(controls, text="Вычислить и Построить", command=self.on_compute_plot, bg=ACCENT, fg="white", relief="flat").pack(side="left", padx=12)
        
        self._setup_keyboard_navigation()
    # ---------- Верхняя правая: график φ vs B*L ----------
    def _build_top_right(self):
        self.fig = Figure(figsize=(4.5, 3.5), dpi=100, facecolor=BG_PANEL)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(BG_PANEL)
        self.ax.tick_params(colors=FG_TEXT)
        for s in self.ax.spines.values():
            s.set_color(FG_TEXT)
        self.ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frameTopRight)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # ---------- Низ: две таблицы (лево: вычисления; право: точки графика) ----------
    def _build_bottom_tables(self):
        # left: final table with B, k, phi, V, V_погр
        tk.Label(self.frameBottomLeft, text="Результаты (B, k, φ, V, V_погр)", bg=BG_PANEL, fg=FG_TEXT).pack(anchor="w", padx=6, pady=4)
        self.treeResults = ttk.Treeview(self.frameBottomLeft, columns=("B","k","phi","V","V_погр"), show="headings", height=10)
        for c, w in zip(("B","k","phi","V","V_погр"), (80,80,80,80,100)):
            self.treeResults.heading(c, text=c)
            self.treeResults.column(c, width=w, anchor="center")
        self.treeResults.pack(fill="both", expand=True, padx=6, pady=6)
        
        # Добавляем метку для среднего значения V под таблицей
        self.V_mean_label = tk.Label(self.frameBottomLeft, text="", bg=BG_PANEL, fg=ACCENT, font=("TkDefaultFont", 10, "bold"))
        self.V_mean_label.pack(anchor="w", padx=6, pady=4)

        # right: points for graph (BL, phi)
        tk.Label(self.frameBottomRight, text="Точки для графика (BL, φ)", bg=BG_PANEL, fg=FG_TEXT).pack(anchor="w", padx=6, pady=4)
        self.treePoints = ttk.Treeview(self.frameBottomRight, columns=("BL","phi"), show="headings", height=10)
        self.treePoints.heading("BL", text="B·L")
        self.treePoints.heading("phi", text="φ (rad)")
        self.treePoints.column("BL", width=150, anchor="center")
        self.treePoints.column("phi", width=150, anchor="center")
        self.treePoints.pack(fill="both", expand=True, padx=6, pady=6)

    # ---------- Footer ----------
    def _build_footer(self):
        tk.Label(self.master, text="сделано во благо социализма и тех, кто не хочет учить физику. Слава Советскому Союзу ☭",
                 fg=RED_TEXT, bg=BG_MAIN, font=("TkDefaultFont", 8)).grid(row=2, column=0, columnspan=2, sticky="e", padx=8, pady=4)

    # ------------------- Actions -------------------
    def on_load_calib(self):
        path = filedialog.askopenfilename(title="Выберите калибровочный JSON", filetypes=[("JSON files","*.json"),("All files","*.*")])
        if not path:
            return
        try:
            Ilist, Blist = parse_calib_json(path)
            if len(Ilist) < 2:
                raise ValueError("В файле не найдено достаточно пар I/B.")
            self.calibPath = path
            self.calibLabel.config(text=os.path.basename(path))
            self.predictor, self.modelObj = train_model(Ilist, Blist)
            messagebox.showinfo("ОК", f"Калибровка загружена: {len(Ilist)} точек. Модель обучена.")
        except Exception as e:
            messagebox.showerror("Ошибка при загрузке калибровки", str(e))
            self.calibPath = None
            self.calibLabel.config(text="Файл не выбран")
            self.predictor = None
            self.modelObj = None

    def _read_inputs(self):
        """
        Читает EXPERIMENT_ROWS строк ввода. Возвращает списки Ilist, A1list, A2list (float).
        Валидация: все поля должны быть заполнены и числа.
        """
        Ilist, A1list, A2list = [], [], []
        for idx, (eI, eA1, eA2) in enumerate(self.exEntries, start=1):
            sI = eI.get().strip()
            sA1 = eA1.get().strip()
            sA2 = eA2.get().strip()
            if not sI or not sA1 or not sA2:
                raise ValueError(f"Пустые поля в строке {idx}")
            try:
                Ilist.append(float(sI))
                A1list.append(float(sA1))
                A2list.append(float(sA2))
            except Exception:
                raise ValueError(f"Некорректные числа в строке {idx}")
        return Ilist, A1list, A2list

    def on_compute_plot(self):
        # основной процесс: валидируем, предсказываем B, считаем phi, k, V, строим график, сохраняем
        try:
            Ilist, A1list, A2list = self._read_inputs()
        except Exception as e:
            messagebox.showerror("Ввод", str(e))
            return

        if not self.predictor:
            messagebox.showerror("Калибровка", "Калибровочная модель не загружена. Загрузите JSON и обучите модель.")
            return

        # try kolvo
        try:
            kolvo = int(self.entryKolvo.get().strip())
            if kolvo < 2:
                raise ValueError
        except Exception:
            messagebox.showerror("kolvo", "kolvo должен быть целым >= 2")
            return

        # предсказание B по Ilist
        try:
            Bpred = self.predictor(Ilist)
            Bpred = [float(x) for x in Bpred]
        except Exception as e:
            messagebox.showerror("Model error", f"Ошибка предсказания B: {e}")
            return

        # вычисление phi, BL, V, k
        phi_list = []
        BL_list = []
        V_list = []
        k_list = []
        for a1, a2, b in zip(A1list, A2list, Bpred):
            # Угол Фарадея в радианах
            phi = (abs(a1 - a2) / 2.0) * (math.pi / 180.0)
            
            # B из мТл в Тл, затем B*L
            B_T = b / 1000.0  # мТл → Тл
            BL = B_T * L_CONST  # Тл·м
            
            # Постоянная Верде V = φ / (B*L)
            V = phi / BL if BL != 0 else float("inf")  # рад/(Тл·м)
            
            # k = V (в данной модели)
            k_val = V
            
            phi_list.append(phi)
            BL_list.append(BL)
            V_list.append(V)
            k_list.append(k_val)
        # ===== Статистическая обработка V =====
        import numpy as np
        V_arr = np.array(V_list, dtype=float)
        
        # Вычисляем среднее значение V (истинное значение)
        V_mean = np.mean(V_arr)
        
        # Вычисляем погрешности каждого измерения: |V_i - V_ср|
        V_errors_individual = np.abs(V_arr - V_mean)
        
        # Вычисляем среднюю погрешность (среднее отклонение от среднего)
        V_mean_error = np.mean(V_errors_individual)
        
        # Стандартное отклонение
        V_std = np.std(V_arr, ddof=1)  # ddof=1 для несмещённой оценки
        
        # Погрешность среднего (стандартная ошибка)
        V_mean_std_error = V_std / np.sqrt(len(V_arr))
        
        # ===== КОНЕЦ статистики =====
        
        # На график добавим точки (phi vs BL) и подгоним прямую через 0: phi = slope * BL
        BL_arr = np.array(BL_list, dtype=float)
        phi_arr = np.array(phi_list, dtype=float)

        # линейная аппроксимация через 0:
        if np.allclose(BL_arr, 0):
            slope = 0.0
        else:
            slope = (BL_arr @ phi_arr) / (BL_arr @ BL_arr)  # решение min ||phi - s*BL||, без свободного члена
        # также вычислим обычную линейную регрессию (с intercept) для контроля
        try:
            if SKLEARN_AVAILABLE:
                model_lin = LinearRegression(fit_intercept=True)
                model_lin.fit(BL_arr.reshape(-1,1), phi_arr)
                slope_with_intercept = model_lin.coef_[0]
                intercept = model_lin.intercept_
            else:
                coef = np.polyfit(BL_arr, phi_arr, 1)
                slope_with_intercept, intercept = coef[0], coef[1]
        except Exception:
            slope_with_intercept, intercept = slope, 0.0

        # Заполнение графика
        self.ax.clear()
        self.ax.set_facecolor(BG_PANEL)
        self.ax.scatter(BL_arr, phi_arr, label="Экспериментальные точки", color=ACCENT)
        # линия через ноль
        x_line = np.linspace(BL_arr.min()*0.9, BL_arr.max()*1.1, 200) if len(BL_arr)>0 else np.linspace(0,1,2)
        y_line = slope * x_line
        self.ax.plot(x_line, y_line, label=f"Прямая (через 0), slope={slope:.6g}", color="#ffd166")
        # линия с intercept для контроля
        y_line2 = slope_with_intercept * x_line + intercept
        self.ax.plot(x_line, y_line2, label=f"Регресс. с сдвигом, slope={slope_with_intercept:.6g}", color="#06d6a0", linestyle="--")

        self.ax.set_xlabel("B · L", color=FG_TEXT)
        self.ax.set_ylabel("φ (rad)", color=FG_TEXT)
        self.ax.tick_params(colors=FG_TEXT)
        for s in self.ax.spines.values():
            s.set_color(FG_TEXT)
        self.ax.legend(facecolor=BG_PANEL, edgecolor=FG_TEXT, labelcolor=FG_TEXT)
        self.ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5)
        
        # Добавляем на график информацию о среднем V
        self.ax.text(0.02, 0.98, f'V_mean = {V_mean:.4g} ± {V_mean_std_error:.2g}\nV_средн.погр = {V_mean_error:.2g}',
                    transform=self.ax.transAxes,
                    fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    color='black')
        
        self.canvas.draw()

        # заполним таблицы (теперь с погрешностями V)
        self._fill_results_table(Bpred, k_list, phi_list, V_list, V_errors_individual, V_mean, V_mean_error, V_mean_std_error)
        self._fill_points_table(BL_list, phi_list)

        # сохраним результаты (добавляем статистику)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        pngName = f"phi_vs_BL_{ts}.png"
        jsonName = f"results_{ts}.json"
        try:
            self.fig.savefig(pngName, dpi=150, bbox_inches="tight")
            results_json = {
                "experiment_rows": EXPERIMENT_ROWS,
                "L": L_CONST,
                "kolvo": kolvo,
                "slope_through_zero": slope,
                "slope_with_intercept": slope_with_intercept,
                "intercept": intercept,
                "V_statistics": {
                    "V_mean": float(V_mean),
                    "V_mean_error": float(V_mean_error),
                    "V_std": float(V_std),
                    "V_mean_std_error": float(V_mean_std_error),
                    "V_individual_errors": [float(err) for err in V_errors_individual],
                    "V_relative_error_percent": float((V_mean_std_error / V_mean * 100) if V_mean != 0 else 0)
                },
                "inputs": [{"I": I, "A1": a1, "A2": a2} for I,a1,a2 in zip(Ilist, A1list, A2list)],
                "predicted_B": Bpred,
                "BL": BL_list,
                "phi": phi_list,
                "k": k_list,
                "V": V_list,
            }
            with open(jsonName, "w", encoding="utf-8") as f:
                json.dump(results_json, f, ensure_ascii=False, indent=2)
        except Exception as e:
            messagebox.showwarning("Сохранение", f"Не удалось сохранить файл: {e}")
        else:
            messagebox.showinfo("Готово", f"Построено. PNG: {pngName}, JSON: {jsonName}")

    def _fill_results_table(self, Blist, klist, philist, Vlist, V_errors, V_mean, V_mean_error, V_mean_std_error):
        # очистка
        for r in self.treeResults.get_children():
            self.treeResults.delete(r)
        
        # Заполняем таблицу с результатами измерений
        for i, (b, k, phi, v, v_err) in enumerate(zip(Blist, klist, philist, Vlist, V_errors)):
            self.treeResults.insert("", "end", values=(
                f"{b:.4g}",
                f"{k:.4g}",
                f"{phi:.4g}",
                f"{v:.4g}",
                f"{v_err:.2g}"
            ))
        
        # Добавляем строку со средними значениями
        self.treeResults.insert("", "end", values=(
            "СРЕДНЕЕ",
            "",
            "",
            f"{V_mean:.4g}",
            f"{V_mean_error:.2g}"
        ), tags=('mean_row',))
        
        # Настраиваем тег для строки со средним
        self.treeResults.tag_configure('mean_row', background='#3a3a3a', foreground='white')
        
        # Обновляем метку под таблицей
        self.V_mean_label.config(
            text=f"Постоянная Верде: V = {V_mean:.4g} ± {V_mean_std_error:.2g} (отн. погр.: {V_mean_std_error/V_mean*100:.1f}% если V≠0)"
        )

    def _fill_points_table(self, BLlist, philist):
        for r in self.treePoints.get_children():
            self.treePoints.delete(r)
        for bl, phi in zip(BLlist, philist):
            self.treePoints.insert("", "end", values=(f"{bl:.4g}", f"{phi:.4g}"))


if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizeHard(root)
    root.mainloop()