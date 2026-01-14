#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import json
from datetime import datetime
import math
import os

# ---------- ЦВЕТА / СТИЛЬ ----------
BG_MAIN = "#1e1e1e"
BG_PANEL = "#252526"
FG_TEXT = "#d4d4d4"
ACCENT = "#3c7dd9"
ENTRY_BG = "#2d2d2d"
GRID_COLOR = "#444444"
RED_TEXT = "#cc3333"

ROWS = 9  # количество строк в каждой таблице (I,B)

class VisualizeCalib:
    def __init__(self, master):
        self.master = master
        self.master.title("Calibration Visualizer")
        self.master.geometry("1280x1080")
        self.master.configure(bg=BG_MAIN)

        self._style_ttk()
        self._build_layout()

    def _style_ttk(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background=BG_PANEL,
            fieldbackground=BG_PANEL,
            foreground=FG_TEXT,
            rowheight=20,
            borderwidth=0,
        )
        style.configure(
            "Treeview.Heading",
            background="#333333",
            foreground=FG_TEXT,
            relief="flat",
        )
    def _setup_tab_order(self):
        """Настройка полной навигации по таблицам (Tab, Enter, стрелки)"""

        # Собираем все поля ввода в одну плоскую последовательность
        # Порядок: все I и B из 20mm, затем все I и B из 40mm
        all_entries = []
        
        for i in range(ROWS):
            all_entries.append(self.table20Entries[i][0])  # I 20mm
            all_entries.append(self.table20Entries[i][1])  # B 20mm
        
        for i in range(ROWS):
            all_entries.append(self.table40Entries[i][0])  # I 40mm
            all_entries.append(self.table40Entries[i][1])  # B 40mm
        
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
                # В каждой строке 2 поля (I и B)
                current_row = idx // 2
                current_col = idx % 2
                new_row = current_row + delta_rows
                if 0 <= new_row < ROWS * 2:  # Всего ROWS*2 строк (2 таблицы)
                    new_idx = new_row * 2 + current_col
                    all_entries[new_idx].focus_set()
        
        def move_horizontal(delta_cols):
            """Переместить фокус горизонтально между I и B"""
            idx = find_current_index()
            if idx >= 0:
                current_row = idx // 2
                current_col = idx % 2
                new_col = current_col + delta_cols
                if 0 <= new_col < 2:
                    new_idx = current_row * 2 + new_col
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
            
            # Ctrl+Enter для быстрого построения
            entry.bind('<Control-Return>', lambda e: self.on_plot())
            entry.bind('<Control-KP_Enter>', lambda e: self.on_plot())
        
        # Автофокус на первое поле при запуске
        if all_entries:
            all_entries[0].focus_set()
    def _build_layout(self):
        # сетка 2x2 + футер
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_rowconfigure(1, weight=1)
        self.master.grid_rowconfigure(2, weight=0)
        self.master.grid_columnconfigure(0, weight=2)
        self.master.grid_columnconfigure(1, weight=3)

        # панели
        self.frameTopLeft = self._panel(0, 0)
        self.frameTopRight = self._panel(0, 1)
        self.frameBottomLeft = self._panel(1, 0)
        self.frameBottomRight = self._panel(1, 1)

        # компоненты
        self._build_top_left()
        self._build_top_right()
        self._build_bottom_tables()
        self._build_footer()

    def _panel(self, r, c):
        f = tk.Frame(self.master, bg=BG_PANEL, highlightbackground="#3a3a3a", highlightthickness=1)
        f.grid(row=r, column=c, sticky="nsew", padx=6, pady=6)
        return f

    # ---------- Верхняя левая: две таблицы 7x2 (20mm и 40mm) ----------
    def _build_top_left(self):
        tk.Label(
            self.frameTopLeft,
            text="Калибровочные данные (введите вручную)",
            bg=BG_PANEL,
            fg=FG_TEXT,
            font=("TkDefaultFont", 11, "bold"),
        ).pack(anchor="w", padx=8, pady=6)

        tablesFrame = tk.Frame(self.frameTopLeft, bg=BG_PANEL)
        tablesFrame.pack(anchor="w", padx=8, pady=6, fill="x")

        # 20 мм
        leftBox = tk.Frame(tablesFrame, bg=BG_PANEL)
        leftBox.pack(side="left", anchor="n", padx=(0, 16))
        tk.Label(leftBox, text="20 мм", bg=BG_PANEL, fg=ACCENT).grid(row=0, column=1, columnspan=2)
        tk.Label(leftBox, text="", bg=BG_PANEL).grid(row=1, column=0)
        tk.Label(leftBox, text="I", bg=BG_PANEL, fg=FG_TEXT).grid(row=1, column=1)
        tk.Label(leftBox, text="B", bg=BG_PANEL, fg=FG_TEXT).grid(row=1, column=2)

        self.table20Entries = []
        for r in range(ROWS):
            tk.Label(leftBox, text=str(r + 1), bg=BG_PANEL, fg=FG_TEXT).grid(row=r+2, column=0)
            eI = tk.Entry(leftBox, width=12, bg=ENTRY_BG, fg=FG_TEXT, insertbackground=FG_TEXT, relief="flat")
            eB = tk.Entry(leftBox, width=12, bg=ENTRY_BG, fg=FG_TEXT, insertbackground=FG_TEXT, relief="flat")
            eI.grid(row=r+2, column=1, padx=4, pady=3)
            eB.grid(row=r+2, column=2, padx=4, pady=3)
            self.table20Entries.append((eI, eB))

        # 40 мм
        rightBox = tk.Frame(tablesFrame, bg=BG_PANEL)
        rightBox.pack(side="left", anchor="n")
        tk.Label(rightBox, text="40 мм", bg=BG_PANEL, fg=ACCENT).grid(row=0, column=1, columnspan=2)
        tk.Label(rightBox, text="", bg=BG_PANEL).grid(row=1, column=0)
        tk.Label(rightBox, text="I", bg=BG_PANEL, fg=FG_TEXT).grid(row=1, column=1)
        tk.Label(rightBox, text="B", bg=BG_PANEL, fg=FG_TEXT).grid(row=1, column=2)

        self.table40Entries = []
        for r in range(ROWS):
            tk.Label(rightBox, text=str(r + 1), bg=BG_PANEL, fg=FG_TEXT).grid(row=r+2, column=0)
            eI = tk.Entry(rightBox, width=12, bg=ENTRY_BG, fg=FG_TEXT, insertbackground=FG_TEXT, relief="flat")
            eB = tk.Entry(rightBox, width=12, bg=ENTRY_BG, fg=FG_TEXT, insertbackground=FG_TEXT, relief="flat")
            eI.grid(row=r+2, column=1, padx=4, pady=3)
            eB.grid(row=r+2, column=2, padx=4, pady=3)
            self.table40Entries.append((eI, eB))

        # controls: кнопка Построить
        controls = tk.Frame(self.frameTopLeft, bg=BG_PANEL)
        controls.pack(anchor="w", padx=8, pady=10, fill="x")
        tk.Button(
            controls,
            text="Построить B(I) и сохранить",
            command=self.on_plot,
            bg=ACCENT,
            fg="white",
            relief="flat",
            padx=12,
        ).pack(side="left", padx=6)

        # Короткая подсказка справа
        tk.Label(controls, text=f"Ввод: {ROWS} строк в каждой таблице", bg=BG_PANEL, fg=FG_TEXT).pack(side="left", padx=12)
        self._setup_tab_order()
    # ---------- Верхняя правая: график ----------
    def _build_top_right(self):
        self.fig = Figure(figsize=(4.5, 3.5), dpi=100, facecolor=BG_PANEL)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(BG_PANEL)
        self.ax.tick_params(colors=FG_TEXT)
        for s in self.ax.spines.values():
            s.set_color(FG_TEXT)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frameTopRight)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)

    # ---------- Низ: две таблицы (показать введённые данные) ----------
    def _build_bottom_tables(self):
        tk.Label(self.frameBottomLeft, text="Введённые данные — 20 мм (I, B)", bg=BG_PANEL, fg=FG_TEXT).pack(anchor="w", padx=6, pady=4)
        self.tree20 = ttk.Treeview(self.frameBottomLeft, columns=("I", "B"), show="headings", height=10)
        self.tree20.heading("I", text="I")
        self.tree20.heading("B", text="B")
        self.tree20.pack(fill="both", expand=True, padx=6, pady=6)

        tk.Label(self.frameBottomRight, text="Введённые данные — 40 мм (I, B)", bg=BG_PANEL, fg=FG_TEXT).pack(anchor="w", padx=6, pady=4)
        self.tree40 = ttk.Treeview(self.frameBottomRight, columns=("I", "B"), show="headings", height=10)
        self.tree40.heading("I", text="I")
        self.tree40.heading("B", text="B")
        self.tree40.pack(fill="both", expand=True, padx=6, pady=6)

    # ---------- Футер ----------
    def _build_footer(self):
        tk.Label(
            self.master,
            text="сделано во благо социализма и тех, кто не хочет учить физику. Слава Советскому Союзу ☭",
            fg=RED_TEXT,
            bg=BG_MAIN,
            font=("TkDefaultFont", 8),
        ).grid(row=2, column=0, columnspan=2, sticky="e", padx=8, pady=4)

    # ---------- Логика ----------
    def _read_table_entries(self, entries):
        """
        Читает список tuple(entryI, entryB) длины ROWS и возвращает списки Ilist, Blist (float).
        Бросает ValueError при неверных/пустых значениях.
        """
        Ilist = []
        Blist = []
        for idx, (eI, eB) in enumerate(entries, start=1):
            sI = eI.get().strip()
            sB = eB.get().strip()
            if sI == "" or sB == "":
                raise ValueError(f"Пустая ячейка в строке {idx}")
            try:
                Ival = float(sI)
                Bval = float(sB)
            except Exception:
                raise ValueError(f"Некорректное число в строке {idx}")
            Ilist.append(Ival)
            Blist.append(Bval)
        return Ilist, Blist

    def on_plot(self):
        # читаем обе таблицы
        try:
            I20, B20 = self._read_table_entries(self.table20Entries)
            I40, B40 = self._read_table_entries(self.table40Entries)
        except Exception as e:
            messagebox.showerror("Ошибка ввода", str(e))
            return

        # сортируем по I чтобы график был корректным (если пользователь ввёл не по возр.)
        def sort_pair(Ilist, Blist):
            pairs = sorted(zip(Ilist, Blist), key=lambda x: x[0])
            I_s, B_s = zip(*pairs)
            return list(I_s), list(B_s)

        I20_s, B20_s = sort_pair(I20, B20)
        I40_s, B40_s = sort_pair(I40, B40)

        # очищаем и строим график
        self.ax.clear()
        self.ax.set_facecolor(BG_PANEL)
        self.ax.plot(I20_s, B20_s, label="20 мм", marker="o", linestyle="-", linewidth=1.3)
        self.ax.plot(I40_s, B40_s, label="40 мм", marker="o", linestyle="-", linewidth=1.3)
        self.ax.set_xlabel("I", color=FG_TEXT)
        self.ax.set_ylabel("B", color=FG_TEXT)
        self.ax.tick_params(colors=FG_TEXT)
        for s in self.ax.spines.values():
            s.set_color(FG_TEXT)
        self.ax.legend(facecolor=BG_PANEL, edgecolor=FG_TEXT, labelcolor=FG_TEXT)
        self.ax.grid(True, color=GRID_COLOR, linestyle="--", linewidth=0.5)
        self.canvas.draw()

        # заполняем нижние таблицы
        self._fill_tree(self.tree20, I20_s, B20_s)
        self._fill_tree(self.tree40, I40_s, B40_s)

        # сохраняем PNG и JSON
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        png_name = f"calibration_graph_{ts}.png"
        json_name = f"calibration_data_{ts}.json"
        try:
            # сохранить картинку
            self.fig.savefig(png_name, dpi=150, bbox_inches="tight")

            # подготовить JSON
            out = {
                "timestamp": ts,
                "rows": ROWS,
                "20mm": {
                    "i": I20_s,
                    "b": B20_s
                },
                "40mm": {
                    "i": I40_s,
                    "b": B40_s
                }
            }
            with open(json_name, "w", encoding="utf-8") as f:
                json.dump(out, f, ensure_ascii=False, indent=2)

        except Exception as e:
            messagebox.showwarning("Сохранение", f"Не удалось сохранить файлы: {e}")
            return

        messagebox.showinfo("Готово", f"График и данные сохранены:\n{png_name}\n{json_name}")

    def _fill_tree(self, tree, Ilist, Blist):
        tree.delete(*tree.get_children())
        for i, b in zip(Ilist, Blist):
            tree.insert("", "end", values=(round(i, 6), round(b, 6)))


if __name__ == "__main__":
    root = tk.Tk()
    app = VisualizeCalib(root)
    root.mainloop()
    print("""
        Данная установка может не работать, официальный 
        сайт установки: https://www.shivsons.com/product/faraday-effect-experiment/ 
        (на этом же сайте есть информация и по другим установкам).
        Данные с вероятностью в 70% выйдут крайне корявые, Если так будет, то это не ваша вина, а вина установки, 
        она ехала через весь суперматерик, и приехала явно неисправной.
        И кстати, на столах у окон 1.441 аудитории не используйте установку, электромагнит работает крайне плохо, и 
        эффекта от воздействия магнита на поляризированный свет НЕ БУДЕТ. мощности в розетках стола НЕ ХВАТАЕТ
        Удачи с лабораторной! если надо - в релизе будет 1 из отчетов по данной лабе для примера
    """)
