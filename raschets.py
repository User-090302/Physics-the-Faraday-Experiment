import json
import math
import numpy as np

with open('results.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

L = 0.02

print("ВСЕ ФОРМУЛЫ С ПОДСТАНОВКАМИ:")
print("="*50)

# 1. Расчёт для каждой точки
for i in range(9):
    I = data["inputs"][i]["I"]
    A1 = data["inputs"][i]["A1"]
    A2 = data["inputs"][i]["A2"]
    B_mT = data["predicted_B"][i]
    
    # φ в рад
    phi_deg = abs(A1 - A2) / 2
    phi_rad = phi_deg * (math.pi / 180)
    
    # B·L
    B_T = B_mT / 1000
    BL = B_T * L
    
    # V
    V = phi_rad / BL
    
    print(f"\n{i+1}. I={I}A: 2φ=|{A1}-{A2}|={abs(A1-A2)}°, φ={phi_deg}°={phi_rad:.4f}рад")
    print(f"   B={B_mT:.1f}мТл={B_T:.4f}Тл, B·L={BL:.6f}Тл·м")
    print(f"   V=φ/(B·L)={phi_rad:.4f}/{BL:.6f}={V:.1f}рад/(Тл·м)")

# 2. Среднее V
V_list = [data["V"][i] for i in range(9)]
V_mean = np.mean(V_list)
V_std = np.std(V_list, ddof=1)

print(f"\n\nСРЕДНЕЕ И ПОГРЕШНОСТИ:")
print("="*50)
print(f"V₁={V_list[0]:.1f}, V₂={V_list[1]:.1f}, V₃={V_list[2]:.1f}, V₄={V_list[3]:.1f}, V₅={V_list[4]:.1f}")
print(f"V₆={V_list[5]:.1f}, V₇={V_list[6]:.1f}, V₈={V_list[7]:.1f}, V₉={V_list[8]:.1f}")
print(f"\nV_ср = (V₁+V₂+...+V₉)/9 = {sum(V_list):.1f}/9 = {V_mean:.1f} рад/(Тл·м)")
print(f"Стандартное отклонение σ = √(Σ(Vᵢ-V_ср)²/(n-1)) = {V_std:.1f} рад/(Тл·м)")
print(f"Погрешность среднего σ/√n = {V_std:.1f}/√9 = {V_std/3:.1f} рад/(Тл·м)")

# 3. Истинное значение (с поправкой)
V_table = 22.0
B_typical = 250  # мТл при 5А
B_our = data["predicted_B"][8]  # при 5А
k = B_typical / B_our

print(f"\n\nИСТИННОЕ ЗНАЧЕНИЕ:")
print("="*50)
print(f"При I=5A: наше B={B_our:.1f}мТл, типичное B={B_typical}мТл")
print(f"Коэффициент занижения k = {B_typical}/{B_our:.1f} = {k:.2f}")
print(f"V_испр = V_ср / k = {V_mean:.1f}/{k:.2f} = {V_mean/k:.1f} рад/(Тл·м)")
print(f"Табличное V = {V_table} рад/(Тл·м)")
print(f"Совпадение: {abs(V_mean/k - V_table)/V_table*100:.1f}%")

# 4. График (уравнение)
print(f"\n\nУРАВНЕНИЕ ГРАФИКА:")
print("="*50)
slope = data["slope_through_zero"]
print(f"φ = slope × (B·L)")
print(f"slope = {slope:.1f} рад/(Тл·м) (из графика)")
print(f"V_граф = {slope:.1f} рад/(Тл·м)")

# 5. Проверка
print(f"\n\nПРОВЕРКА:")
print("="*50)
print(f"V_ср = {V_mean:.1f}")
print(f"V_граф = {slope:.1f}")
print(f"Разница: {abs(V_mean - slope)/V_mean*100:.1f}%")



input("Для закрытия нажмите на любую клавишу:")