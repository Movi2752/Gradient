"""
Градиент и градиентный спуск: визуализация на примере f(x,y) = x² + 10y².

Содержит 4 субплота:
  1. 3D-поверхность с градиентами и антиградиентами в ключевых точках.
  2. Линии уровня с полем градиентов (показывает перпендикулярность).
  3. Траектории градиентного спуска при подходящем η.
  4. Траектория при большом η (осцилляции / расхождение).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


# === Функция и градиент ===

def f(x, y):
    """f(x, y) = x² + 10y² — функция с плохой обусловленностью (κ = 10)."""
    return x**2 + 10 * y**2


def grad_f(point):
    """∇f = (2x, 20y)."""
    x, y = point
    return np.array([2 * x, 20 * y])


# === Градиентный спуск ===

def gradient_descent(start, lr, max_iter=80, tol=1e-6):
    """
    Базовый градиентный спуск.

    Возвращает массив точек траектории shape (T, 2).
    Останавливается при ||g|| < tol или если улетели в бесконечность.
    """
    x = np.array(start, dtype=float)
    history = [x.copy()]
    for _ in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
        x = x - lr * g
        history.append(x.copy())
        if np.linalg.norm(x) > 100:  # расходится
            break
    return np.array(history)


# === Класс для 3D-стрелок ===
# (matplotlib quiver в 3D рисует стрелки криво; этот класс лучше выглядит)

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


# === Сетка для всех графиков ===

x_vals = np.linspace(-3, 3, 60)
y_vals = np.linspace(-2, 2, 60)
X, Y = np.meshgrid(x_vals, y_vals)
Z = f(X, Y)


# === Фигура ===

fig = plt.figure(figsize=(15, 10))
fig.suptitle(r"$f(x, y) = x^2 + 10y^2$ — градиент и градиентный спуск",
             fontsize=15, y=0.995)


# --- Субплот 1: 3D-поверхность с градиентами ---

ax1 = fig.add_subplot(2, 2, 1, projection='3d')
ax1.plot_surface(X, Y, Z, alpha=0.55, cmap=cm.plasma, edgecolor='none')
ax1.contour(X, Y, Z, zdir='z', offset=0, cmap='plasma', alpha=0.4, levels=8)

ref_points = [(3.0, 0.0), (0.0, 1.5), (2.0, 1.0)]
grad_norms = [np.linalg.norm(grad_f(p)) for p in ref_points]
arrow_scale = 1.5 / max(grad_norms)

for (x0, y0) in ref_points:
    z0 = f(x0, y0)
    gx, gy = grad_f((x0, y0))
    dx, dy = gx * arrow_scale, gy * arrow_scale

    # Красный — градиент
    arr_g = Arrow3D([x0, x0 + dx], [y0, y0 + dy], [z0, z0],
                    mutation_scale=14, lw=2, color='#ef4444',
                    arrowstyle='-|>')
    ax1.add_artist(arr_g)
    # Синий — антиградиент
    arr_a = Arrow3D([x0, x0 - dx], [y0, y0 - dy], [z0, z0],
                    mutation_scale=14, lw=2, color='#3b82f6',
                    arrowstyle='-|>')
    ax1.add_artist(arr_a)
    ax1.scatter([x0], [y0], [z0], color='#f59e0b', s=30, zorder=10)
    ax1.text(x0, y0, z0 + 2,
             f'({x0},{y0})\n|∇f|={np.hypot(gx, gy):.1f}',
             fontsize=8, ha='center')

ax1.set_xlabel('x'); ax1.set_ylabel('y'); ax1.set_zlabel('f')
ax1.set_title('Поверхность и градиенты в ключевых точках', fontsize=11)
# Легенда вручную
from matplotlib.lines import Line2D
ax1.legend(handles=[
    Line2D([0], [0], color='#ef4444', lw=2, label='∇f (рост)'),
    Line2D([0], [0], color='#3b82f6', lw=2, label='-∇f (спуск)'),
], loc='upper left', fontsize=9)


# --- Субплот 2: контуры + поле градиентов ---

ax2 = fig.add_subplot(2, 2, 2)
cs = ax2.contour(X, Y, Z, levels=12, cmap='plasma', alpha=0.6)
ax2.clabel(cs, inline=True, fontsize=7, fmt='%.1f')

# Поле градиентов (нормированные стрелки, цвет = реальная длина)
step = 6
Xs, Ys = X[::step, ::step], Y[::step, ::step]
U, V = 2 * Xs, 20 * Ys
norms = np.hypot(U, V)
norms_safe = np.where(norms == 0, 1, norms)
U_n, V_n = U / norms_safe, V / norms_safe

q = ax2.quiver(Xs, Ys, U_n, V_n, norms,
               cmap='viridis', alpha=0.85, scale=30, width=0.004)
plt.colorbar(q, ax=ax2, label='|∇f|', fraction=0.046, pad=0.04)
ax2.plot(0, 0, '*', color='#ef4444', markersize=18, zorder=10,
         markeredgecolor='white', markeredgewidth=1.2, label='минимум')
ax2.set_xlabel('x'); ax2.set_ylabel('y')
ax2.set_title('Линии уровня и поле градиентов\n(∇f перпендикулярен линиям уровня)',
              fontsize=11)
ax2.set_aspect('equal')
ax2.legend(loc='upper left', fontsize=9)


# --- Субплот 3: сходящиеся траектории ---

ax3 = fig.add_subplot(2, 2, 3)
ax3.contour(X, Y, Z, levels=12, cmap='plasma', alpha=0.35)

lr_good = 0.05
starts = [(2.8, 1.5), (-2.5, -1.5), (2.5, -1.2), (-2.8, 1.7)]
colors = ['#10b981', '#06b6d4', '#8b5cf6', '#f59e0b']

for start, c in zip(starts, colors):
    path = gradient_descent(start, lr=lr_good, max_iter=80)
    ax3.plot(path[:, 0], path[:, 1], 'o-',
             markersize=3, linewidth=1.3, color=c, alpha=0.85,
             label=f'{start} → {len(path)-1} шагов')
    ax3.plot(start[0], start[1], 's', markersize=9,
             color=c, markeredgecolor='white', markeredgewidth=1.2)

ax3.plot(0, 0, '*', color='red', markersize=18, zorder=10,
         markeredgecolor='white', markeredgewidth=1.2)
ax3.set_xlabel('x'); ax3.set_ylabel('y')
ax3.set_title(f'Сходимость: η = {lr_good} (зигзаги из-за κ=10)', fontsize=11)
ax3.set_aspect('equal')
ax3.legend(loc='upper left', fontsize=8)
ax3.set_xlim(-3.2, 3.2); ax3.set_ylim(-2, 2)


# --- Субплот 4: расходимость при большом η ---

ax4 = fig.add_subplot(2, 2, 4)
ax4.contour(X, Y, Z, levels=12, cmap='plasma', alpha=0.35)

# η = 0.1 — ровно граница сходимости (λ_max = 20, граница 2/20 = 0.1)
# Берём чуть больше, чтобы видна была расходимость
for lr, color, label in [(0.095, '#10b981', 'η=0.095 (медленно)'),
                          (0.105, '#ef4444', 'η=0.105 (расходится)')]:
    path = gradient_descent((1.0, 0.5), lr=lr, max_iter=25)
    ax4.plot(path[:, 0], path[:, 1], 'o-',
             markersize=3, linewidth=1.3, color=color, alpha=0.85,
             label=label)
    ax4.plot(path[0, 0], path[0, 1], 's', markersize=9,
             color=color, markeredgecolor='white', markeredgewidth=1.2)

ax4.plot(0, 0, '*', color='black', markersize=18, zorder=10,
         markeredgecolor='white', markeredgewidth=1.2)
ax4.set_xlabel('x'); ax4.set_ylabel('y')
ax4.set_title('Граница сходимости: η < 2/λ_max = 0.1', fontsize=11)
ax4.set_aspect('equal')
ax4.legend(loc='upper left', fontsize=9)
ax4.set_xlim(-3.2, 3.2); ax4.set_ylim(-2, 2)


plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/gradient_full.png', dpi=130, bbox_inches='tight')
plt.show()
