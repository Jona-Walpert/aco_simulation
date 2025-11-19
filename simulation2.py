import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

# ============================================================
#                    PARAMETER – ÜBERSICHT
# ============================================================
num_circles = 20           # Anzahl der Knoten (Kreise)
num_ants = 50            # Anzahl der Ameisen pro Iteration
iterations = 100          # Max. Iterationen der ACO-Simulation
area_size = 10.0         # Größe der Zeichenfläche (Quadrat 0..area_size)

alpha = 1.0              # Einfluss der Pheromone
beta = 2               # Einfluss der Distanz (größer = stärkerer Fokus auf kurze Wege)
evaporation = 0.06       # Wieviel Pheromon pro Iteration verdampft
pheromone_init = 0.05    # Anfangspheromon pro Kantenpaar
pheromone_min = 0.01     # Minimales Pheromonlevel (verhindert komplettes Aussterben)
q = 100.0                # Menge an Pheromon pro Ameise

min_distance = 0.8       # Minimaler Abstand zwischen Knoten
speed_factor = 2       # >1 = schneller, <1 = langsamer (steuert Frame-Interval)
stagnation_limit = 50    # Abbruchkriterium: Iterationen ohne Verbesserung
# ============================================================


class AntColonyOptimization:
    def __init__(self):
        self.num_circles = num_circles
        self.num_ants = num_ants
        self.iterations = iterations
        self.area_size = area_size

        self.pheromone = np.ones((self.num_circles, self.num_circles)) * pheromone_init

        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.q = q
        self.pheromone_min = pheromone_min

        self.best_path_length = float('inf')
        self.best_path = None
        self.paths = []
        self.no_improve_counter = 0

        self.circles = self.generate_spaced_points(self.num_circles, self.area_size, min_distance)

    def generate_spaced_points(self, count, area, min_dist):
        points = []
        attempts = 0
        max_attempts = 10000

        while len(points) < count and attempts < max_attempts:
            attempts += 1
            candidate = np.random.rand(2) * area
            ok = True
            for p in points:
                if np.linalg.norm(candidate - p) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(candidate)

        if len(points) < count:
            print("WARNUNG: konnte nicht genug Punkte mit Mindestabstand erzeugen — fülle den Rest zufällig auf.")
            while len(points) < count:
                points.append(np.random.rand(2) * area)

        return np.array(points)

    def distance(self, i, j):
        return np.linalg.norm(self.circles[i] - self.circles[j])

    def run_iteration(self):
        ant_paths = []
        improved = False

        for _ in range(self.num_ants):
            path = self.construct_path()
            ant_paths.append(path)
            length = self.calculate_path_length(path)

            if length < self.best_path_length:
                self.best_path_length = length
                self.best_path = path.copy()
                improved = True

            self.deposit_pheromone(path, length)

        self.pheromone *= (1 - self.evaporation)
        self.pheromone = np.maximum(self.pheromone, self.pheromone_min)

        if improved:
            self.no_improve_counter = 0
        else:
            self.no_improve_counter += 1

        self.paths = ant_paths

    def construct_path(self):
        unvisited = set(range(self.num_circles))
        current = random.randint(0, self.num_circles - 1)
        path = [current]
        unvisited.remove(current)

        while unvisited:
            nxt = self.select_next(current, unvisited)
            path.append(nxt)
            unvisited.remove(nxt)
            current = nxt

        return path

    def select_next(self, current, unvisited):
        unvisited_list = list(unvisited)
        probs = []

        for node in unvisited_list:
            tau = self.pheromone[current, node] ** self.alpha
            eta = 1.0 / (self.distance(current, node) + 1e-8)
            probs.append(tau * (eta ** self.beta))

        s = sum(probs)
        if s <= 0:
            return random.choice(unvisited_list)

        probs = np.array(probs) / s
        idx = np.random.choice(len(unvisited_list), p=probs)
        return unvisited_list[idx]

    def calculate_path_length(self, path):
        length = 0.0
        for i in range(len(path) - 1):
            length += self.distance(path[i], path[i+1])
        length += self.distance(path[-1], path[0])
        return length

    def deposit_pheromone(self, path, length):
        if length <= 0:
            return
        dep = self.q / length
        for i in range(len(path) - 1):
            a, b = path[i], path[i+1]
            self.pheromone[a, b] += dep
            self.pheromone[b, a] += dep
        a, b = path[-1], path[0]
        self.pheromone[a, b] += dep
        self.pheromone[b, a] += dep


# ============================================================
#                    ANIMATION TEIL
# ============================================================
def start_animation():
    aco = AntColonyOptimization()
    frame_interval = max(1, int(150 / speed_factor))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-1, area_size + 1)
    ax.set_ylim(-1, area_size + 1)
    ax.set_aspect('equal')

    for (x, y) in aco.circles:
        ax.add_patch(Circle((x, y), 0.28, color="red", zorder=4))

    for i, (x, y) in enumerate(aco.circles):
        ax.text(x, y, str(i), color="white", ha="center", va="center", fontsize=8, zorder=5)

    info_text = ax.text(0.01, -0.06, '', transform=ax.transAxes, va='top', fontsize=9)

    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Pheromone (dunkler = stärker)'),
        Line2D([0], [0], color='blue', lw=1, alpha=0.5, label='Ant paths'),
        Line2D([0], [0], color='green', lw=2, label='Best path'),
        Line2D([0], [0], color='red', lw=4, label='Final Best Path')
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    pheromone_lines = []
    ant_path_lines = []
    best_path_lines = []

    def draw_final_best():
        if aco.best_path is not None:
            bp = aco.best_path
            xs = [aco.circles[n][0] for n in bp] + [aco.circles[bp[0]][0]]
            ys = [aco.circles[n][1] for n in bp] + [aco.circles[bp[0]][1]]
            ax.plot(xs, ys, color="red", linewidth=4, zorder=10, label="Final Best Path")
            for (x, y) in aco.circles:
                ax.add_patch(Circle((x, y), 0.28, color="red", zorder=11))
            for i, (x, y) in enumerate(aco.circles):
                ax.text(x, y, str(i), color="white", ha="center", va="center", fontsize=8, zorder=12)
            plt.draw()

    def animate(frame):
        # Abbruchkriterium: Stagnation
        if aco.no_improve_counter >= stagnation_limit:
            print("Frühzeitig beendet: keine Verbesserung nach", stagnation_limit, "Iterationen")
            draw_final_best()
            ani.event_source.stop()
            return

        # Beim letzten Frame auch: best path hervorheben
        if frame == iterations - 1:
            draw_final_best()

        aco.run_iteration()

        for ln in pheromone_lines + ant_path_lines + best_path_lines:
            try:
                ln.remove()
            except Exception:
                pass
        pheromone_lines.clear()
        ant_path_lines.clear()
        best_path_lines.clear()

        max_ph = np.max(aco.pheromone)
        if max_ph <= 0:
            max_ph = 1.0

        for i in range(aco.num_circles):
            for j in range(i+1, aco.num_circles):
                ph = aco.pheromone[i, j]
                alpha_v = (ph - pheromone_min) / (max_ph - pheromone_min + 1e-12)
                alpha_v = np.clip(alpha_v, 0.01, 1.0)
                lw = 0.5 + alpha_v * 3.0
                x1, y1 = aco.circles[i]
                x2, y2 = aco.circles[j]
                ln, = ax.plot([x1, x2], [y1, y2],
                              alpha=0.9 * alpha_v, linewidth=lw, color='black', zorder=1)
                pheromone_lines.append(ln)

        for path in aco.paths:
            xs = [aco.circles[n][0] for n in path] + [aco.circles[path[0]][0]]
            ys = [aco.circles[n][1] for n in path] + [aco.circles[path[0]][1]]
            ln, = ax.plot(xs, ys, color="blue", alpha=0.08, linewidth=0.8, zorder=2)
            ant_path_lines.append(ln)

        if aco.best_path is not None:
            bp = aco.best_path
            xs = [aco.circles[n][0] for n in bp] + [aco.circles[bp[0]][0]]
            ys = [aco.circles[n][1] for n in bp] + [aco.circles[bp[0]][1]]
            ln, = ax.plot(xs, ys, color="green", linewidth=2.5, zorder=3)
            best_path_lines.append(ln)

        ax.set_title(
            f"Iteration {frame+1}/{iterations} | Best distance: {aco.best_path_length:.3f}"
        )

        info_text.set_text(
            f"Nodes: {num_circles}   Ants: {num_ants}   "
            f"Evaporation: {evaporation:.3f}   No-Improve: {aco.no_improve_counter}"
        )

    ani = FuncAnimation(fig, animate, frames=iterations, interval=frame_interval, repeat=False)
    plt.show()


if __name__ == "__main__":
    start_animation()
