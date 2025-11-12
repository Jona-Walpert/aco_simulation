import random, math

# Hilfsfunktionen aller möglichen Schritte
dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

def neighbors(pos, grid):
    x,y = pos
    h,w = len(grid), len(grid[0])
    for dx,dy in dirs:
        nx,ny = x+dx, y+dy
        if 0<=nx<h and 0<=ny<w and grid[nx][ny]==0:
            yield (nx,ny)
# Euklidische Distanz
def euclid(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

# ACO-Parameter
alpha, beta = 1.0, 2.0
rho = 0.1
Q = 100.0
ants = 20
iters = 100
tau0 = 1.0

def aco_grid(grid, start, goal):
    h,w = len(grid), len(grid[0])
    tau = {}
    for i in range(h):
        for j in range(w):
            if grid[i][j]==0:
                for nb in neighbors((i,j), grid):
                    tau[((i,j), nb)] = tau0

    best = None
    best_len = float('inf')

    for it in range(iters):
        solutions = []
        for _ in range(ants):
            pos = start
            visited = set([start])
            path = [start]
            steps = 0
            max_steps = h*w*2
            while pos != goal and steps < max_steps:
                nbrs = [n for n in neighbors(pos, grid) if n not in visited]
                if not nbrs:
                    break
                probs = []
                denom = 0.0
                for n in nbrs:
                    t = tau.get((pos,n), tau0)
                    hval = 1.0 / (euclid(n, goal) + 1e-6)
                    val = (t**alpha) * (hval**beta)
                    probs.append((n,val))
                    denom += val
                if denom==0:
                    break
                r = random.random()
                cum = 0.0
                chosen = None
                for n,val in probs:
                    cum += val/denom
                    if r <= cum:
                        chosen = n
                        break
                if chosen is None:
                    chosen = probs[-1][0]
                path.append(chosen)
                visited.add(chosen)
                pos = chosen
                steps += 1
            if pos == goal:
                L = sum(euclid(path[i], path[i+1]) for i in range(len(path)-1))
                solutions.append((path, L))
                if L < best_len:
                    best_len = L
                    best = path

        # Pheromone Verdunsten mit der Zeit
        for k in list(tau.keys()):
            tau[k] *= (1.0 - rho)
            if tau[k] < 1e-8: tau[k] = 1e-8

        
        for path,L in solutions:
            deposit = Q / L
            for i in range(len(path)-1):
                a,b = path[i], path[i+1]
                tau[(a,b)] = tau.get((a,b),0.0) + deposit

    return best, best_len


# Beispiel Grid. 1=Hindernis, 0=freier Weg
grid = [
    [0,0,0,0,0,0,0],
    [0,1,1,1,1,1,0],
    [0,0,0,0,0,1,0],
    [0,1,1,1,0,1,0],
    [0,0,0,0,0,0,0]
]

start = (0,0)
goal  = (4,6)

best_path, best_len = aco_grid(grid, start, goal)

print("Beste gefundene Pfadlaenge:", best_len)
print("Pfad:")
print(best_path)
