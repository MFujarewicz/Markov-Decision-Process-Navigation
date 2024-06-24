import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random

N = 15
WALL_START_PROBABILITY = 0.9
WALL_STOP_PROBABILITY = 0.1


def main():
    #
    walls = generateWalls()
    generate_exit(walls)

    print(np.matrix(walls))


    printWalls(walls)


def generate_exit(walls):
    border_coords = border_coordinates()

    while True:
        exit = random.choice(border_coords)
        if walls[exit[0], exit[1]] == 0:
            walls[exit[0], exit[1]] = 2
            break


def border_coordinates():
    border_coords = []
    for i in range(N):
        for j in range(N):
            if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                border_coords.append((i, j))
    return border_coords

def generateWalls():
    def areAnyWallsAround(i, j, direction=(0, 0)):
        directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < N and 0 <= nj < N:
                if -di == direction[0] and -dj == direction[1]:
                    continue
                if walls[ni, nj] == 1:
                    return True
        return False

    def isBorder(i, j):
        return i == 0 or i == N - 1 or j == 0 or j == N - 1

    def isOutOfBounds(i, j):
        return i < 0 or i >= N or j < 0 or j >= N

    walls = np.zeros((N, N))  # wall = 1

    points = [(random.randint(0, N - 1), random.randint(0, N - 1)) for _ in range(N * N)]
    for i, j in points:

        if walls[i, j] == 1:
            continue

        if areAnyWallsAround(i, j):
            continue

        borderStart = isBorder(i, j)

        if random.random() < WALL_START_PROBABILITY:
            walls[i, j] = 1

            direction = [0, 0]
            for x in range(10):
                direction = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                wi = i + direction[0]
                wj = j + direction[1]
                if not areAnyWallsAround(wi, wj, direction):
                    break

            wi = i
            wj = j

            while random.random() > WALL_STOP_PROBABILITY:
                wi += direction[0]
                wj += direction[1]

                if areAnyWallsAround(wi, wj, direction):
                    break
                if isBorder(wi, wj) and borderStart:
                    break

                if isOutOfBounds(wi, wj):
                    break
                walls[wi, wj] = 1

    return walls


def printWalls(map):
    # data = np.random.rand(10, 10) * 20
    data = map

    # create discrete colormap
    cmap = colors.ListedColormap(['white', 'black', 'green'])
    bounds = [0, 0.5, 1.5, 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

    plt.xticks(np.arange(0.5, data.shape[1], 1))  # correct grid sizes
    plt.yticks(np.arange(0.5, data.shape[0], 1))

    plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    plt.savefig('map.png')

    plt.show()


main()
