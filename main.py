import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import random

N = 15
WALL_START_PROBABILITY = 0.9
WALL_STOP_PROBABILITY = 0.01

NON_EXIT_STATE_WEIGHT = -0.1

EXIT_STATE_WEIGHT = 1

WALL_INT = 1
NON_EXIT_STATE_INT = 0
EXIT_STATE_INT = 2

EPSILON = 0.001
DISCOUNT_FACTOR = 0.95


SHOW_PLOTS = True


ERROR_INTERATION_FILENAME = "error_iterations.txt"

AVERAGE_STEPS_FILENAME = "steps_iterations.txt"


def main():

    #clean file
    with open(ERROR_INTERATION_FILENAME, 'w') as file:
        pass

    with open(AVERAGE_STEPS_FILENAME, 'w') as file:
        pass

    #
    walls = generateWalls()
    generate_exit(walls)

    # print(np.matrix(walls))

    printWalls(walls)

    weights = generateWeights(walls)

    # print(np.matrix(weights))

    utility_function = learn(walls, weights, DISCOUNT_FACTOR, EPSILON)

    print(np.matrix(utility_function))

    printUtilities(utility_function, walls)

    can_navigate, average_steps = check_navigation_and_average_steps_with_utility(walls, utility_function)
    if can_navigate:
        print("All points can navigate to the exit.")
        print(f"Average steps to the exit: {average_steps}")
    else:
        print("Some points cannot navigate to the exit.")



def learn(walls, weights, discount, epsilon):
    utility_function = np.zeros((N, N))
    biggest_change = 999

    while biggest_change >= epsilon * (1 - discount) / discount:
        next_utility_function = np.zeros((N, N))
        biggest_change = 0

        for i in range(N):
            for j in range(N):

                if walls[i, j] == 1:
                    continue

                neighbors = []
                if i > 0 and walls[i - 1, j] != 1:  # Up
                    neighbors.append(utility_function[i - 1, j])
                if i < N - 1 and walls[i + 1, j] != 1:  # Down
                    neighbors.append(utility_function[i + 1, j])
                if j > 0 and walls[i, j - 1] != 1:  # Left
                    neighbors.append(utility_function[i, j - 1])
                if j < N - 1 and walls[i, j + 1] != 1:  # Right
                    neighbors.append(utility_function[i, j + 1])

                max_utility = max(neighbors) if neighbors else 0

                next_utility_function[i, j] = weights[i, j] + discount * max_utility

                change = abs(next_utility_function[i, j] - utility_function[i, j])
                if change > biggest_change:
                    biggest_change = change

        with open(ERROR_INTERATION_FILENAME, 'a') as file:
            file.write(str(biggest_change) + '\n')
        utility_function = next_utility_function

    return utility_function


def generateWeights(map):
    weights = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if map[i, j] == NON_EXIT_STATE_INT:
                weights[i, j] = NON_EXIT_STATE_WEIGHT
            elif map[i, j] == EXIT_STATE_INT:
                weights[i, j] = EXIT_STATE_WEIGHT

    return weights


def generate_exit(walls):
    border_coords = border_coordinates()

    while True:
        exit = random.choice(border_coords)
        if walls[exit[0], exit[1]] == 0:
            walls[exit[0], exit[1]] = 2
            break


def isOutOfBounds(i, j):
    return i < 0 or i >= N or j < 0 or j >= N


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
    data = map

    cmap = colors.ListedColormap(['white', 'black', 'green'])
    bounds = [0, 0.5, 1.5, 2]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm)

    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

    plt.xticks(np.arange(0.5, data.shape[1], 1))
    plt.yticks(np.arange(0.5, data.shape[0], 1))

    plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    plt.savefig('map.png')

    if SHOW_PLOTS:
        plt.show()

def printUtilities(utility_function, walls):
    data = utility_function
    wall_color = 'saddlebrown'

    norm = plt.Normalize(vmin=np.min(data), vmax=np.max(data))

    fig, ax = plt.subplots()
    cax = ax.imshow(data,  norm=norm)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if walls[i, j] == 1:
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=True, color=wall_color))

    cbar = fig.colorbar(cax, ax=ax, orientation='vertical')
    cbar.set_label('Utility Value')

    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)

    plt.xticks(np.arange(0.5, data.shape[1], 1))
    plt.yticks(np.arange(0.5, data.shape[0], 1))

    plt.tick_params(bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)

    plt.savefig('utility_map_with_walls.png')

    if SHOW_PLOTS:
        plt.show()


def follow_utility_path(walls, utility_function, start, exit):
    current = start
    steps = 0
    visited = set()

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while current != exit:
        visited.add(current)
        steps += 1

        max_utility = float('-inf')
        next_cell = None

        for direction in directions:
            next_cell_candidate = (current[0] + direction[0], current[1] + direction[1])

            if (0 <= next_cell_candidate[0] < N and 0 <= next_cell_candidate[1] < N and
                    next_cell_candidate not in visited and walls[next_cell_candidate[0], next_cell_candidate[1]] != WALL_INT):
                candidate_utility = utility_function[next_cell_candidate[0], next_cell_candidate[1]]
                if candidate_utility > max_utility:
                    max_utility = candidate_utility
                    next_cell = next_cell_candidate

        if next_cell is None:
            return False, steps

        current = next_cell

    return True, steps

def check_navigation_and_average_steps_with_utility(walls, utility_function):
    exit_coords = np.argwhere(walls == EXIT_STATE_INT)
    if len(exit_coords) == 0:
        raise ValueError("No exit found on the map.")
    exit_coords = tuple(exit_coords[0])

    total_steps = 0
    reachable_cells = 0

    for i in range(N):
        for j in range(N):
            if walls[i, j] == NON_EXIT_STATE_INT:
                can_reach, steps = follow_utility_path(walls, utility_function, (i, j), exit_coords)
                if can_reach:
                    total_steps += steps
                    reachable_cells += 1
                else:
                    return False, None

    average_steps = total_steps / reachable_cells if reachable_cells > 0 else None
    return True, average_steps

main()
