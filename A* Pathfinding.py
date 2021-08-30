import random
import pygame
import math
from queue import PriorityQueue
from queue import LifoQueue
from queue import Queue

# GLOBAL VARIABLES #
# NEED TO SCALE WIDTH/HEIGHT with ROWS/COLS #
WIDTH = 720
HEIGHT = 720
NUM_ROWS = 30
NUM_COLS = 30
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))

# Colors #
BG = (255, 255, 255)
DEFAULT = (204, 220, 253)
START = (158, 163, 245)
END = (208, 153, 226)
TRAV = (232, 191, 251)
BLOCKED = (242, 165, 232)
RAND = (187, 190, 254)
PAST = (255, 105, 97)
PRESENT = (97, 247, 255)
BORDER = (128, 128, 128)

# Type Node -- has several fields and functions to set and check attributes
class Node:

    def __init__(self, row, col, x_len, y_len):
        self.row = row
        self.col = col
        self.x_len = x_len
        self.y_len = y_len
        self.x_pos = row * x_len
        self.y_pos = col * y_len
        self.color = DEFAULT
        self.neighbors = []

    # necessary for comparison between two nodes in pqueue
    def __lt__(self, other):
        return False

    def bool_past(self):
        return self.color == PAST

    def bool_present(self):
        return self.color == PRESENT

    def bool_block(self):
        return self.color == BLOCKED

    def bool_start(self):
        return self.color == START

    def bool_end(self):
        return self.color == END

    def make_default(self):
        self.color = DEFAULT

    def make_node_past(self):
        self.color = PAST

    def make_node_present(self):
        self.color = PRESENT

    def make_node_block(self):
        self.color = BLOCKED

    def make_node_start(self):
        self.color = START

    def make_node_end(self):
        self.color = END

    def make_node_path(self):
        self.color = RAND

    def make_node_traversed(self):
        self.color = TRAV

    def draw_node(self, screen):
        pygame.draw.rect(screen, self.color, (self.x_pos, self.y_pos, self.x_len, self.y_len))

    def get_node_neighbors(self, board):
        self.neighbors = []

        # Get neighbor up one if valid
        if self.row > 0 and not board[self.row - 1][self.col].bool_block():
            self.neighbors.append(board[self.row - 1][self.col])

        # Get neighbor down one if valid
        if self.row < NUM_ROWS - 1 and not board[self.row + 1][self.col].bool_block():
            self.neighbors.append(board[self.row + 1][self.col])

        # Get neighbor right one if valid
        if self.col < NUM_COLS - 1 and not board[self.row][self.col + 1].bool_block():
            self.neighbors.append(board[self.row][self.col + 1])

        # Get neighbor left one if valid
        if self.col > 0 and not board[self.row][self.col - 1].bool_block():  # LEFT
            self.neighbors.append(board[self.row][self.col - 1])

# Returns row, col of mouse-click location
def get_mouse_clicked(rows, cols, width, height, position):
    # pygame.mouse.get_pos() returns y as first elem in tuple
    y = position[0]
    x = position[1]

    x_gap = width // cols
    y_gap = height // rows

    col = x // x_gap
    row = y // y_gap

    return (row, col)

# Draws the lines of the board
def draw_board(screen, width, height, rows, cols):
    x_gap = width // cols
    y_gap = height // rows
    for row in range(rows):
        pygame.draw.line(screen, BORDER, (0, row * y_gap), (width, row * y_gap))
        for col in range(cols):
            pygame.draw.line(screen, BORDER, (col * x_gap, 0), (col * x_gap, height))

# Draws each node on the board
def draw_nodes(screen, board):
    for row in board:
        for node in row:
            node.draw_node(screen)

# Creates a 2D array of Nodes
def make_board(width, height, rows, cols):
    board = []
    x_gap = width // cols
    y_gap = height // rows
    for row in range(rows):
        board.append([])
        for col in range(cols):
            board[row].append(Node(row, col, x_gap, y_gap))
    return board

# Draws final path to proper node
def draw_traversed_path(draw_func, dict, start):
    current = start
    while current in dict:
        current.make_node_path()
        draw_func()
        draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
        pygame.display.update()
        current = dict[current]
    start.make_node_end()

# BFS Algorithm
def BFS(start, end, draw_func):
    visiting_queue = Queue()
    visited_nodes = []
    neighborMap = {}
    visiting_queue.put(start)
    if start == end:
        draw_traversed_path(draw_func, neighborMap, end)
        return True
    while not visiting_queue.empty():
        node = visiting_queue.get()
        node.make_node_traversed()
        start.make_node_start()
        draw_func()
        draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
        pygame.display.update()
        if node not in visited_nodes:
            if node == end:
                draw_traversed_path(draw_func, neighborMap, end)
                return True
            visited_nodes.append(node)
            for neighbor in node.neighbors:
                visiting_queue.put(neighbor)
                if neighbor not in visited_nodes:
                    neighborMap[neighbor] = node
                    neighbor.make_node_present()
    return False

# DFS Algorithm
def DFS(start, end, draw_func):
    visiting_stack = LifoQueue()
    visited_nodes = []
    neighborMap = {}
    visiting_stack.put(start)
    if start == end:
        draw_traversed_path(draw_func, neighborMap, end)
        return True
    while not visiting_stack.empty():
        node = visiting_stack.get()
        node.make_node_traversed()
        start.make_node_start()
        draw_func()
        draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
        pygame.display.update()
        if node not in visited_nodes:
            if node == end:
                draw_traversed_path(draw_func, neighborMap, end)
                return True
            visited_nodes.append(node)
            for neighbor in node.neighbors:
                visiting_stack.put(neighbor)
                if neighbor not in visited_nodes:
                    neighborMap[neighbor] = node
                    neighbor.make_node_present()
    return False

# Heuristic function for A*, calculates 2D Euclidean Distance
def h_score_calc(point_one, point_two):
    (x1, y1) = point_one
    (x2, y2) = point_two
    return math.isqrt((abs(x2 - x1) ** 2) + (abs(y2 - y1) ** 2))

# A* algorithm
# Weighted but assumes each edge has weight 1
def ASTAR(board, start, end, draw_func):
    open_set = PriorityQueue()

    # using a set cuz p-queue interface doesn't have membership check
    open_set_set = {start}
    neighborMap = {}
    g_score_map = {}
    f_score_map = {}
    for row in board:
        for node in row:
            g_score_map[node] = float("inf")
            f_score_map[node] = float("inf")
    g_score_map[start] = 0
    f_score_map[start] = h_score_calc((start.row, start.col), (end.row, end.col))
    open_set.put((f_score_map[start], start))
    while not open_set.empty():
        current = open_set.get()[1]
        open_set_set.remove(current)
        if current == end:
            draw_traversed_path(draw_func, neighborMap, current)
            return True
        for neighbor in current.neighbors:
            neighbor_g_score = g_score_map[current] + 1
            if neighbor_g_score < g_score_map[neighbor]:
                neighborMap[neighbor] = current
                g_score_map[neighbor] = neighbor_g_score
                f_score_map[neighbor] = neighbor_g_score + h_score_calc((neighbor.row, neighbor.col), (end.row, end.col))
                if neighbor not in open_set_set:
                    open_set_set.add(neighbor)
                    open_set.put((f_score_map[neighbor], neighbor))
                    neighbor.make_node_present()
        if current is not start and current is not end:
            current.make_node_past()
        draw_func()
        draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
        pygame.display.update()
    return False

# Returns true or false based on a RNG
def random_tf():
    if random.randint(0, 1) == 0:
        return False
    return True

# Gets a random row value within start-end range
def get_rand_row(start_row, end_row):
    return random.randint(start_row + 1, end_row - 2)

# Gets a random column value within start-end range
def get_rand_col(start_col, end_col):
    return random.randint(start_col + 1, end_col - 2)

# Recursive Division algorithm
def RECURSIVEDIVISION(screen, board, start_row, end_row, start_col, end_col, draw_func):

    width = end_col - start_col
    height = end_row - start_row

    if width < 4 or height < 4:
        return

    if width == height:
        bool_horizontal_cut = random_tf()

    elif width > height:
        bool_horizontal_cut = False

    else:
        bool_horizontal_cut = True

    if bool_horizontal_cut:
        row = height // 2 + start_row
        for col in range(start_col, end_col):
            if col != 1 and col != NUM_COLS - 2:
                board[row][col].make_node_block()
            draw_func()
            draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
            pygame.display.update()
        rand_col = get_rand_col(start_col, end_col)
        board[row][rand_col].make_default()
        rand_col = get_rand_col(start_col, end_col)
        board[row][rand_col].make_default()
        draw_func()
        draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
        pygame.display.update()

    else:
        col = width // 2 + start_col
        for row in range(start_row, end_row):
            if row != 1 and row != NUM_ROWS - 2:
                board[row][col].make_node_block()
            draw_func()
            draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
            pygame.display.update()
        rand_row = get_rand_row(start_row, end_row)
        board[rand_row][col].make_default()
        rand_row = get_rand_row(start_row, end_row)
        board[rand_row][col].make_default()
        draw_func()
        draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
        pygame.display.update()

    if bool_horizontal_cut:
        RECURSIVEDIVISION(screen, board, start_row, height // 2 + start_row, start_col, end_col, draw_func)
        RECURSIVEDIVISION(screen, board, height // 2 + start_row, end_row, start_col, end_col, draw_func)
    else:
        RECURSIVEDIVISION(screen, board, start_row, end_row, start_col, width // 2 + start_col, draw_func)
        RECURSIVEDIVISION(screen, board, start_row, end_row, width // 2 + start_col, end_col, draw_func)

# Runs the actual function
def main(screen, width, height, rows, cols):
    board = make_board(width, height, rows, cols)
    on = True
    tracking = False
    start_node = None
    end_node = None
    while on:
        screen.fill(BG)
        draw_nodes(screen, board)
        draw_board(screen, width, height, rows, cols)
        pygame.display.update()

        # LEFT - MOUSE - CLICK
        if pygame.mouse.get_pressed()[0] and not tracking:
            pos = pygame.mouse.get_pos()
            (row, col) = get_mouse_clicked(rows, cols, width, height, pos)

            if board[row][col] != end_node and not board[row][col].bool_block() and start_node is None:
                start_node = board[row][col]
                start_node.make_node_start()

            elif board[row][col] != start_node and not board[row][col].bool_block() and end_node is None:
                end_node = board[row][col]
                end_node.make_node_end()

            elif board[row][col] != start_node and board[row][col] != end_node:
                board[row][col].make_node_block()

        # RIGHT - MOUSE - CLICK
        elif pygame.mouse.get_pressed()[2] and not tracking:
            pos = pygame.mouse.get_pos()
            (row, col) = get_mouse_clicked(rows, cols, width, height, pos)
            board[row][col].make_default()
            if board[row][col] == start_node:
                start_node = None
            elif board[row][col] == end_node:
                end_node = None

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                on = False

            if event.type == pygame.KEYDOWN and not tracking:

                # Press "b" on the keyboard to run BFS after selecting two nodes
                if event.key == pygame.K_b and start_node and end_node:
                    for row in board:
                        for node in row:
                            node.get_node_neighbors(board)
                    tracking = True
                    BFS(start_node, end_node, lambda: draw_nodes(screen, board))

                # Press "d" on the keyboard to run DFS after selecting two nodes
                elif event.key == pygame.K_d and start_node and end_node:
                    for row in board:
                        for node in row:
                            node.get_node_neighbors(board)
                    tracking = True
                    DFS(start_node, end_node, lambda: draw_nodes(screen, board))

                # Press "d" on the keyboard to run A* after selecting two nodes
                elif event.key == pygame.K_a and start_node and end_node:
                    for row in board:
                        for node in row:
                            node.get_node_neighbors(board)
                    tracking = True
                    ASTAR(board, start_node, end_node, lambda: draw_nodes(screen, board))

                # Press "r" on the keyboard to create a maze
                elif event.key == pygame.K_r:
                    for row in range(rows):
                        for col in range(cols):
                            if row == 0 or row == rows - 1:
                                board[row][col].make_node_block()
                                draw_nodes(screen, board)
                                draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
                                pygame.display.update()
                            if col == 0 or col == cols - 1:
                                board[row][col].make_node_block()
                                draw_nodes(screen, board)
                                draw_board(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
                                pygame.display.update()
                    RECURSIVEDIVISION(screen, board, 0, rows, 0, cols, lambda: draw_nodes(screen, board))

                # Press "c" on the keyboard to keep start, end, and blocks, but remove traversed
                elif event.key == pygame.K_c:
                    for row in board:
                        for node in row:
                            if not (node.bool_block() or node.bool_start() or node.bool_end()):
                                node.make_default()

                # Press "BACKSPACE" on the keyboard to clear the board
                elif event.key == pygame.K_BACKSPACE:
                    for row in board:
                        for node in row:
                            node.make_default()
                    start_node = None
                    end_node = None

                tracking = False

    pygame.quit()

main(SCREEN, WIDTH, HEIGHT, NUM_ROWS, NUM_COLS)
