import sys

# The Tic Tac Toe board
board = [' '] * 9

# Constants for players and their symbols
PLAYER_X = 'X'
PLAYER_O = 'O'

# Win conditions
WIN_CONDITIONS = [
    [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
    [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
    [0, 4, 8], [2, 4, 6]  # Diagonals
]


def print_board():
    print("-------------")
    print("|", board[0], "|", board[1], "|", board[2], "|")
    print("-------------")
    print("|", board[3], "|", board[4], "|", board[5], "|")
    print("-------------")
    print("|", board[6], "|", board[7], "|", board[8], "|")
    print("-------------")


def is_winner(player):
    for condition in WIN_CONDITIONS:
        if all(board[i] == player for i in condition):
            return True
    return False


def is_draw():
    return ' ' not in board


def get_empty_cells():
    return [i for i, cell in enumerate(board) if cell == ' ']


def minimax(position, depth, maximizing_player):
    if is_winner(PLAYER_X):
        return depth - 10
    elif is_winner(PLAYER_O):
        return 10 - depth
    elif is_draw():
        return 0

    if maximizing_player:
        max_eval = -sys.maxsize
        for empty_cell in get_empty_cells():
            board[empty_cell] = PLAYER_O
            eval = minimax(empty_cell, depth + 1, False)
            board[empty_cell] = ' '
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = sys.maxsize
        for empty_cell in get_empty_cells():
            board[empty_cell] = PLAYER_X
            eval = minimax(empty_cell, depth + 1, True)
            board[empty_cell] = ' '
            min_eval = min(min_eval, eval)
        return min_eval


def get_best_move():
    best_eval = -sys.maxsize
    best_move = None
    for empty_cell in get_empty_cells():
        board[empty_cell] = PLAYER_O
        eval = minimax(empty_cell, 0, False)
        board[empty_cell] = ' '
        if eval > best_eval:
            best_eval = eval
            best_move = empty_cell
    return best_move


def play_game():
    print("Tic Tac Toe - Minimax Algorithm")
    player = PLAYER_X
    while True:
        print_board()
        if player == PLAYER_X:
            position = int(input("Player X, enter your move (1-9): "))
            position -= 1
            if board[position] != ' ':
                print("Invalid move. Try again.")
                continue
        else:
            print("Player O is thinking...")
            position = get_best_move()
        board[position] = player

        if is_winner(player):
            print_board()
            print("Player", player, "wins!")
            break
        elif is_draw():
            print_board()
            print("It's a draw!")
            break

        player = PLAYER_O if player == PLAYER_X else PLAYER_X


# Start the game
play_game()
