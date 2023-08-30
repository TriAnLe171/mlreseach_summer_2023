import random
import math

class Game:
    def __init__(self):
        self.steps = 0

    def update(self):
        self.steps += 1
    
    def checkTie(self):
        return self.steps == 9
    
board = [["-", "-", "-"],
         ["-", "-", "-"],
         ["-", "-", "-"]]

class Checker:
    def __init__(self):
        self.running = True
    
    def end(self):
        self.running = False

    def check(self):
        return self.running
    
    def reset(self):
        self.running = True
    
class Winner:
    def __init__(self, checker):
        self.winner = ""
        self.checker = checker

    def setWinner(self, win):
        if win == "-":
            return
        else:
            self.winner = win
            self.checker.end()

    def getWinner(self):
        return self.winner
    
    def reset(self):
        self.winner = ""

def print_board(board):
    for i in range(3):
        string = ""
        for j in range(3):
            if j == 0:
                print("-------------")
                string += "| "
            string += board[i][j]
            string += " | "
        print(string)
    print("-------------")

def horizontal(board, winner):
    if (board[0][0] == board[0][1] == board[0][2]) and (board[0][0] != "-"):
        winner.setWinner(board[0][0])
        return True
    if (board[1][0] == board[1][1] == board[1][2]) and (board[1][0] != "-"):
        winner.setWinner(board[1][0])
        return True
    if (board[2][0] == board[2][1] == board[2][2]) and (board[0][2] != "-"):
        winner.setWinner(board[2][0])
        return True
    else:
        return False

def vertical(board, winner):
    if (board[0][0] == board[1][0] == board[2][0]) and (board[0][0] != "-"):
        winner.setWinner(board[0][0])
        return True
    if (board[0][1] == board[1][1] == board[2][1]) and (board[0][1] != "-"):
        winner.setWinner(board[0][1])
        return True
    if (board[0][2] == board[1][2] == board[2][2]) and (board[0][2] != "-"):
        winner.setWinner(board[0][2])
        return True
    else:
        return False

def cross(board, winner):
    if (board[0][0] == board[1][1] == board[2][2]) and (board[0][0] != "-"):
        winner.setWinner(board[0][0])
        return True
    if (board[0][2] == board[1][1] == board[2][0]) and (board[0][1] != "-"):
        winner.setWinner(board[0][2])
        return True
    else:
        return False
    
def checkGame(board, winner):
    if horizontal(board, winner):
        return True
    if vertical(board, winner):
        return True
    if cross(board, winner):
        return True
    return False

def checkTie(board):
    if "-" not in board:
        return True
    else:
        return False
    
def checkWinner(board, checker, winner):
    if checkGame(board, winner):
        return winner.getWinner()
    return ""


def input_user(board, checker):
    if checker.check():
        inp = int(input("Enter the value in the range 1-9"))
        if inp >=1 and inp <= 9:
            inp -= 1
            i = int(inp / 3)
            j = inp - i*3
            if board[i][j] == "-":
                board[i][j] = "I"
            else:
                print("cell is already occupied")
        else:
            print("please choose the number from the range [1,9]")

def minimax(board, depth, iF, jF, player, checker, winner):
    K = iF
    N = jF
    if player:
        maxEval = float("-inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == "-":
                    board[i][j] = "R"
                    a = checkWinner(board, checker, winner)
                    if a is not "":
                        board[i][j] = "-"
                        checker.reset()
                        winner.reset()
                        if a == "R":
                            return (10 - depth, K, N)
                        else:
                            return (depth - 10, K, N)
                    elif checkTie(board):
                        board[i][j] = "-"
                        return (0, K, N)
                    else:
                        val, iK, jN = minimax(board, depth + 1, K, N, False, checker, winner)
                        board[i][j] = "-"
                        if val > maxEval:
                            maxEval = val
                            K = iK
                            N = jN
        return (maxEval, K, N)
    else:
        minEval = float("inf")
        for i in range(3):
            for j in range(3):
                if board[i][j] == "-":
                    board[i][j] = "I"
                    a = checkWinner(board, checker, winner)
                    if a is not "":
                        board[i][j] = "-"
                        checker.reset()
                        winner.reset()
                        if a == "R":
                            return (10 - depth, K, N)
                        else:
                            return (depth - 10, K, N)
                    elif checkTie(board):
                        board[i][j] = "-"
                        return (0, K, N)
                    else:
                        val, iK, jN = minimax(board, depth + 1, K, N, True, checker, winner)
                        board[i][j] = "-"
                        if val < minEval:
                            minEval = val
                            K = iK
                            N = jN
        return (minEval, K, N)

def first_ai(board):
    step = random.randint(0,8)
    i = int(step / 3)
    j = step - i*3
    board[i][j] = "R"

def ai_step(board, checker, winner):
    score = float("-inf")
    iT = 0
    jT = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] == "-":
                board[i][j] = "R"
                if checkWinner(board, checker, winner):
                    return
                for k in range(3):
                    for d in range(3):
                        if board[k][d] == "-":
                            board[k][d] = "I"
                            if checkWinner(board, checker, winner):
                                return 
                            val, iD, jD = minimax(board, 0, k, d, True, checker, winner)
                            if val > score:
                                score = val
                                iT = iD
                                jT = jD
                            board[k][d] = "-"
                board[i][j] = "-"
    print(score)
    board[iT][jT] = "R"

if __name__ == "__main__":
    checker = Checker()
    win = Winner(checker)
    game = Game()
    print_board(board)
    first = random.randint(0,1)
    if first == 0:
        while checker.check():
            print("Your step:")
            input_user(board, checker)
            print_board(board)
            game.update()
            a = checkWinner(board, checker, win)
            if a is not "":
                print(win.getWinner() +  " wins!")
                # checker.end()
                break
            if game.checkTie():
                checker.end()
                print("It's a tie!")  
                break            
            print("Robot step")
            ai_step(board, checker, win)
            print_board(board)
            game.update()
            b = checkWinner(board, checker, win)
            if b is not "":
                print(win.getWinner() +  " wins!")
                # checker.end()
                break
            if game.checkTie():
                checker.end()
                print("It's a tie!")              

    else:
        print("Robot step")
        first_ai(board)
        print_board(board)
        game.update()
        while checker.check():
            print("It's your step:")
            input_user(board, checker)
            print_board(board)
            game.update()
            a = checkWinner(board, checker, win)
            if a is not "":
                print(win.getWinner() +  " wins!")
                break          
            if game.checkTie():
                # checker.end()
                print("It's a tie!")  
                break
            print("Robot step:")
            ai_step(board, checker, win)
            print_board(board)
            game.update()
            b = checkWinner(board, checker, win)
            if b is not "":
                print(win.getWinner() +  " wins!")
                # checker.end()
                break
            if game.checkTie():
                checker.end()            
                print("It's a tie!")



        