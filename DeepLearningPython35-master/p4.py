#-*-coding:utf8;-*-
#qpy:3
#qpy:console   

class Grid :
    
    def __init__(self) :
        self.grid = [ [0 for i in range(7)] for j in range(6)]
        
    def __str__(self) :
        string = ""
        for row in self.grid :
            string += str(row) + "\n"
        return string

    def add(self, col, num) :
        if col >= len(self.grid[col]) or col < 0 or self.grid [0][col] != 0 :
            return False
        else :
            row = 0
            while row+1 < len(self.grid) and self.grid [row+1][col] == 0 :
                row += 1
            self.grid [row][col] = num
            return True
            
    def isFull(self) :
        for row in self.grid :
            for elmt in row :
                if elmt == 0 :
                    return False
        return True
        
    def winner(self) :
        for i, row in enumerate(self.grid) :
            for j, elmt in enumerate(row) :
                if elmt != 0 :
                    wonR = True
                    wonC = True
                    wonD = True
                    wonA = True
                    for a in range(1,4) :                 #       try :
                            if j+a >= len(row) or elmt != row[j+a] :
                                wonR = False
                            if i+a >= len(self.grid) or elmt != self.grid[i+a][j] :
                                wonC = False
                            if i+a >= len(self.grid) or j+a >= len(self.grid[i+a]) or elmt != self.grid[i+a][j+a] :
                                wonD = False   
                            if i+a >= len(self.grid) or j-a >= len(self.grid[i+a]) or elmt != self.grid[i+a][j-a] :
                                wonA = False
                                
                    if wonC or wonR or wonD :
                        return elmt
        return 0
    
    def play(self) :
        player = 1
        print(self)
        while not(self.isFull()) and self.winner() == 0 :
            col = int(input())
            while not(self.add(col,player)) :
                col = int(input())
            print(self)
            player *= -1
            print(self.winner())

grid = Grid()
grid.play()