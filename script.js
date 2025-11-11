class Gomoku {
    constructor() {
        this.boardSize = 15;
        this.cellSize = 32;
        this.board = Array(this.boardSize).fill().map(() => Array(this.boardSize).fill(0));
        this.currentPlayer = 1; // 1 for black, 2 for white
        this.gameOver = false;
        
        this.canvas = document.getElementById('chessboard');
        this.ctx = this.canvas.getContext('2d');
        this.currentPlayerSpan = document.getElementById('current-player');
        this.winnerMessage = document.getElementById('winner-message');
        this.resetBtn = document.getElementById('reset-btn');
        
        this.drawBoard();
        this.addEventListeners();
    }
    
    drawBoard() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Draw grid lines
        this.ctx.strokeStyle = '#000';
        this.ctx.lineWidth = 1;
        
        for (let i = 0; i < this.boardSize; i++) {
            // Vertical lines
            this.ctx.beginPath();
            this.ctx.moveTo(i * this.cellSize + this.cellSize/2, this.cellSize/2);
            this.ctx.lineTo(i * this.cellSize + this.cellSize/2, this.canvas.height - this.cellSize/2);
            this.ctx.stroke();
            
            // Horizontal lines
            this.ctx.beginPath();
            this.ctx.moveTo(this.cellSize/2, i * this.cellSize + this.cellSize/2);
            this.ctx.lineTo(this.canvas.width - this.cellSize/2, i * this.cellSize + this.cellSize/2);
            this.ctx.stroke();
        }
        
        // Draw pieces
        for (let row = 0; row < this.boardSize; row++) {
            for (let col = 0; col < this.boardSize; col++) {
                if (this.board[row][col] !== 0) {
                    this.drawPiece(row, col, this.board[row][col]);
                }
            }
        }
    }
    
    drawPiece(row, col, player) {
        const x = col * this.cellSize + this.cellSize/2;
        const y = row * this.cellSize + this.cellSize/2;
        const radius = this.cellSize/2 - 2;
        
        this.ctx.beginPath();
        this.ctx.arc(x, y, radius, 0, Math.PI * 2);
        
        if (player === 1) { // Black piece
            this.ctx.fillStyle = '#000';
        } else { // White piece
            this.ctx.fillStyle = '#fff';
        }
        
        this.ctx.fill();
        this.ctx.stroke();
    }
    
    addEventListeners() {
        this.canvas.addEventListener('click', (e) => {
            if (this.gameOver) return;
            
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            const col = Math.round((x - this.cellSize/2) / this.cellSize);
            const row = Math.round((y - this.cellSize/2) / this.cellSize);
            
            if (this.isValidMove(row, col)) {
                this.makeMove(row, col);
            }
        });
        
        this.resetBtn.addEventListener('click', () => {
            this.resetGame();
        });
    }
    
    isValidMove(row, col) {
        return row >= 0 && row < this.boardSize && 
               col >= 0 && col < this.boardSize && 
               this.board[row][col] === 0;
    }
    
    makeMove(row, col) {
        this.board[row][col] = this.currentPlayer;
        this.drawPiece(row, col, this.currentPlayer);
        
        if (this.checkWin(row, col)) {
            this.gameOver = true;
            const winner = this.currentPlayer === 1 ? '黑子' : '白子';
            this.winnerMessage.textContent = `游戏结束! ${winner}获胜!`;
            this.winnerMessage.className = `winner ${this.currentPlayer === 1 ? 'black-win' : 'white-win'}`;
        } else {
            this.switchPlayer();
        }
    }
    
    switchPlayer() {
        this.currentPlayer = this.currentPlayer === 1 ? 2 : 1;
        this.currentPlayerSpan.textContent = this.currentPlayer === 1 ? '黑子' : '白子';
    }
    
    checkWin(row, col) {
        const player = this.board[row][col];
        const directions = [
            [0, 1],  // horizontal
            [1, 0],  // vertical
            [1, 1],  // diagonal down-right
            [1, -1]  // diagonal down-left
        ];
        
        for (let [dx, dy] of directions) {
            let count = 1; // Count the current piece
            
            // Check in positive direction
            for (let i = 1; i < 5; i++) {
                const r = row + dx * i;
                const c = col + dy * i;
                if (r >= 0 && r < this.boardSize && c >= 0 && c < this.boardSize && 
                    this.board[r][c] === player) {
                    count++;
                } else {
                    break;
                }
            }
            
            // Check in negative direction
            for (let i = 1; i < 5; i++) {
                const r = row - dx * i;
                const c = col - dy * i;
                if (r >= 0 && r < this.boardSize && c >= 0 && c < this.boardSize && 
                    this.board[r][c] === player) {
                    count++;
                } else {
                    break;
                }
            }
            
            if (count >= 5) {
                return true;
            }
        }
        
        return false;
    }
    
    resetGame() {
        this.board = Array(this.boardSize).fill().map(() => Array(this.boardSize).fill(0));
        this.currentPlayer = 1;
        this.gameOver = false;
        this.currentPlayerSpan.textContent = '黑子';
        this.winnerMessage.textContent = '';
        this.winnerMessage.className = 'winner';
        this.drawBoard();
    }
}

// Initialize the game when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new Gomoku();
});