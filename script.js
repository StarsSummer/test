class MonaLisaPuzzle {
    constructor() {
        this.gridSize = 4;
        this.pieces = [];
        this.emptyPosition = null;
        this.moves = 0;
        this.startTime = null;
        this.timerInterval = null;
        this.isPlaying = false;
        this.imageUrl = '83fe91f28c23c421c586ee2ff8f36043.jpg';
        
        this.initializeElements();
        this.bindEvents();
        this.initializeGame();
    }

    initializeElements() {
        this.puzzleGrid = document.getElementById('puzzleGrid');
        this.difficultySelect = document.getElementById('difficulty');
        this.movesDisplay = document.getElementById('moves');
        this.timerDisplay = document.getElementById('timer');
        this.shuffleBtn = document.getElementById('shuffleBtn');
        this.showOriginalBtn = document.getElementById('showOriginalBtn');
        this.solveBtn = document.getElementById('solveBtn');
        this.winModal = document.getElementById('winModal');
        this.previewModal = document.getElementById('previewModal');
        this.finalMoves = document.getElementById('finalMoves');
        this.finalTime = document.getElementById('finalTime');
        this.playAgainBtn = document.getElementById('playAgainBtn');
        this.closeModalBtn = document.getElementById('closeModalBtn');
        this.closePreviewBtn = document.getElementById('closePreviewBtn');
    }

    bindEvents() {
        this.difficultySelect.addEventListener('change', () => this.changeDifficulty());
        this.shuffleBtn.addEventListener('click', () => this.shufflePuzzle());
        this.showOriginalBtn.addEventListener('click', () => this.showOriginal());
        this.solveBtn.addEventListener('click', () => this.solvePuzzle());
        this.playAgainBtn.addEventListener('click', () => this.playAgain());
        this.closeModalBtn.addEventListener('click', () => this.closeWinModal());
        this.closePreviewBtn.addEventListener('click', () => this.closePreviewModal());
        
        // 点击模态框外部关闭
        this.winModal.addEventListener('click', (e) => {
            if (e.target === this.winModal) this.closeWinModal();
        });
        this.previewModal.addEventListener('click', (e) => {
            if (e.target === this.previewModal) this.closePreviewModal();
        });

        // 键盘控制
        document.addEventListener('keydown', (e) => this.handleKeyPress(e));
    }

    changeDifficulty() {
        this.gridSize = parseInt(this.difficultySelect.value);
        this.resetGame();
        this.initializeGame();
    }

    initializeGame() {
        this.createPuzzle();
        this.setupGrid();
        this.renderPuzzle();
    }

    createPuzzle() {
        this.pieces = [];
        const totalPieces = this.gridSize * this.gridSize;
        
        for (let i = 0; i < totalPieces - 1; i++) {
            this.pieces.push(i);
        }
        this.pieces.push(null); // 空格
        this.emptyPosition = totalPieces - 1;
    }

    setupGrid() {
        this.puzzleGrid.innerHTML = '';
        this.puzzleGrid.style.gridTemplateColumns = `repeat(${this.gridSize}, 1fr)`;
        this.puzzleGrid.style.gridTemplateRows = `repeat(${this.gridSize}, 1fr)`;
    }

    renderPuzzle() {
        this.puzzleGrid.innerHTML = '';
        const pieceSize = 600 / this.gridSize;

        this.pieces.forEach((piece, index) => {
            const pieceElement = document.createElement('div');
            pieceElement.className = 'puzzle-piece';
            pieceElement.dataset.index = index;

            if (piece === null) {
                pieceElement.classList.add('empty');
            } else {
                const row = Math.floor(piece / this.gridSize);
                const col = piece % this.gridSize;
                
                pieceElement.style.backgroundImage = `url(${this.imageUrl})`;
                pieceElement.style.backgroundPosition = 
                    `-${col * pieceSize}px -${row * pieceSize}px`;
                
                pieceElement.addEventListener('click', () => this.movePiece(index));
            }

            this.puzzleGrid.appendChild(pieceElement);
        });
    }

    movePiece(index) {
        if (!this.isPlaying) return;
        
        if (this.isAdjacent(index, this.emptyPosition)) {
            // 交换拼图块
            [this.pieces[index], this.pieces[this.emptyPosition]] = 
            [this.pieces[this.emptyPosition], this.pieces[index]];
            
            this.emptyPosition = index;
            this.moves++;
            this.updateMoves();
            
            this.renderPuzzle();
            
            // 检查是否完成
            if (this.checkWin()) {
                this.handleWin();
            }
        }
    }

    isAdjacent(index1, index2) {
        const row1 = Math.floor(index1 / this.gridSize);
        const col1 = index1 % this.gridSize;
        const row2 = Math.floor(index2 / this.gridSize);
        const col2 = index2 % this.gridSize;

        return (Math.abs(row1 - row2) === 1 && col1 === col2) ||
               (Math.abs(col1 - col2) === 1 && row1 === row2);
    }

    shufflePuzzle() {
        this.resetGame();
        
        // 通过随机移动来打乱拼图（确保可解）
        const shuffleMoves = this.gridSize * this.gridSize * 10;
        
        for (let i = 0; i < shuffleMoves; i++) {
            const adjacentIndices = this.getAdjacentIndices(this.emptyPosition);
            const randomIndex = adjacentIndices[Math.floor(Math.random() * adjacentIndices.length)];
            
            [this.pieces[randomIndex], this.pieces[this.emptyPosition]] = 
            [this.pieces[this.emptyPosition], this.pieces[randomIndex]];
            
            this.emptyPosition = randomIndex;
        }
        
        this.startGame();
        this.renderPuzzle();
    }

    getAdjacentIndices(index) {
        const adjacent = [];
        const row = Math.floor(index / this.gridSize);
        const col = index % this.gridSize;

        // 上
        if (row > 0) adjacent.push(index - this.gridSize);
        // 下
        if (row < this.gridSize - 1) adjacent.push(index + this.gridSize);
        // 左
        if (col > 0) adjacent.push(index - 1);
        // 右
        if (col < this.gridSize - 1) adjacent.push(index + 1);

        return adjacent;
    }

    startGame() {
        this.isPlaying = true;
        this.startTime = Date.now();
        this.startTimer();
    }

    startTimer() {
        this.timerInterval = setInterval(() => {
            const elapsed = Date.now() - this.startTime;
            const minutes = Math.floor(elapsed / 60000);
            const seconds = Math.floor((elapsed % 60000) / 1000);
            this.timerDisplay.textContent = 
                `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }, 1000);
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    updateMoves() {
        this.movesDisplay.textContent = this.moves;
    }

    checkWin() {
        for (let i = 0; i < this.pieces.length - 1; i++) {
            if (this.pieces[i] !== i) return false;
        }
        return this.pieces[this.pieces.length - 1] === null;
    }

    handleWin() {
        this.isPlaying = false;
        this.stopTimer();
        
        // 添加完成动画
        const pieces = this.puzzleGrid.querySelectorAll('.puzzle-piece:not(.empty)');
        pieces.forEach((piece, index) => {
            setTimeout(() => {
                piece.classList.add('correct');
            }, index * 50);
        });

        // 显示胜利模态框
        setTimeout(() => {
            this.finalMoves.textContent = this.moves;
            this.finalTime.textContent = this.timerDisplay.textContent;
            this.winModal.classList.add('show');
        }, 1000);
    }

    solvePuzzle() {
        this.stopTimer();
        this.isPlaying = false;
        
        // 创建正确顺序
        this.pieces = [];
        for (let i = 0; i < this.gridSize * this.gridSize - 1; i++) {
            this.pieces.push(i);
        }
        this.pieces.push(null);
        this.emptyPosition = this.pieces.length - 1;
        
        this.renderPuzzle();
        
        // 添加完成动画
        setTimeout(() => {
            const pieces = this.puzzleGrid.querySelectorAll('.puzzle-piece:not(.empty)');
            pieces.forEach((piece, index) => {
                setTimeout(() => {
                    piece.classList.add('correct');
                }, index * 30);
            });
        }, 100);
    }

    resetGame() {
        this.stopTimer();
        this.isPlaying = false;
        this.moves = 0;
        this.updateMoves();
        this.timerDisplay.textContent = '00:00';
    }

    playAgain() {
        this.closeWinModal();
        this.shufflePuzzle();
    }

    showOriginal() {
        this.previewModal.classList.add('show');
    }

    closeWinModal() {
        this.winModal.classList.remove('show');
    }

    closePreviewModal() {
        this.previewModal.classList.remove('show');
    }

    handleKeyPress(e) {
        if (!this.isPlaying) return;

        const emptyRow = Math.floor(this.emptyPosition / this.gridSize);
        const emptyCol = this.emptyPosition % this.gridSize;
        let targetIndex = -1;

        switch(e.key) {
            case 'ArrowUp':
                if (emptyRow < this.gridSize - 1) {
                    targetIndex = this.emptyPosition + this.gridSize;
                }
                break;
            case 'ArrowDown':
                if (emptyRow > 0) {
                    targetIndex = this.emptyPosition - this.gridSize;
                }
                break;
            case 'ArrowLeft':
                if (emptyCol < this.gridSize - 1) {
                    targetIndex = this.emptyPosition + 1;
                }
                break;
            case 'ArrowRight':
                if (emptyCol > 0) {
                    targetIndex = this.emptyPosition - 1;
                }
                break;
        }

        if (targetIndex >= 0) {
            e.preventDefault();
            this.movePiece(targetIndex);
        }
    }
}

// 初始化游戏
document.addEventListener('DOMContentLoaded', () => {
    new MonaLisaPuzzle();
});