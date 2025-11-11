document.addEventListener('DOMContentLoaded', () => {
    const board = document.getElementById('board');
    const currentPlayerElement = document.getElementById('current-player');
    const gameStatusElement = document.getElementById('game-status');
    const restartBtn = document.getElementById('restart-btn');

    // 游戏状态
    let gameState = {
        board: [],
        currentPlayer: 'red', // 'red' 或 'black'
        selectedPiece: null,
        validMoves: [],
        gameOver: false
    };

    // 初始化棋盘
    function initializeBoard() {
        board.innerHTML = '';
        gameState.board = [];
        
        // 创建8x8棋盘
        for (let row = 0; row < 8; row++) {
            gameState.board[row] = [];
            for (let col = 0; col < 8; col++) {
                const square = document.createElement('div');
                square.className = 'square';
                square.classList.add((row + col) % 2 === 0 ? 'light' : 'dark');
                square.dataset.row = row;
                square.dataset.col = col;
                
                // 设置棋子
                let piece = null;
                if (row < 3 && (row + col) % 2 === 1) {
                    // 黑方棋子（顶部3行）
                    piece = { color: 'black', isKing: false };
                } else if (row > 4 && (row + col) % 2 === 1) {
                    // 红方棋子（底部3行）
                    piece = { color: 'red', isKing: false };
                }
                
                gameState.board[row][col] = piece;
                
                if (piece) {
                    const pieceElement = document.createElement('div');
                    pieceElement.className = `piece ${piece.color}`;
                    if (piece.isKing) {
                        pieceElement.classList.add('king');
                    }
                    pieceElement.dataset.row = row;
                    pieceElement.dataset.col = col;
                    pieceElement.addEventListener('click', handlePieceClick);
                    square.appendChild(pieceElement);
                }
                
                square.addEventListener('click', handleSquareClick);
                board.appendChild(square);
            }
        }
    }

    // 处理棋子点击
    function handlePieceClick(e) {
        e.stopPropagation();
        if (gameState.gameOver) return;
        
        const row = parseInt(e.target.dataset.row);
        const col = parseInt(e.target.dataset.col);
        
        const piece = gameState.board[row][col];
        if (piece && piece.color === gameState.currentPlayer) {
            // 选择棋子
            selectPiece(row, col);
        }
    }

    // 处理方格点击
    function handleSquareClick(e) {
        if (gameState.gameOver) return;
        
        const row = parseInt(e.target.dataset.row);
        const col = parseInt(e.target.dataset.col);
        
        // 检查是否点击了有效的移动位置
        const move = gameState.validMoves.find(m => m.row === row && m.col === col);
        if (move) {
            movePiece(gameState.selectedPiece.row, gameState.selectedPiece.col, row, col);
        } else {
            // 如果点击的不是有效移动位置，取消选择
            clearSelection();
        }
    }

    // 选择棋子
    function selectPiece(row, col) {
        clearSelection();
        
        gameState.selectedPiece = { row, col };
        
        // 高亮选中的棋子
        const selectedSquare = document.querySelector(`.square[data-row="${row}"][data-col="${col}"]`);
        selectedSquare.classList.add('selected');
        
        // 计算有效移动
        gameState.validMoves = calculateValidMoves(row, col);
        
        // 高亮有效移动位置
        gameState.validMoves.forEach(move => {
            const moveSquare = document.querySelector(`.square[data-row="${move.row}"][data-col="${move.col}"]`);
            moveSquare.classList.add('valid-move');
        });
    }

    // 清除选择
    function clearSelection() {
        // 移除所有高亮
        document.querySelectorAll('.square').forEach(square => {
            square.classList.remove('selected', 'valid-move');
        });
        
        gameState.selectedPiece = null;
        gameState.validMoves = [];
    }

    // 计算有效移动
    function calculateValidMoves(row, col) {
        const piece = gameState.board[row][col];
        const moves = [];
        
        // 检查基本移动（对角线）
        const directions = [];
        
        if (piece.isKing) {
            // 王可以向前和向后移动
            directions.push([-1, -1], [-1, 1], [1, -1], [1, 1]);
        } else if (piece.color === 'red') {
            // 红方只能向上移动
            directions.push([-1, -1], [-1, 1]);
        } else {
            // 黑方只能向下移动
            directions.push([1, -1], [1, 1]);
        }
        
        // 检查基本移动和跳跃
        for (const [dRow, dCol] of directions) {
            // 基本移动
            const newRow = row + dRow;
            const newCol = col + dCol;
            
            if (isValidPosition(newRow, newCol) && !gameState.board[newRow][newCol]) {
                moves.push({ row: newRow, col: newCol });
            }
            
            // 跳跃移动
            const jumpRow = row + 2 * dRow;
            const jumpCol = col + 2 * dCol;
            
            if (isValidPosition(jumpRow, jumpCol) && 
                !gameState.board[jumpRow][jumpCol] && 
                isValidPosition(row + dRow, col + dCol) && 
                gameState.board[row + dRow][col + dCol] && 
                gameState.board[row + dRow][col + dCol].color !== piece.color) {
                moves.push({ row: jumpRow, col: jumpCol, jump: { row: row + dRow, col: col + dCol } });
            }
        }
        
        return moves;
    }

    // 检查位置是否有效
    function isValidPosition(row, col) {
        return row >= 0 && row < 8 && col >= 0 && col < 8;
    }

    // 移动棋子
    function movePiece(fromRow, fromCol, toRow, toCol) {
        const piece = gameState.board[fromRow][fromCol];
        const move = gameState.validMoves.find(m => m.row === toRow && m.col === toCol);
        
        // 移动棋子
        gameState.board[toRow][toCol] = piece;
        gameState.board[fromRow][fromCol] = null;
        
        // 处理跳跃
        if (move.jump) {
            // 移除被跳过的棋子
            gameState.board[move.jump.row][move.jump.col] = null;
        }
        
        // 检查是否成为王
        if ((piece.color === 'red' && toRow === 0) || (piece.color === 'black' && toRow === 7)) {
            piece.isKing = true;
        }
        
        // 更新界面
        updateBoard();
        
        // 检查游戏结束
        if (checkGameOver()) {
            gameState.gameOver = true;
            gameStatusElement.textContent = `${gameState.currentPlayer === 'red' ? '黑方' : '红方'} 获胜！`;
            return;
        }
        
        // 切换玩家
        gameState.currentPlayer = gameState.currentPlayer === 'red' ? 'black' : 'red';
        currentPlayerElement.textContent = gameState.currentPlayer === 'red' ? '红方' : '黑方';
        
        clearSelection();
    }

    // 更新棋盘界面
    function updateBoard() {
        const squares = document.querySelectorAll('.square');
        squares.forEach(square => {
            square.innerHTML = '';
            const row = parseInt(square.dataset.row);
            const col = parseInt(square.dataset.col);
            
            const piece = gameState.board[row][col];
            if (piece) {
                const pieceElement = document.createElement('div');
                pieceElement.className = `piece ${piece.color}`;
                if (piece.isKing) {
                    pieceElement.classList.add('king');
                }
                pieceElement.dataset.row = row;
                pieceElement.dataset.col = col;
                pieceElement.addEventListener('click', handlePieceClick);
                square.appendChild(pieceElement);
            }
        });
    }

    // 检查游戏是否结束
    function checkGameOver() {
        let redPieces = 0;
        let blackPieces = 0;
        
        for (let row = 0; row < 8; row++) {
            for (let col = 0; col < 8; col++) {
                const piece = gameState.board[row][col];
                if (piece) {
                    if (piece.color === 'red') {
                        redPieces++;
                    } else {
                        blackPieces++;
                    }
                }
            }
        }
        
        return redPieces === 0 || blackPieces === 0;
    }

    // 重新开始游戏
    function restartGame() {
        gameState = {
            board: [],
            currentPlayer: 'red',
            selectedPiece: null,
            validMoves: [],
            gameOver: false
        };
        
        currentPlayerElement.textContent = '红方';
        gameStatusElement.textContent = '';
        initializeBoard();
    }

    // 事件监听器
    restartBtn.addEventListener('click', restartGame);
    
    // 初始化游戏
    initializeBoard();
});