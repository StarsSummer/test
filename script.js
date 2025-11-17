document.addEventListener('DOMContentLoaded', () => {
    const boardElement = document.getElementById('board');
    const currentPlayerElement = document.getElementById('current-player');
    const blackScoreElement = document.getElementById('black-score');
    const whiteScoreElement = document.getElementById('white-score');
    const resetButton = document.getElementById('reset-btn');
    
    const BOARD_SIZE = 8;
    let board = [];
    let currentPlayer = 'black'; // 'black' or 'white'
    let gameActive = true;
    
    // 初始化棋盘
    function initializeBoard() {
        board = Array(BOARD_SIZE).fill().map(() => Array(BOARD_SIZE).fill(null));
        
        // 初始四个棋子
        const mid = BOARD_SIZE / 2;
        board[mid-1][mid-1] = 'white';
        board[mid-1][mid] = 'black';
        board[mid][mid-1] = 'black';
        board[mid][mid] = 'white';
    }
    
    // 创建棋盘界面
    function createBoard() {
        boardElement.innerHTML = '';
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                const cell = document.createElement('div');
                cell.className = 'cell';
                cell.dataset.row = row;
                cell.dataset.col = col;
                
                const piece = board[row][col];
                if (piece) {
                    const pieceElement = document.createElement('div');
                    pieceElement.className = `piece ${piece}`;
                    cell.appendChild(pieceElement);
                }
                
                cell.addEventListener('click', () => handleCellClick(row, col));
                boardElement.appendChild(cell);
            }
        }
    }
    
    // 处理点击事件
    function handleCellClick(row, col) {
        if (!gameActive || board[row][col] !== null) return;
        
        const validMoves = getValidMoves();
        const moveKey = `${row},${col}`;
        
        if (!validMoves.includes(moveKey)) return;
        
        // 放置棋子
        board[row][col] = currentPlayer;
        
        // 翻转棋子
        const directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ];
        
        for (const [dx, dy] of directions) {
            flipPieces(row, col, dx, dy);
        }
        
        // 切换玩家
        currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
        
        // 检查下一个玩家是否有有效移动
        if (getValidMoves().length === 0) {
            // 如果当前玩家也没有有效移动，则游戏结束
            if (getValidMovesForPlayer(currentPlayer).length === 0) {
                gameActive = false;
                // 切换回之前的玩家以确定赢家
                currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
                updateStatus();
                alert('游戏结束！');
                return;
            } else {
                // 当前玩家无有效移动，跳过该玩家回合
                currentPlayer = currentPlayer === 'black' ? 'white' : 'black';
            }
        }
        
        updateStatus();
        createBoard();
    }
    
    // 翻转棋子
    function flipPieces(row, col, dx, dy) {
        const toFlip = [];
        let r = row + dx;
        let c = col + dy;
        
        // 沿着方向寻找敌方棋子
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] !== null && board[r][c] !== currentPlayer) {
            toFlip.push([r, c]);
            r += dx;
            c += dy;
        }
        
        // 如果找到己方棋子，则翻转中间的敌方棋子
        if (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE && board[r][c] === currentPlayer) {
            for (const [fr, fc] of toFlip) {
                board[fr][fc] = currentPlayer;
            }
        }
    }
    
    // 获取有效移动
    function getValidMoves() {
        const moves = [];
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (board[row][col] === null && isValidMove(row, col, currentPlayer)) {
                    moves.push(`${row},${col}`);
                }
            }
        }
        return moves;
    }
    
    // 获取指定玩家的有效移动
    function getValidMovesForPlayer(player) {
        const moves = [];
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (board[row][col] === null && isValidMove(row, col, player)) {
                    moves.push(`${row},${col}`);
                }
            }
        }
        return moves;
    }
    
    // 检查是否为有效移动
    function isValidMove(row, col, player) {
        if (board[row][col] !== null) return false;
        
        const directions = [
            [-1, -1], [-1, 0], [-1, 1],
            [0, -1],           [0, 1],
            [1, -1],  [1, 0],  [1, 1]
        ];
        
        for (const [dx, dy] of directions) {
            if (checkDirection(row, col, dx, dy, player)) {
                return true;
            }
        }
        return false;
    }
    
    // 检查特定方向是否有可翻转的棋子
    function checkDirection(row, col, dx, dy, player) {
        let r = row + dx;
        let c = col + dy;
        let foundOpponent = false;
        
        while (r >= 0 && r < BOARD_SIZE && c >= 0 && c < BOARD_SIZE) {
            if (board[r][c] === null) {
                return false;
            } else if (board[r][c] === player) {
                return foundOpponent;
            } else {
                foundOpponent = true;
            }
            
            r += dx;
            c += dy;
        }
        
        return false;
    }
    
    // 更新状态显示
    function updateStatus() {
        currentPlayerElement.textContent = currentPlayer === 'black' ? '黑子' : '白子';
        
        // 计算比分
        let blackCount = 0;
        let whiteCount = 0;
        for (let row = 0; row < BOARD_SIZE; row++) {
            for (let col = 0; col < BOARD_SIZE; col++) {
                if (board[row][col] === 'black') {
                    blackCount++;
                } else if (board[row][col] === 'white') {
                    whiteCount++;
                }
            }
        }
        
        blackScoreElement.textContent = blackCount;
        whiteScoreElement.textContent = whiteCount;
    }
    
    // 重置游戏
    resetButton.addEventListener('click', () => {
        initializeBoard();
        currentPlayer = 'black';
        gameActive = true;
        updateStatus();
        createBoard();
    });
    
    // 初始化游戏
    initializeBoard();
    updateStatus();
    createBoard();
});