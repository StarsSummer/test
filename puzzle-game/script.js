// 游戏配置
const config = {
    rows: 4,
    cols: 4,
    moves: 0,
    startTime: null,
    timerInterval: null,
    gameActive: false,
    pieces: [],
    emptyIndex: 0,
    imageUrl: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=600&q=80' // 使用一个示例图片，你也可以替换为本地图片
};

// DOM 元素
const puzzleContainer = document.getElementById('puzzleContainer');
const movesDisplay = document.getElementById('moves');
const timerDisplay = document.getElementById('timer');
const shuffleBtn = document.getElementById('shuffleBtn');
const resetBtn = document.getElementById('resetBtn');
const difficultySelect = document.getElementById('difficulty');
const messageDisplay = document.getElementById('message');

// 初始化游戏
function initGame() {
    config.rows = parseInt(difficultySelect.value);
    config.cols = parseInt(difficultySelect.value);
    config.moves = 0;
    config.gameActive = true;
    config.startTime = new Date();
    movesDisplay.textContent = config.moves;
    messageDisplay.textContent = '';
    messageDisplay.classList.remove('won');
    
    if (config.timerInterval) {
        clearInterval(config.timerInterval);
    }
    
    config.timerInterval = setInterval(updateTimer, 1000);
    updateTimer();
    
    createPuzzle();
}

// 创建拼图
function createPuzzle() {
    puzzleContainer.innerHTML = '';
    puzzleContainer.style.gridTemplateColumns = `repeat(${config.cols}, 1fr)`;
    puzzleContainer.style.gridTemplateRows = `repeat(${config.rows}, 1fr)`;
    
    const totalPieces = config.rows * config.cols;
    config.pieces = [];
    
    // 创建拼图块
    for (let i = 0; i < totalPieces - 1; i++) {
        const piece = document.createElement('div');
        piece.className = 'puzzle-piece';
        
        // 计算每个拼图块在原图中的位置
        const row = Math.floor(i / config.cols);
        const col = i % config.cols;
        
        // 设置背景图片及其位置
        piece.style.backgroundImage = `url('${config.imageUrl}')`;
        piece.style.backgroundSize = `${config.cols * 100}% ${config.rows * 100}%`;
        piece.style.backgroundPosition = `-${col * (100 / config.cols)}% -${row * (100 / config.rows)}%`;
        
        piece.dataset.index = i;
        piece.dataset.correctIndex = i;
        
        piece.addEventListener('click', () => movePiece(i));
        
        config.pieces.push(piece);
        puzzleContainer.appendChild(piece);
    }
    
    // 添加空白块
    config.emptyIndex = totalPieces - 1;
    const emptyPiece = document.createElement('div');
    emptyPiece.className = 'puzzle-piece empty';
    emptyPiece.dataset.index = config.emptyIndex;
    puzzleContainer.appendChild(emptyPiece);
    
    // 洗牌
    shufflePuzzle();
}

// 洗牌
function shufflePuzzle() {
    const totalPieces = config.rows * config.cols;
    for (let i = 0; i < 1000; i++) {
        const movableIndices = getMovableIndices();
        if (movableIndices.length > 0) {
            const randomIndex = movableIndices[Math.floor(Math.random() * movableIndices.length)];
            movePieceByIndex(randomIndex);
        }
    }
}

// 获取可移动的拼图块索引
function getMovableIndices() {
    const movableIndices = [];
    const emptyRow = Math.floor(config.emptyIndex / config.cols);
    const emptyCol = config.emptyIndex % config.cols;
    
    // 检查四个方向
    const directions = [
        config.emptyIndex - config.cols, // 上
        config.emptyIndex + config.cols, // 下
        config.emptyIndex - 1,          // 左
        config.emptyIndex + 1           // 右
    ];
    
    for (const index of directions) {
        if (index >= 0 && index < config.pieces.length + 1) {
            const row = Math.floor(index / config.cols);
            const col = index % config.cols;
            
            // 检查是否在相邻位置
            if ((Math.abs(row - emptyRow) === 1 && col === emptyCol) || 
                (Math.abs(col - emptyCol) === 1 && row === emptyRow)) {
                movableIndices.push(index);
            }
        }
    }
    
    return movableIndices;
}

// 移动拼图块
function movePiece(index) {
    if (!config.gameActive) return;
    
    // 检查是否可以移动
    const movableIndices = getMovableIndices();
    if (!movableIndices.includes(index)) return;
    
    movePieceByIndex(index);
    config.moves++;
    movesDisplay.textContent = config.moves;
    
    // 检查是否完成
    checkWin();
}

// 通过索引移动拼图块
function movePieceByIndex(index) {
    const piece = config.pieces[index] || 
                  Array.from(puzzleContainer.children).find(child => 
                      parseInt(child.dataset.index) === config.emptyIndex);
    
    const emptyPiece = Array.from(puzzleContainer.children).find(child => 
        parseInt(child.dataset.index) === config.emptyIndex);
    
    // 交换位置
    const tempIndex = piece.dataset.index;
    piece.dataset.index = emptyPiece.dataset.index;
    emptyPiece.dataset.index = tempIndex;
    
    // 如果移动的是拼图块而不是空块，则交换DOM位置
    if (piece !== emptyPiece) {
        const pieceIndex = Array.from(puzzleContainer.children).indexOf(piece);
        const emptyIndex = Array.from(puzzleContainer.children).indexOf(emptyPiece);
        
        // 交换DOM位置
        if (pieceIndex < emptyIndex) {
            puzzleContainer.insertBefore(emptyPiece, piece);
        } else {
            puzzleContainer.insertBefore(emptyPiece, piece.nextSibling);
        }
    }
    
    // 更新空块索引
    config.emptyIndex = parseInt(emptyPiece.dataset.index);
}

// 检查是否获胜
function checkWin() {
    const pieces = Array.from(puzzleContainer.children);
    
    // 检查除了最后一个（空块）之外的所有块是否在正确的位置
    for (let i = 0; i < pieces.length - 1; i++) {
        if (parseInt(pieces[i].dataset.index) !== i) {
            return false;
        }
    }
    
    // 游戏结束
    config.gameActive = false;
    clearInterval(config.timerInterval);
    messageDisplay.textContent = `恭喜！你用了 ${config.moves} 步，耗时 ${timerDisplay.textContent} 完成了拼图！`;
    messageDisplay.classList.add('won');
    
    return true;
}

// 更新计时器
function updateTimer() {
    if (config.startTime) {
        const now = new Date();
        const diff = Math.floor((now - config.startTime) / 1000);
        const minutes = Math.floor(diff / 60).toString().padStart(2, '0');
        const seconds = (diff % 60).toString().padStart(2, '0');
        timerDisplay.textContent = `${minutes}:${seconds}`;
    }
}

// 事件监听器
shuffleBtn.addEventListener('click', () => {
    if (config.timerInterval) {
        clearInterval(config.timerInterval);
    }
    initGame();
});

resetBtn.addEventListener('click', () => {
    if (config.timerInterval) {
        clearInterval(config.timerInterval);
    }
    config.moves = 0;
    movesDisplay.textContent = config.moves;
    messageDisplay.textContent = '';
    messageDisplay.classList.remove('won');
    initGame();
});

difficultySelect.addEventListener('change', () => {
    if (config.timerInterval) {
        clearInterval(config.timerInterval);
    }
    initGame();
});

// 启动游戏
window.addEventListener('DOMContentLoaded', initGame);