document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');

    const widthInput = document.getElementById('mapWidth');
    const heightInput = document.getElementById('mapHeight');
    const createGridBtn = document.getElementById('createGridBtn');
    const asciiPreviewTextarea = document.getElementById('asciiPreview');
    const copyAsciiBtn = document.getElementById('copyAsciiBtn');
    const loadAsciiBtn = document.getElementById('loadAsciiBtn');
    const copyStatusMessage = document.getElementById('copyStatusMessage');
    const entitySelector = document.getElementById('entitySelector');
    const toggleAsciiEditBtn = document.getElementById('toggleAsciiEditBtn');

    let gridWidth = parseInt(widthInput.value);
    let gridHeight = parseInt(heightInput.value);
    const cellSize = 20; // Size of each cell in pixels
    let grid = []; // Stores the internal drawable map
    let selectedEntity = 'wall';
    let asciiEditMode = false;
    let asciiEditorInput = null;
    let editingCell = null;

    const asciiSymbols = {
        empty: ' ',
        wall: '#',
        'agent.agent': 'A',
        mine: 'g',
        generator: 'c',
        altar: 'a',
        armory: 'r',
        lasery: 'l',
        lab: 'b',
        factory: 'f',
        temple: 't',
        'agent.team_1': 'Q',
        'agent.team_2': 'E',
        'agent.team_3': 'R',
        'agent.team_4': 'T'
    };
    const asciiToObject = {};
    for (const [obj, ch] of Object.entries(asciiSymbols)) {
        asciiToObject[ch] = obj;
    }
    // Support old wall character
    asciiToObject['W'] = 'wall';

    const objectIcons = {
        wall: '../mettascope/data/objects/wall.png',
        'agent.agent': '../mettascope/data/objects/agent.png',
        mine: '../mettascope/data/objects/mine.png',
        generator: '../mettascope/data/objects/generator.png',
        altar: '../mettascope/data/objects/altar.png',
        armory: '../mettascope/data/objects/armory.png',
        lasery: '../mettascope/data/objects/lasery.png',
        lab: '../mettascope/data/objects/lab.png',
        factory: '../mettascope/data/objects/factory.png',
        temple: '../mettascope/data/objects/temple.png'
    };

    const teamColors = {
        'agent.team_1': '#d9534f', // red
        'agent.team_2': '#0275d8', // blue
        'agent.team_3': '#5cb85c', // green
        'agent.team_4': '#f0ad4e'  // orange
    };

    const objectImages = {};
    for (const [obj, src] of Object.entries(objectIcons)) {
        const img = new Image();
        img.src = src;
        objectImages[obj] = img;
    }

    function prepareTeamIcons(callback) {
        const baseSrc = objectIcons['agent.agent'];
        const baseImg = new Image();
        baseImg.src = baseSrc;
        baseImg.onload = () => {
            Object.entries(teamColors).forEach(([obj, color]) => {
                const canvasEl = document.createElement('canvas');
                canvasEl.width = baseImg.width;
                canvasEl.height = baseImg.height;
                const ictx = canvasEl.getContext('2d');
                ictx.drawImage(baseImg, 0, 0);
                ictx.globalCompositeOperation = 'source-atop';
                ictx.fillStyle = color;
                ictx.fillRect(0, 0, canvasEl.width, canvasEl.height);
                const img = new Image();
                img.src = canvasEl.toDataURL();
                objectImages[obj] = img;
                objectIcons[obj] = img.src;
            });
            if (typeof callback === 'function') callback();
        };
        baseImg.onerror = () => {
            if (typeof callback === 'function') callback();
        };
    }

    function createEntityButtons() {
        Object.entries(objectIcons).forEach(([obj, src], index) => {
            const btn = document.createElement('button');
            btn.className = 'entity-btn';
            btn.dataset.entity = obj;
            const img = document.createElement('img');
            img.src = src;
            img.alt = obj;
            btn.appendChild(img);
            btn.addEventListener('click', () => {
                selectedEntity = obj;
                document.querySelectorAll('.entity-btn').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
            });
            entitySelector.appendChild(btn);
            if (index === 0) btn.classList.add('selected');
        });
    }

    let mouseButtonPressed = null; // null = no button, 0 = left, 2 = right
    let lastProcessedCell = { row: null, col: null };

    let isConfirmingReset = false;
    let resetTimeoutId = null;
    const originalCreateBtnText = createGridBtn.textContent;
    const originalCreateBtnBgColor = window.getComputedStyle(createGridBtn).backgroundColor;

    function initializeGrid(width, height) {
        gridWidth = width;
        gridHeight = height;
        grid = Array(gridHeight)
            .fill(null)
            .map(() => Array(gridWidth).fill(asciiSymbols.empty));

        // Canvas size includes the outer border
        canvas.width = (gridWidth + 2) * cellSize;
        canvas.height = (gridHeight + 2) * cellSize;

        drawCanvas();
        updateAsciiPreview();
    }

    function drawCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (let r = 0; r < gridHeight + 2; r++) {
            for (let c = 0; c < gridWidth + 2; c++) {
                const char = (r === 0 || r === gridHeight + 1 || c === 0 || c === gridWidth + 1)
                    ? asciiSymbols.wall
                    : grid[r - 1][c - 1];
                const obj = asciiToObject[char] || 'empty';

                if (asciiEditMode) {
                    ctx.fillStyle = '#e0e0e0';
                    ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
                    if (char !== asciiSymbols.empty) {
                        ctx.fillStyle = '#000';
                        ctx.font = '16px monospace';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(char, c * cellSize + cellSize / 2, r * cellSize + cellSize / 2);
                    }
                } else if (obj !== 'empty' && objectImages[obj]) {
                    ctx.drawImage(objectImages[obj], c * cellSize, r * cellSize, cellSize, cellSize);
                } else {
                    ctx.fillStyle = '#e0e0e0';
                    ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
                    if (char !== asciiSymbols.empty) {
                        ctx.fillStyle = '#000';
                        ctx.font = '16px monospace';
                        ctx.textAlign = 'center';
                        ctx.textBaseline = 'middle';
                        ctx.fillText(char, c * cellSize + cellSize / 2, r * cellSize + cellSize / 2);
                    }
                }
            }
        }

        // Draw grid lines
        ctx.strokeStyle = '#ccc';
        ctx.lineWidth = 1;
        for (let r = 0; r <= gridHeight + 2; r++) {
            ctx.beginPath();
            ctx.moveTo(0, r * cellSize);
            ctx.lineTo(canvas.width, r * cellSize);
            ctx.stroke();
        }
        for (let c = 0; c <= gridWidth + 2; c++) {
            ctx.beginPath();
            ctx.moveTo(c * cellSize, 0);
            ctx.lineTo(c * cellSize, canvas.height);
            ctx.stroke();
        }
    }

    function updateAsciiPreview() {
        let ascii = '';
        const wallChar = asciiSymbols.wall;
        ascii += wallChar.repeat(gridWidth + 2) + '\n';

        for (let r = 0; r < gridHeight; r++) {
            ascii += wallChar;
            for (let c = 0; c < gridWidth; c++) {
                ascii += grid[r][c];
            }
            ascii += wallChar + '\n';
        }

        ascii += wallChar.repeat(gridWidth + 2) + '\n';
        asciiPreviewTextarea.value = ascii.trim();

        // Adjust textarea attributes and style for full content visibility
        // asciiPreviewTextarea.cols = gridWidth + 2; // Removed, as CSS now handles width via white-space: pre and width: auto

        // Adjust textarea height to fit content
        asciiPreviewTextarea.style.height = 'auto'; // Reset height for accurate scrollHeight calculation
        asciiPreviewTextarea.style.height = (asciiPreviewTextarea.scrollHeight) + 'px';
    }

    function loadFromAscii(text) {
        const lines = text.trim().split(/\r?\n/).filter(l => l.length);
        if (lines.length < 3) {
            return false;
        }
        const innerWidth = lines[0].length - 2;
        const innerHeight = lines.length - 2;
        if (innerWidth <= 0 || innerHeight <= 0) {
            return false;
        }
        for (const line of lines) {
            if (line.length !== innerWidth + 2) return false;
        }
        initializeGrid(innerWidth, innerHeight);
        for (let r = 0; r < innerHeight; r++) {
            const row = lines[r + 1];
            for (let c = 0; c < innerWidth; c++) {
                grid[r][c] = row[c + 1];
            }
        }
        drawCanvas();
        updateAsciiPreview();
        return true;
    }

    function getMouseGridPos(event) {
        const rect = canvas.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;

        const col = Math.floor(mouseX / cellSize);
        const row = Math.floor(mouseY / cellSize);

        // Return coordinates for the *internal* grid
        return { row: row - 1, col: col - 1 };
    }

    // Function to apply change to a single cell
    function applyToCell(row, col, cellValueToSet) {
        if (row >= 0 && row < gridHeight && col >= 0 && col < gridWidth) {
            if (grid[row][col] !== cellValueToSet) {
                grid[row][col] = cellValueToSet;
                return true; // Indicates a change was made
            }
        }
        return false; // No change
    }

    // Bresenham's line algorithm or similar simple grid line interpolation
    function drawLine(r0, c0, r1, c1, cellValueToSet) {
        let changed = false;
        const dr = Math.abs(r1 - r0);
        const dc = Math.abs(c1 - c0);
        const sr = (r0 < r1) ? 1 : -1;
        const sc = (c0 < c1) ? 1 : -1;
        let err = dr - dc;

        let r = r0;
        let c = c0;

        while (true) {
            if (applyToCell(r, c, cellValueToSet)) changed = true;
            if (r === r1 && c === c1) break;
            const e2 = 2 * err;
            if (e2 > -dc) {
                err -= dc;
                r += sr;
            }
            if (e2 < dr) {
                err += dr;
                c += sc;
            }
        }
        return changed;
    }

    function openAsciiEditor(row, col) {
        closeAsciiEditor(false);
        editingCell = { row, col };
        if (!asciiEditorInput) {
            asciiEditorInput = document.createElement('input');
            asciiEditorInput.maxLength = 1;
            asciiEditorInput.style.position = 'absolute';
            asciiEditorInput.style.textAlign = 'center';
            asciiEditorInput.style.fontFamily = 'monospace';
            asciiEditorInput.style.padding = '0';
            asciiEditorInput.style.margin = '0';
            asciiEditorInput.style.border = '1px solid #666';
            asciiEditorInput.style.boxSizing = 'border-box';
            asciiEditorInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter') {
                    closeAsciiEditor(true);
                } else if (e.key === 'Escape') {
                    closeAsciiEditor(false);
                }
            });
            asciiEditorInput.addEventListener('blur', () => closeAsciiEditor(true));
        }
        asciiEditorInput.style.width = cellSize + 'px';
        asciiEditorInput.style.height = cellSize + 'px';
        asciiEditorInput.style.left = canvas.offsetLeft + (col + 1) * cellSize + 'px';
        asciiEditorInput.style.top = canvas.offsetTop + (row + 1) * cellSize + 'px';
        asciiEditorInput.value = grid[row][col];
        document.body.appendChild(asciiEditorInput);
        asciiEditorInput.focus();
        asciiEditorInput.select();
    }

    function closeAsciiEditor(commit) {
        if (!editingCell) return;
        if (commit) {
            const ch = asciiEditorInput.value ? asciiEditorInput.value[0] : asciiSymbols.empty;
            grid[editingCell.row][editingCell.col] = ch;
            updateAsciiPreview();
            drawCanvas();
        }
        asciiEditorInput.remove();
        editingCell = null;
    }

    function handleInteraction(event) {
        if (asciiEditMode) return;
        const { row, col } = getMouseGridPos(event);
        let needsRedraw = false;

        let cellValueToSet;
        if (mouseButtonPressed === 0) { // Left button for drawing
            cellValueToSet = asciiSymbols[selectedEntity];
        } else if (mouseButtonPressed === 2) { // Right button for erasing
            cellValueToSet = asciiSymbols.empty;
        } else {
            return; // No button we care about is pressed
        }

        if (event.type === 'mousedown' || event.type === 'contextmenu') {
            // Single point for initial click
            if (applyToCell(row, col, cellValueToSet)) {
                needsRedraw = true;
            }
            lastProcessedCell = { row, col };
        } else if (event.type === 'mousemove' && mouseButtonPressed !== null) {
            if (lastProcessedCell.row !== null && (lastProcessedCell.row !== row || lastProcessedCell.col !== col)) {
                if (drawLine(lastProcessedCell.row, lastProcessedCell.col, row, col, cellValueToSet)) {
                    needsRedraw = true;
                }
                lastProcessedCell = { row, col };
            } else if (lastProcessedCell.row === null) { // Mouse moved onto canvas while button already pressed
                 if (applyToCell(row, col, cellValueToSet)) {
                    needsRedraw = true;
                 }
                 lastProcessedCell = { row, col };
            }
        }

        if (needsRedraw) {
            drawCanvas();
            updateAsciiPreview();
        }
    }

    canvas.addEventListener('mousedown', (event) => {
        if (asciiEditMode) return;
        if (event.button === 0) { // Primary (left) button
            mouseButtonPressed = 0;
            handleInteraction(event);
        }
    });

    canvas.addEventListener('contextmenu', (event) => {
        if (asciiEditMode) return;
        event.preventDefault(); // Prevent context menu
        mouseButtonPressed = 2; // Right button
        handleInteraction(event);
    });

    canvas.addEventListener('mousemove', (event) => {
        if (asciiEditMode) return;
        if (mouseButtonPressed !== null) { // If left or right button is held down
            handleInteraction(event);
        }
    });

    canvas.addEventListener('mouseup', (event) => {
        mouseButtonPressed = null;
        lastProcessedCell = { row: null, col: null };
    });

    canvas.addEventListener('mouseleave', () => {
        mouseButtonPressed = null;
        lastProcessedCell = { row: null, col: null };
    });

    canvas.addEventListener('click', (event) => {
        if (!asciiEditMode) return;
        const { row, col } = getMouseGridPos(event);
        if (row >= 0 && row < gridHeight && col >= 0 && col < gridWidth) {
            openAsciiEditor(row, col);
        }
    });

    copyAsciiBtn.addEventListener('click', () => {
        asciiPreviewTextarea.select();
        navigator.clipboard.writeText(asciiPreviewTextarea.value)
            .then(() => {
                copyStatusMessage.textContent = 'Copied!';
                copyStatusMessage.style.color = '#28a745'; // Ensure default color
                copyStatusMessage.style.visibility = 'visible';
                setTimeout(() => {
                    copyStatusMessage.style.visibility = 'hidden';
                }, 2000); // Hide message after 2 seconds
            })
            .catch(err => {
                copyStatusMessage.textContent = 'Failed to copy!';
                copyStatusMessage.style.color = 'red'; // Indicate error
                copyStatusMessage.style.visibility = 'visible';
                console.error('Failed to copy ASCII map: ', err);
                setTimeout(() => {
                    copyStatusMessage.style.visibility = 'hidden';
                    // copyStatusMessage.style.color = ''; // Color will be reset on next success
                }, 3000); // Hide error message after 3 seconds
            });
    });

    createGridBtn.addEventListener('click', () => {
        if (isConfirmingReset) {
            // Clear confirmation state
            clearTimeout(resetTimeoutId);
            document.removeEventListener('click', handleOutsideClickForReset, true);
            resetCreateButtonState();
            isConfirmingReset = false;

            // Proceed with reset
            const newWidth = parseInt(widthInput.value);
            const newHeight = parseInt(heightInput.value);
            if (newWidth >= 3 && newHeight >= 3 && newWidth <= 100 && newHeight <= 100) {
                initializeGrid(newWidth, newHeight);
            } else {
                alert('Width and Height must be between 3 and 100.');
            }
        } else {
            // Enter confirmation state
            isConfirmingReset = true;
            createGridBtn.textContent = 'Are you sure?';
            createGridBtn.style.backgroundColor = 'red';

            // Set timeout to revert
            resetTimeoutId = setTimeout(() => {
                resetCreateButtonState();
                isConfirmingReset = false;
                document.removeEventListener('click', handleOutsideClickForReset, true);
            }, 5000);

            // Add listener for outside click
            // Use `true` for capture phase to ensure it runs before other click listeners
            // that might stop propagation.
            document.addEventListener('click', handleOutsideClickForReset, true);
        }
    });

    function resetCreateButtonState() {
        createGridBtn.textContent = originalCreateBtnText;
        createGridBtn.style.backgroundColor = originalCreateBtnBgColor;
        if (resetTimeoutId) {
            clearTimeout(resetTimeoutId);
            resetTimeoutId = null;
        }
    }

    function handleOutsideClickForReset(event) {
        if (isConfirmingReset && event.target !== createGridBtn) {
            resetCreateButtonState();
            isConfirmingReset = false;
            document.removeEventListener('click', handleOutsideClickForReset, true);
        }
    }

    loadAsciiBtn.addEventListener('click', () => {
        const text = asciiPreviewTextarea.value;
        if (!loadFromAscii(text)) {
            alert('Invalid ASCII map format.');
        }
    });

    toggleAsciiEditBtn.addEventListener('click', () => {
        asciiEditMode = !asciiEditMode;
        toggleAsciiEditBtn.classList.toggle('active', asciiEditMode);
        if (!asciiEditMode) {
            closeAsciiEditor(true);
        }
        drawCanvas();
    });

    // Initial setup
    prepareTeamIcons(() => {
        createEntityButtons();
        initializeGrid(gridWidth, gridHeight);
    });
});