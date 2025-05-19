document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');

    const widthInput = document.getElementById('mapWidth');
    const heightInput = document.getElementById('mapHeight');
    const createGridBtn = document.getElementById('createGridBtn');
    const asciiPreviewTextarea = document.getElementById('asciiPreview');
    const copyAsciiBtn = document.getElementById('copyAsciiBtn');
    const copyStatusMessage = document.getElementById('copyStatusMessage');

    let gridWidth = parseInt(widthInput.value);
    let gridHeight = parseInt(heightInput.value);
    const cellSize = 20; // Size of each cell in pixels
    let grid = []; // Stores the internal drawable map (0 for empty, 1 for wall)

    let mouseButtonPressed = null; // null = no button, 0 = left, 2 = right
    let lastProcessedCell = { row: null, col: null };

    let isConfirmingReset = false;
    let resetTimeoutId = null;
    const originalCreateBtnText = createGridBtn.textContent;
    const originalCreateBtnBgColor = window.getComputedStyle(createGridBtn).backgroundColor;

    function initializeGrid(width, height) {
        gridWidth = width;
        gridHeight = height;
        grid = Array(gridHeight).fill(null).map(() => Array(gridWidth).fill(0));

        // Canvas size includes the outer border
        canvas.width = (gridWidth + 2) * cellSize;
        canvas.height = (gridHeight + 2) * cellSize;

        drawCanvas();
        updateAsciiPreview();
    }

    function drawCanvas() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw grid cells
        for (let r = 0; r < gridHeight + 2; r++) {
            for (let c = 0; c < gridWidth + 2; c++) {
                ctx.fillStyle = '#555555'; // Wall color

                if (r === 0 || r === gridHeight + 1 || c === 0 || c === gridWidth + 1) {
                    // Outer border wall
                    ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
                } else {
                    // Inner grid
                    if (grid[r - 1][c - 1] === 1) { // Wall
                        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
                    } else { // Empty
                        ctx.fillStyle = '#e0e0e0'; // Empty cell color
                        ctx.fillRect(c * cellSize, r * cellSize, cellSize, cellSize);
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
        // Top border
        ascii += 'W'.repeat(gridWidth + 2) + '\n';

        for (let r = 0; r < gridHeight; r++) {
            ascii += 'W'; // Left border
            for (let c = 0; c < gridWidth; c++) {
                ascii += grid[r][c] === 1 ? 'W' : ' ';
            }
            ascii += 'W\n'; // Right border
        }

        // Bottom border
        ascii += 'W'.repeat(gridWidth + 2) + '\n';
        asciiPreviewTextarea.value = ascii.trim();

        // Adjust textarea attributes and style for full content visibility
        // asciiPreviewTextarea.cols = gridWidth + 2; // Removed, as CSS now handles width via white-space: pre and width: auto

        // Adjust textarea height to fit content
        asciiPreviewTextarea.style.height = 'auto'; // Reset height for accurate scrollHeight calculation
        asciiPreviewTextarea.style.height = (asciiPreviewTextarea.scrollHeight) + 'px';
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

    function handleInteraction(event) {
        const { row, col } = getMouseGridPos(event);
        let needsRedraw = false;

        let cellValueToSet;
        if (mouseButtonPressed === 0) { // Left button for drawing
            cellValueToSet = 1;
        } else if (mouseButtonPressed === 2) { // Right button for erasing
            cellValueToSet = 0;
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
        if (event.button === 0) { // Primary (left) button
            mouseButtonPressed = 0;
            handleInteraction(event);
        }
    });

    canvas.addEventListener('contextmenu', (event) => {
        event.preventDefault(); // Prevent context menu
        mouseButtonPressed = 2; // Right button
        handleInteraction(event);
    });

    canvas.addEventListener('mousemove', (event) => {
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

    // Initial setup
    initializeGrid(gridWidth, gridHeight);
});