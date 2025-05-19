document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('mapCanvas');
    const ctx = canvas.getContext('2d');

    const widthInput = document.getElementById('mapWidth');
    const heightInput = document.getElementById('mapHeight');
    const createGridBtn = document.getElementById('createGridBtn');
    const modeToggleBtn = document.getElementById('modeToggleBtn');
    const asciiPreviewTextarea = document.getElementById('asciiPreview');
    const copyAsciiBtn = document.getElementById('copyAsciiBtn');
    const copyStatusMessage = document.getElementById('copyStatusMessage');

    let gridWidth = parseInt(widthInput.value);
    let gridHeight = parseInt(heightInput.value);
    const cellSize = 20; // Size of each cell in pixels
    let grid = []; // Stores the internal drawable map (0 for empty, 1 for wall)

    let isDrawingMode = true; // true for draw wall, false for delete wall
    let isMouseDown = false;

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

    function handleCellChange(event) {
        const { row, col } = getMouseGridPos(event);

        // Check if within the internal drawable grid
        if (row >= 0 && row < gridHeight && col >= 0 && col < gridWidth) {
            const currentVal = grid[row][col];
            const newVal = isDrawingMode ? 1 : 0;
            if (currentVal !== newVal) {
                grid[row][col] = newVal;
                drawCanvas();
                updateAsciiPreview();
            }
        }
    }

    canvas.addEventListener('mousedown', (event) => {
        isMouseDown = true;
        handleCellChange(event);
    });

    canvas.addEventListener('mousemove', (event) => {
        if (isMouseDown) {
            handleCellChange(event);
        }
    });

    canvas.addEventListener('mouseup', () => {
        isMouseDown = false;
    });

    canvas.addEventListener('mouseleave', () => {
        isMouseDown = false; // Stop drawing if mouse leaves canvas while pressed
    });

    modeToggleBtn.addEventListener('click', () => {
        isDrawingMode = !isDrawingMode;
        modeToggleBtn.textContent = isDrawingMode ? 'Mode: Draw Wall' : 'Mode: Erase Wall';
        modeToggleBtn.style.backgroundColor = isDrawingMode ? '#007bff' : '#dc3545';
    });

    copyAsciiBtn.addEventListener('click', () => {
        asciiPreviewTextarea.select();
        navigator.clipboard.writeText(asciiPreviewTextarea.value)
            .then(() => {
                copyStatusMessage.textContent = 'Copied!';
                setTimeout(() => {
                    copyStatusMessage.textContent = '';
                }, 2000); // Clear message after 2 seconds
            })
            .catch(err => {
                copyStatusMessage.textContent = 'Failed to copy!';
                copyStatusMessage.style.color = 'red'; // Indicate error
                console.error('Failed to copy ASCII map: ', err);
                setTimeout(() => {
                    copyStatusMessage.textContent = '';
                    copyStatusMessage.style.color = ''; // Reset color
                }, 3000); // Clear error message after 3 seconds
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