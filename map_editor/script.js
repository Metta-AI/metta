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

        // Determine action based on which mouse button is logically down for this event stream
        let cellValueToSet;
        if (event.type === 'contextmenu' || mouseButtonPressed === 2) { // Erasing for initial right-click or right-drag
            cellValueToSet = 0;
        } else if (event.type === 'mousedown' || mouseButtonPressed === 0) { // Drawing for initial left-click or left-drag
            cellValueToSet = 1;
        } else {
            return; // Should not happen if mouseButtonPressed is correctly managed
        }

        // Check if within the internal drawable grid
        if (row >= 0 && row < gridHeight && col >= 0 && col < gridWidth) {
            const currentVal = grid[row][col];
            if (currentVal !== cellValueToSet) {
                grid[row][col] = cellValueToSet;
                drawCanvas();
                updateAsciiPreview();
            }
        }
    }

    canvas.addEventListener('mousedown', (event) => {
        if (event.button === 0) { // Primary (left) button
            mouseButtonPressed = 0;
            handleCellChange(event);
        }
        // Middle button (event.button === 1) is ignored
    });

    canvas.addEventListener('contextmenu', (event) => {
        event.preventDefault(); // Prevent context menu
        mouseButtonPressed = 2; // Right button
        handleCellChange(event);
    });

    canvas.addEventListener('mousemove', (event) => {
        if (mouseButtonPressed !== null) { // If left or right button is held down
            handleCellChange(event);
        }
    });

    canvas.addEventListener('mouseup', (event) => {
        // Check which button was released if needed, but generally reset for any mouseup
        mouseButtonPressed = null;
    });

    canvas.addEventListener('mouseleave', () => {
        mouseButtonPressed = null; // Stop drawing/erasing if mouse leaves canvas while pressed
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