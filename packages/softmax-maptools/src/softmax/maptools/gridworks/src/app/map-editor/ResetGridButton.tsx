"use client";
import { FC, useState } from "react";

import { Button } from "@/components/Button";
import { Modal } from "@/components/Modal";
import { NumberInput } from "@/components/NumberInput";
import { MettaGrid } from "@/lib/MettaGrid";

const ResetGridModal: FC<{
  isOpen: boolean;
  onClose: () => void;
  onCreateGrid: (width: number, height: number) => void;
  currentWidth: number;
  currentHeight: number;
}> = ({ isOpen, onClose, onCreateGrid, currentWidth, currentHeight }) => {
  const [width, setWidth] = useState(currentWidth);
  const [height, setHeight] = useState(currentHeight);

  if (!isOpen) return null;

  return (
    <Modal onClose={onClose}>
      <Modal.Header>Reset Grid</Modal.Header>
      <Modal.Body>
        <label>
          <span className="mb-1 block text-sm">Width:</span>
          <NumberInput
            value={width}
            onChange={(e) => setWidth(+e.target.value)}
          />
        </label>
        <label>
          <span className="mb-1 block text-sm">Height:</span>
          <NumberInput
            value={height}
            onChange={(e) => setHeight(+e.target.value)}
          />
        </label>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          onClick={() => {
            onCreateGrid(width, height);
            onClose();
          }}
        >
          Create Grid
        </Button>
      </Modal.Footer>
    </Modal>
  );
};
export const ResetGridButton: FC<{
  currentGrid: MettaGrid;
  setGrid: (grid: MettaGrid) => void;
}> = ({ currentGrid, setGrid }) => {
  const [showResetModal, setShowResetModal] = useState(false);

  const handleCreateGrid = (width: number, height: number) => {
    setGrid(MettaGrid.empty(width, height));
  };

  return (
    <>
      <Button onClick={() => setShowResetModal(true)}>Reset Grid</Button>
      <ResetGridModal
        isOpen={showResetModal}
        onClose={() => setShowResetModal(false)}
        onCreateGrid={handleCreateGrid}
        currentWidth={currentGrid.width}
        currentHeight={currentGrid.height}
      />
    </>
  );
};
