"use client";
import { FC, useCallback, useState } from "react";

import { Button } from "@/components/Button";
import { Modal } from "@/components/Modal";
import { NumberInput } from "@/components/NumberInput";
import { useGlobalShortcuts } from "@/hooks/useGlobalShortcut";
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

  const onSubmit = useCallback(() => {
    onCreateGrid(width, height);
    onClose();
  }, [onCreateGrid, onClose, width, height]);

  useGlobalShortcuts(isOpen ? [[{ key: "Enter" }, onSubmit]] : []);

  if (!isOpen) return null;

  return (
    <Modal onClose={onClose}>
      <Modal.Header>Reset Grid</Modal.Header>
      <Modal.Body>
        <div className="flex flex-col gap-2">
          <label>
            <span className="mb-1 block text-sm font-semibold">Width:</span>
            <NumberInput
              value={width}
              onChange={(e) => setWidth(+e.target.value)}
            />
          </label>
          <label>
            <span className="mb-1 block text-sm font-semibold">Height:</span>
            <NumberInput
              value={height}
              onChange={(e) => setHeight(+e.target.value)}
            />
          </label>
        </div>
      </Modal.Body>
      <Modal.Footer>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={onSubmit} theme="primary">
          Create Grid
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export const ResetGridButton: FC<{
  currentGrid: MettaGrid;
  setGrid: (grid: MettaGrid) => void;
  onModalStateChange: (isOpen: boolean) => void;
}> = ({ currentGrid, setGrid, onModalStateChange }) => {
  const [showModal, setShowModal] = useState(false);

  const handleCreateGrid = (width: number, height: number) => {
    setGrid(MettaGrid.empty(width, height));
  };

  const handleShowModal = useCallback(
    (isOpen: boolean) => {
      setShowModal(isOpen);
      onModalStateChange(isOpen);
    },
    [onModalStateChange]
  );

  return (
    <>
      <Button onClick={() => handleShowModal(true)}>Reset Grid</Button>
      <ResetGridModal
        isOpen={showModal}
        onClose={() => handleShowModal(false)}
        onCreateGrid={handleCreateGrid}
        currentWidth={currentGrid.width}
        currentHeight={currentGrid.height}
      />
    </>
  );
};
