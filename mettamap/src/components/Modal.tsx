"use client";
import { motion } from "framer-motion";
import {
  createContext,
  FC,
  forwardRef,
  PropsWithChildren,
  useContext,
  useEffect,
  useState,
} from "react";
import { createPortal } from "react-dom";

import { XIcon } from "@/icons/XIcon";

type ModalContextShape = {
  onClose: () => void;
};
const ModalContext = createContext<ModalContextShape>({
  onClose: () => undefined,
});

const Overlay: FC = () => {
  const { onClose } = useContext(ModalContext);
  return (
    <motion.div
      className="absolute inset-0 -z-10 bg-black"
      initial={{ opacity: 0 }}
      animate={{ opacity: 0.2 }}
      onClick={onClose}
    />
  );
};

const ModalHeader: FC<PropsWithChildren> = ({ children }) => {
  const { onClose } = useContext(ModalContext);
  return (
    <header className="flex items-center justify-between border-b border-gray-200 py-3 pr-4 pl-5 font-bold">
      <div>{children}</div>
      <button
        className="cursor-pointer bg-transparent text-slate-400 hover:text-slate-600"
        type="button"
        onClick={onClose}
      >
        <XIcon size={16} className="m-1" />
      </button>
    </header>
  );
};

// TODO - get rid of forwardRef, support `focus` and `{...hotkeys}` via smart props
const ModalBody = forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(function ModalBody(props, ref) {
  return <div ref={ref} className="overflow-auto px-5 py-3" {...props} />;
});

const ModalFooter: FC<PropsWithChildren> = ({ children }) => (
  <div className="flex justify-end gap-2 border-t border-gray-200 px-5 py-3">
    {children}
  </div>
);

const ModalWindow: FC<PropsWithChildren> = ({ children }) => {
  const naturalWidth = 576; // maximum possible width; modal tries to take this much space, but can be smaller

  return (
    <div
      className="shadow-toast flex flex-col overflow-auto rounded-md border bg-white"
      style={{ width: naturalWidth }}
    >
      {children}
    </div>
  );
};

type ModalType = FC<
  PropsWithChildren<{
    onClose: () => void;
  }>
> & {
  Body: typeof ModalBody;
  Footer: typeof ModalFooter;
  Header: typeof ModalHeader;
};

export const Modal: ModalType = ({ children, onClose }) => {
  const [el] = useState(() => document.createElement("div"));

  useEffect(() => {
    document.body.appendChild(el);
    return () => {
      document.body.removeChild(el);
    };
  }, [el]);

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        onClose();
      }
    };
    document.addEventListener("keydown", handleEscape);
    return () => {
      document.removeEventListener("keydown", handleEscape);
    };
  }, [onClose]);

  const modal = (
    <ModalContext.Provider value={{ onClose }}>
      <div>
        <div className="fixed inset-0 z-40 flex items-center justify-center">
          <Overlay />
          <ModalWindow>{children}</ModalWindow>
        </div>
      </div>
    </ModalContext.Provider>
  );

  return createPortal(modal, el);
};

Modal.Body = ModalBody;
Modal.Footer = ModalFooter;
Modal.Header = ModalHeader;
