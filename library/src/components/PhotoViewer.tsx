"use client";

import { FC, useEffect, useState, useCallback } from "react";
import { X, ChevronLeft, ChevronRight } from "lucide-react";

interface PhotoViewerProps {
  images: string[];
  initialIndex?: number;
  isOpen: boolean;
  onClose: () => void;
  postAuthor?: string;
}

export const PhotoViewer: FC<PhotoViewerProps> = ({
  images,
  initialIndex = 0,
  isOpen,
  onClose,
  postAuthor,
}) => {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [isLoading, setIsLoading] = useState(false);

  // Reset current index when viewer opens
  useEffect(() => {
    if (isOpen) {
      setCurrentIndex(initialIndex);
    }
  }, [isOpen, initialIndex]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!isOpen) return;

      switch (e.key) {
        case "Escape":
          onClose();
          break;
        case "ArrowLeft":
          e.preventDefault();
          navigatePrevious();
          break;
        case "ArrowRight":
          e.preventDefault();
          navigateNext();
          break;
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, currentIndex, images.length]);

  // Prevent body scroll when modal is open
  useEffect(() => {
    if (isOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "unset";
    }

    return () => {
      document.body.style.overflow = "unset";
    };
  }, [isOpen]);

  const navigateNext = useCallback(() => {
    setCurrentIndex((prev) => (prev + 1) % images.length);
  }, [images.length]);

  const navigatePrevious = useCallback(() => {
    setCurrentIndex((prev) => (prev - 1 + images.length) % images.length);
  }, [images.length]);

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  const handleImageLoad = () => {
    setIsLoading(false);
  };

  const handleImageLoadStart = () => {
    setIsLoading(true);
  };

  if (!isOpen || images.length === 0) return null;

  const currentImage = images[currentIndex];
  const hasMultipleImages = images.length > 1;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/90 backdrop-blur-sm"
        onClick={handleBackdropClick}
      />

      {/* Content */}
      <div className="relative z-10 flex h-full w-full items-center justify-center p-4">
        {/* Close button */}
        <button
          onClick={(e) => {
            e.stopPropagation();
            onClose();
          }}
          className="absolute top-4 right-4 z-20 flex h-10 w-10 items-center justify-center rounded-full bg-black/50 text-white transition-colors hover:bg-black/70"
          aria-label="Close photo viewer"
        >
          <X className="h-5 w-5" />
        </button>

        {/* Previous button */}
        {hasMultipleImages && currentIndex > 0 && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              navigatePrevious();
            }}
            className="absolute top-1/2 left-4 z-20 flex h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-black/50 text-white transition-colors hover:bg-black/70"
            aria-label="Previous image"
          >
            <ChevronLeft className="h-6 w-6" />
          </button>
        )}

        {/* Next button */}
        {hasMultipleImages && currentIndex < images.length - 1 && (
          <button
            onClick={(e) => {
              e.stopPropagation();
              navigateNext();
            }}
            className="absolute top-1/2 right-4 z-20 flex h-12 w-12 -translate-y-1/2 items-center justify-center rounded-full bg-black/50 text-white transition-colors hover:bg-black/70"
            aria-label="Next image"
          >
            <ChevronRight className="h-6 w-6" />
          </button>
        )}

        {/* Loading indicator */}
        {isLoading && (
          <div className="absolute inset-0 flex items-center justify-center">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-white border-t-transparent" />
          </div>
        )}

        {/* Image */}
        <div className="flex max-h-full max-w-full items-center justify-center">
          <img
            src={currentImage}
            alt={`Image ${currentIndex + 1} of ${images.length}`}
            className="max-h-[90vh] max-w-[90vw] object-contain"
            onLoad={handleImageLoad}
            onLoadStart={handleImageLoadStart}
            style={{ display: isLoading ? "none" : "block" }}
          />
        </div>

        {/* Image info */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-lg bg-black/50 px-3 py-2 text-sm text-white backdrop-blur-sm">
          {hasMultipleImages ? (
            <span>
              {currentIndex + 1} of {images.length}
            </span>
          ) : (
            <span>Image</span>
          )}
          {postAuthor && (
            <span className="ml-2 opacity-75">by {postAuthor}</span>
          )}
        </div>
      </div>
    </div>
  );
};
