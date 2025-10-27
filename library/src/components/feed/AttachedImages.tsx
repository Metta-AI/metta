"use client";

interface AttachedImagesProps {
  images: string[];
  onImageClick: (imageIndex: number) => void;
}

export function AttachedImages({ images, onImageClick }: AttachedImagesProps) {
  if (!images || images.length === 0) return null;

  return (
    <div className="px-4 pb-2">
      <div
        className={`grid gap-2 ${
          images.length === 1
            ? "grid-cols-1"
            : images.length === 2
              ? "grid-cols-2"
              : "grid-cols-2 sm:grid-cols-3"
        }`}
      >
        {images.map((imageUrl, index) => (
          <div
            key={index}
            data-image-container="true"
            className="group relative overflow-hidden rounded-lg border border-gray-200 bg-gray-50"
            onClick={(e) => {
              e.stopPropagation();
              onImageClick(index);
            }}
            style={{ cursor: "pointer" }}
          >
            <img
              src={imageUrl}
              alt={`Attached image ${index + 1}`}
              className="h-32 w-full object-cover transition-transform group-hover:scale-105 sm:h-40"
            />
          </div>
        ))}
      </div>
    </div>
  );
}
