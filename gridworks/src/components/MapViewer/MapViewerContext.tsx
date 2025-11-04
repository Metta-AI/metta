import { createContext, FC, ReactNode, useState } from "react";

export const MapViewerContext = createContext<{
  showDebugInfo: boolean;
  setShowDebugInfo: (showDebugInfo: boolean) => void;
  showHoverInfo: boolean;
  setShowHoverInfo: (showHoverInfo: boolean) => void;
}>({
  showDebugInfo: false,
  setShowDebugInfo: () => {},
  showHoverInfo: false,
  setShowHoverInfo: () => {},
});

export const MapViewerContextProvider: FC<{
  children: ReactNode;
}> = ({ children }) => {
  const [showDebugInfo, setShowDebugInfo] = useState(false);
  const [showHoverInfo, setShowHoverInfo] = useState(false);
  return (
    <MapViewerContext.Provider
      value={{
        showDebugInfo,
        setShowDebugInfo,
        showHoverInfo,
        setShowHoverInfo,
      }}
    >
      {children}
    </MapViewerContext.Provider>
  );
};
