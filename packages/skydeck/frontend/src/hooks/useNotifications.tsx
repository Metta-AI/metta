import { createContext, useContext, useState, useCallback } from 'react';
import type { ReactNode } from 'react';
import type { NotificationData } from '../types';

interface NotificationContextType {
  notifications: NotificationData[];
  showNotification: (message: string, type: NotificationData['type'], undoCallback?: () => void) => void;
  dismissNotification: (id: string) => void;
}

const NotificationContext = createContext<NotificationContextType | null>(null);

export function NotificationProvider({ children }: { children: ReactNode }) {
  const [notifications, setNotifications] = useState<NotificationData[]>([]);

  const showNotification = useCallback((message: string, type: NotificationData['type'], undoCallback?: () => void) => {
    const id = Math.random().toString(36).substr(2, 9);
    const notification: NotificationData = { id, message, type, undoCallback };

    setNotifications(prev => [...prev, notification]);

    // Auto-dismiss after 5 seconds
    setTimeout(() => {
      setNotifications(prev => prev.filter(n => n.id !== id));
    }, 5000);
  }, []);

  const dismissNotification = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  return (
    <NotificationContext.Provider value={{ notifications, showNotification, dismissNotification }}>
      {children}
    </NotificationContext.Provider>
  );
}

export function useNotifications() {
  const context = useContext(NotificationContext);
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider');
  }
  return context;
}
