import { useNotifications } from '../hooks/useNotifications';
import './Notifications.css';

export function Notifications() {
  const { notifications, dismissNotification } = useNotifications();

  return (
    <div className="notification-container">
      {notifications.map(notification => (
        <div key={notification.id} className={`notification notification-${notification.type}`}>
          <span>{notification.message}</span>
          {notification.undoCallback && (
            <button
              onClick={() => {
                notification.undoCallback?.();
                dismissNotification(notification.id);
              }}
              className="undo-btn"
            >
              Undo
            </button>
          )}
        </div>
      ))}
    </div>
  );
}
