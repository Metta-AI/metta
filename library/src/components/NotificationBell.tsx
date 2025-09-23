"use client";

import React, { useState, useEffect, useRef } from "react";
import { Bell, X, Check, CheckCheck } from "lucide-react";
import Link from "next/link";

interface Notification {
  id: string;
  type: string;
  title: string;
  message?: string | null;
  actionUrl?: string | null;
  isRead: boolean;
  createdAt: string;
  mentionText?: string | null;
  actor?: {
    id: string;
    name?: string | null;
    email?: string | null;
  } | null;
  post?: {
    id: string;
    title: string;
  } | null;
  comment?: {
    id: string;
    content: string;
    post: {
      id: string;
      title: string;
    };
  } | null;
}

interface NotificationCounts {
  total: number;
  unread: number;
}

interface NotificationBellProps {
  className?: string;
}

export const NotificationBell: React.FC<NotificationBellProps> = ({ 
  className = "" 
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [counts, setCounts] = useState<NotificationCounts>({ total: 0, unread: 0 });
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Load notification counts on mount and periodically
  useEffect(() => {
    loadCounts();
    
    // Poll for new notifications every 30 seconds
    const interval = setInterval(loadCounts, 30000);
    return () => clearInterval(interval);
  }, []);

  // Load full notifications when dropdown opens
  useEffect(() => {
    if (isOpen && !hasLoaded) {
      loadNotifications();
    }
  }, [isOpen, hasLoaded]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const loadCounts = async () => {
    try {
      const response = await fetch("/api/notifications/count");
      if (response.ok) {
        const newCounts = await response.json();
        setCounts(newCounts);
      }
    } catch (error) {
      console.error("Error loading notification counts:", error);
    }
  };

  const loadNotifications = async () => {
    setIsLoading(true);
    try {
      const response = await fetch("/api/notifications?limit=20&includeRead=true");
      if (response.ok) {
        const data = await response.json();
        setNotifications(data.notifications);
        setCounts(data.counts);
        setHasLoaded(true);
      }
    } catch (error) {
      console.error("Error loading notifications:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const markAsRead = async (notificationIds: string[]) => {
    try {
      const response = await fetch("/api/notifications/mark-read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ notificationIds }),
      });

      if (response.ok) {
        // Update local state
        setNotifications(prev =>
          prev.map(n =>
            notificationIds.includes(n.id) ? { ...n, isRead: true } : n
          )
        );
        
        // Update counts
        setCounts(prev => ({
          ...prev,
          unread: Math.max(0, prev.unread - notificationIds.length),
        }));
      }
    } catch (error) {
      console.error("Error marking notifications as read:", error);
    }
  };

  const markAllAsRead = async () => {
    try {
      const response = await fetch("/api/notifications/mark-read", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ markAllRead: true }),
      });

      if (response.ok) {
        // Update local state
        setNotifications(prev =>
          prev.map(n => ({ ...n, isRead: true }))
        );
        setCounts(prev => ({ ...prev, unread: 0 }));
      }
    } catch (error) {
      console.error("Error marking all notifications as read:", error);
    }
  };

  const handleNotificationClick = (notification: Notification) => {
    // Mark as read if unread
    if (!notification.isRead) {
      markAsRead([notification.id]);
    }

    // Close dropdown
    setIsOpen(false);

    // Navigate to action URL if provided (handled by Link component)
  };

  const formatTimeAgo = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diffInSeconds < 60) return "just now";
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h`;
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d`;
    return `${Math.floor(diffInSeconds / 604800)}w`;
  };

  const getNotificationIcon = (type: string) => {
    switch (type) {
      case "MENTION":
        return "💬";
      case "COMMENT":
        return "💬";
      case "REPLY":
        return "↩️";
      case "LIKE":
        return "❤️";
      case "FOLLOW":
        return "👤";
      case "SYSTEM":
        return "🔔";
      default:
        return "📢";
    }
  };

  return (
    <div className={`relative ${className}`}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="relative flex items-center justify-center rounded-full p-2 text-gray-600 hover:bg-gray-100 hover:text-gray-900 transition-colors"
        aria-label="Notifications"
      >
        <Bell className="h-5 w-5" />
        {counts.unread > 0 && (
          <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-xs font-medium text-white">
            {counts.unread > 99 ? "99+" : counts.unread}
          </span>
        )}
      </button>

      {isOpen && (
        <div
          ref={dropdownRef}
          className="absolute right-0 top-full z-50 mt-2 w-96 max-h-96 overflow-hidden rounded-lg bg-white shadow-lg border border-gray-200"
        >
          <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
            <h3 className="font-semibold text-gray-900">Notifications</h3>
            <div className="flex items-center gap-2">
              {counts.unread > 0 && (
                <button
                  onClick={markAllAsRead}
                  className="flex items-center gap-1 rounded px-2 py-1 text-xs text-gray-600 hover:bg-gray-100"
                  title="Mark all as read"
                >
                  <CheckCheck className="h-3 w-3" />
                  Mark all read
                </button>
              )}
              <button
                onClick={() => setIsOpen(false)}
                className="rounded p-1 text-gray-400 hover:bg-gray-100 hover:text-gray-600"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>

          <div className="max-h-80 overflow-y-auto">
            {isLoading ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
              </div>
            ) : notifications.length === 0 ? (
              <div className="px-4 py-8 text-center text-gray-500">
                <Bell className="mx-auto h-8 w-8 text-gray-300 mb-2" />
                <p>No notifications yet</p>
              </div>
            ) : (
              notifications.map((notification) => (
                <NotificationItem
                  key={notification.id}
                  notification={notification}
                  onNotificationClick={handleNotificationClick}
                  formatTimeAgo={formatTimeAgo}
                  getNotificationIcon={getNotificationIcon}
                />
              ))
            )}
          </div>

          {notifications.length > 0 && (
            <div className="border-t border-gray-200 px-4 py-2">
              <Link
                href="/notifications"
                className="block text-center text-sm text-blue-600 hover:text-blue-800"
                onClick={() => setIsOpen(false)}
              >
                View all notifications
              </Link>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

interface NotificationItemProps {
  notification: Notification;
  onNotificationClick: (notification: Notification) => void;
  formatTimeAgo: (dateString: string) => string;
  getNotificationIcon: (type: string) => string;
}

const NotificationItem: React.FC<NotificationItemProps> = ({
  notification,
  onNotificationClick,
  formatTimeAgo,
  getNotificationIcon,
}) => {
  const actorName = notification.actor?.name || 
                   notification.actor?.email?.split("@")[0] || 
                   "Someone";

  const content = (
    <div
      className={`flex items-start gap-3 px-4 py-3 hover:bg-gray-50 cursor-pointer transition-colors ${
        !notification.isRead ? "bg-blue-50" : ""
      }`}
      onClick={() => onNotificationClick(notification)}
    >
      <div className="flex-shrink-0 text-lg">
        {getNotificationIcon(notification.type)}
      </div>
      
      <div className="flex-1 min-w-0">
        <p className={`text-sm ${!notification.isRead ? "font-semibold" : ""} text-gray-900`}>
          {notification.title}
        </p>
        
        {notification.message && (
          <p className="text-xs text-gray-600 mt-1 line-clamp-2">
            {notification.message}
          </p>
        )}
        
        {notification.mentionText && (
          <span className="inline-block mt-1 px-2 py-0.5 bg-blue-100 text-blue-800 text-xs rounded-full">
            {notification.mentionText}
          </span>
        )}
      </div>
      
      <div className="flex flex-col items-end gap-1">
        <span className="text-xs text-gray-500">
          {formatTimeAgo(notification.createdAt)}
        </span>
        {!notification.isRead && (
          <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
        )}
      </div>
    </div>
  );

  if (notification.actionUrl) {
    return (
      <Link href={notification.actionUrl}>
        {content}
      </Link>
    );
  }

  return content;
};
