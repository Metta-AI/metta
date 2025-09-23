"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import { ArrowLeft, Bell, CheckCheck, Trash2 } from "lucide-react";

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

export default function NotificationsPage() {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [counts, setCounts] = useState<NotificationCounts>({ total: 0, unread: 0 });
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "unread">("all");

  useEffect(() => {
    loadNotifications();
  }, [filter]);

  const loadNotifications = async () => {
    setIsLoading(true);
    try {
      const includeRead = filter === "all";
      const response = await fetch(`/api/notifications?limit=50&includeRead=${includeRead}`);
      if (response.ok) {
        const data = await response.json();
        setNotifications(data.notifications);
        setCounts(data.counts);
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
        setNotifications(prev =>
          prev.map(n =>
            notificationIds.includes(n.id) ? { ...n, isRead: true } : n
          )
        );
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
        setNotifications(prev =>
          prev.map(n => ({ ...n, isRead: true }))
        );
        setCounts(prev => ({ ...prev, unread: 0 }));
      }
    } catch (error) {
      console.error("Error marking all notifications as read:", error);
    }
  };

  const formatTimeAgo = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diffInSeconds < 60) return "just now";
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)} minutes ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)} hours ago`;
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)} days ago`;
    return date.toLocaleDateString();
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

  const handleNotificationClick = (notification: Notification) => {
    if (!notification.isRead) {
      markAsRead([notification.id]);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="mx-auto max-w-4xl px-4 py-8">
        <div className="mb-6 flex items-center gap-4">
          <Link 
            href="/"
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900"
          >
            <ArrowLeft className="h-5 w-5" />
            Back to Feed
          </Link>
        </div>

        <div className="bg-white rounded-lg shadow-sm">
          <div className="border-b border-gray-200 px-6 py-4">
            <div className="flex items-center justify-between">
              <h1 className="text-2xl font-bold text-gray-900 flex items-center gap-3">
                <Bell className="h-6 w-6" />
                Notifications
              </h1>
              
              <div className="flex items-center gap-4">
                <div className="flex rounded-lg border border-gray-200 p-1">
                  <button
                    onClick={() => setFilter("all")}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                      filter === "all"
                        ? "bg-blue-100 text-blue-700"
                        : "text-gray-500 hover:text-gray-700"
                    }`}
                  >
                    All ({counts.total})
                  </button>
                  <button
                    onClick={() => setFilter("unread")}
                    className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                      filter === "unread"
                        ? "bg-blue-100 text-blue-700"
                        : "text-gray-500 hover:text-gray-700"
                    }`}
                  >
                    Unread ({counts.unread})
                  </button>
                </div>

                {counts.unread > 0 && (
                  <button
                    onClick={markAllAsRead}
                    className="flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium text-blue-600 hover:bg-blue-50"
                  >
                    <CheckCheck className="h-4 w-4" />
                    Mark all read
                  </button>
                )}
              </div>
            </div>
          </div>

          <div className="divide-y divide-gray-200">
            {isLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
              </div>
            ) : notifications.length === 0 ? (
              <div className="px-6 py-12 text-center">
                <Bell className="mx-auto h-12 w-12 text-gray-300 mb-4" />
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  {filter === "unread" ? "No unread notifications" : "No notifications yet"}
                </h3>
                <p className="text-gray-500">
                  {filter === "unread" 
                    ? "You're all caught up! Check back later for new notifications."
                    : "When someone mentions you or interacts with your content, you'll see it here."
                  }
                </p>
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
        </div>
      </div>
    </div>
  );
}

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
      className={`px-6 py-4 hover:bg-gray-50 transition-colors ${
        !notification.isRead ? "bg-blue-50" : ""
      }`}
    >
      <div className="flex items-start gap-4">
        <div className="flex-shrink-0">
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-gray-100">
            <span className="text-lg">{getNotificationIcon(notification.type)}</span>
          </div>
        </div>
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <p className={`text-sm ${!notification.isRead ? "font-semibold" : "font-medium"} text-gray-900`}>
              {notification.title}
            </p>
            {!notification.isRead && (
              <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
            )}
          </div>
          
          {notification.message && (
            <p className="text-sm text-gray-600 mb-2">
              {notification.message}
            </p>
          )}
          
          {notification.mentionText && (
            <div className="mb-2">
              <span className="inline-block px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">
                {notification.mentionText}
              </span>
            </div>
          )}
          
          <p className="text-xs text-gray-500">
            {formatTimeAgo(notification.createdAt)}
          </p>
        </div>
      </div>
    </div>
  );

  const handleClick = () => {
    onNotificationClick(notification);
  };

  if (notification.actionUrl) {
    return (
      <Link href={notification.actionUrl} onClick={handleClick}>
        {content}
      </Link>
    );
  }

  return (
    <div onClick={handleClick} className="cursor-pointer">
      {content}
    </div>
  );
};
