"use client";

import React, { useState, useEffect, useRef } from "react";
import { Bell, X, CheckCheck } from "lucide-react";
import Link from "next/link";

import { type NotificationDTO } from "@/lib/api/resources/notifications";
import {
  useNotifications,
  useNotificationCounts,
} from "@/hooks/queries/useNotifications";
import { useMarkNotificationsRead } from "@/hooks/mutations/useMarkNotificationsRead";

type Notification = NotificationDTO;

interface NotificationBellProps {
  className?: string;
}

export const NotificationBell: React.FC<NotificationBellProps> = ({
  className = "",
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Query for notification counts with automatic polling
  const { data: counts } = useNotificationCounts();

  // Query for full notifications - only fetch when dropdown is open
  const {
    data: notificationsData,
    isLoading,
    refetch: refetchNotifications,
  } = useNotifications({ limit: 20, includeRead: true }, { enabled: isOpen });

  // Mutation for marking notifications as read
  const markReadMutation = useMarkNotificationsRead();

  const notifications = notificationsData?.notifications ?? [];

  // Refetch notifications when dropdown opens
  useEffect(() => {
    if (isOpen) {
      refetchNotifications();
    }
  }, [isOpen, refetchNotifications]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const markAsRead = async (notificationIds: string[]) => {
    markReadMutation.mutate({ notificationIds });
  };

  const markAllAsRead = async () => {
    markReadMutation.mutate({ markAllRead: true });
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
        className="relative flex items-center justify-center rounded-full p-2 text-gray-600 transition-colors hover:bg-gray-100 hover:text-gray-900"
        aria-label="Notifications"
      >
        <Bell className="h-5 w-5" />
        {counts && counts.unread > 0 && (
          <span className="absolute -top-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full bg-red-500 text-xs font-medium text-white">
            {counts.unread > 99 ? "99+" : counts.unread}
          </span>
        )}
      </button>

      {isOpen && (
        <div
          ref={dropdownRef}
          className="absolute top-full right-0 z-50 mt-2 max-h-96 w-96 overflow-hidden rounded-lg border border-gray-200 bg-white shadow-lg"
        >
          <div className="flex items-center justify-between border-b border-gray-200 px-4 py-3">
            <h3 className="font-semibold text-gray-900">Notifications</h3>
            <div className="flex items-center gap-2">
              {counts && counts.unread > 0 && (
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
                <div className="h-6 w-6 animate-spin rounded-full border-b-2 border-blue-600"></div>
              </div>
            ) : notifications.length === 0 ? (
              <div className="px-4 py-8 text-center text-gray-500">
                <Bell className="mx-auto mb-2 h-8 w-8 text-gray-300" />
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
  const content = (
    <div
      className={`flex cursor-pointer items-start gap-3 px-4 py-3 transition-colors hover:bg-gray-50 ${
        !notification.isRead ? "bg-blue-50" : ""
      }`}
      onClick={() => onNotificationClick(notification)}
    >
      <div className="flex-shrink-0 text-lg">
        {getNotificationIcon(notification.type)}
      </div>

      <div className="min-w-0 flex-1">
        <p
          className={`text-sm ${!notification.isRead ? "font-semibold" : ""} text-gray-900`}
        >
          {notification.title}
        </p>

        {notification.message && (
          <p className="mt-1 line-clamp-2 text-xs text-gray-600">
            {notification.message}
          </p>
        )}

        {notification.mentionText && (
          <span className="mt-1 inline-block rounded-full bg-blue-100 px-2 py-0.5 text-xs text-blue-800">
            {notification.mentionText}
          </span>
        )}
      </div>

      <div className="flex flex-col items-end gap-1">
        <span className="text-xs text-gray-500">
          {formatTimeAgo(notification.createdAt)}
        </span>
        {!notification.isRead && (
          <div className="h-2 w-2 rounded-full bg-blue-600"></div>
        )}
      </div>
    </div>
  );

  if (notification.actionUrl) {
    return <Link href={notification.actionUrl}>{content}</Link>;
  }

  return content;
};
