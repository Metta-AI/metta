/**
 * UserHoverCard Component
 * 
 * Displays a hover tooltip with user information when hovering over user avatars.
 * This component is rendered as a portal to appear above other content.
 * 
 * Features:
 * - Portal rendering to avoid layout constraints
 * - User avatar and basic information display
 * - Link to user profile
 * - Positioned relative to mouse cursor
 */

import React from 'react';
import ReactDOM from 'react-dom';

interface UserHoverCardProps {
    user: any;
    position: { x: number; y: number };
}

export function UserHoverCard({ user, position }: UserHoverCardProps) {
    return ReactDOM.createPortal(
        <div
            className="fixed z-50 bg-white border border-gray-200 rounded-lg shadow-lg p-4 min-w-[180px] max-w-xs pointer-events-auto"
            style={{ left: position.x, top: position.y }}
        >
            <div className="flex items-center gap-3 mb-2">
                <div className="w-10 h-10 bg-primary-500 text-white rounded-full flex items-center justify-center text-lg font-semibold">
                    {user.avatar}
                </div>
                <div>
                    <div className="font-semibold text-gray-900 text-base leading-tight">{user.name}</div>
                    {user.email && <div className="text-xs text-gray-500">{user.email}</div>}
                </div>
            </div>
            <a
                href={`/scholars/${user.id}`}
                className="inline-block mt-2 text-xs text-primary-600 hover:underline font-medium"
            >
                View profile
            </a>
        </div>,
        document.body
    );
} 