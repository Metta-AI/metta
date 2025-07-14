import { useState, useEffect } from 'react';
import { mockUsers } from './mockData/users';

interface UserProfileProps {
    repo: unknown;
    currentUser: string;
}

export function UserProfile({ repo: _repo, currentUser }: UserProfileProps) {
    const [user, setUser] = useState<any>(null);

    useEffect(() => {
        // Find the user by email (currentUser is the email)
        const foundUser = mockUsers.find(u => u.email === currentUser);
        if (foundUser) {
            setUser(foundUser);
        } else {
            // If user not found in mock data, create a basic user object
            setUser({
                id: 'current-user',
                name: currentUser.split('@')[0], // Use email prefix as name
                avatar: currentUser.charAt(0).toUpperCase(),
                email: currentUser
            });
        }
    }, [currentUser]);

    if (!user) {
        return (
            <div className="min-h-screen bg-gray-50 flex items-center justify-center">
                <div className="text-center">
                    <h1 className="text-2xl font-bold text-gray-900 mb-2">Loading profile...</h1>
                    <p className="text-gray-600">Loading user information.</p>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white border-b border-gray-200">
                <div className="max-w-4xl mx-auto px-4 py-6">
                    {/* User Info */}
                    <div className="flex items-start gap-6">
                        <div className="w-24 h-24 bg-primary-500 text-white rounded-full flex items-center justify-center text-3xl font-semibold flex-shrink-0">
                            {user.avatar}
                        </div>
                        <div className="flex-1 min-w-0">
                            <h2 className="text-3xl font-bold text-gray-900 mb-2">{user.name}</h2>
                            <p className="text-xl text-gray-600 mb-3">{user.email}</p>
                            
                            <div className="flex items-center gap-4 mb-4">
                                <span className="px-3 py-1 rounded-full text-sm font-semibold bg-blue-100 text-blue-700 border border-blue-200">
                                    Active User
                                </span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Content */}
            <div className="max-w-4xl mx-auto px-4 py-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Account Information */}
                    <div className="bg-white rounded-lg border border-gray-200 p-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Information</h3>
                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">User ID</label>
                                <p className="text-sm text-gray-900 bg-gray-50 px-3 py-2 rounded border">{user.id}</p>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Display Name</label>
                                <p className="text-sm text-gray-900 bg-gray-50 px-3 py-2 rounded border">{user.name}</p>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
                                <p className="text-sm text-gray-900 bg-gray-50 px-3 py-2 rounded border">{user.email}</p>
                            </div>
                        </div>
                    </div>

                    {/* Account Status */}
                    <div className="bg-white rounded-lg border border-gray-200 p-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Account Status</h3>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-700">Account Status</span>
                                <span className="px-2 py-1 rounded-full text-xs font-semibold bg-green-100 text-green-700">
                                    Active
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-700">Member Since</span>
                                <span className="text-sm text-gray-900">2024</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-700">Last Login</span>
                                <span className="text-sm text-gray-900">Today</span>
                            </div>
                        </div>
                    </div>

                    {/* Preferences */}
                    <div className="bg-white rounded-lg border border-gray-200 p-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Preferences</h3>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-700">Email Notifications</span>
                                <span className="px-2 py-1 rounded-full text-xs font-semibold bg-blue-100 text-blue-700">
                                    Enabled
                                </span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-700">Theme</span>
                                <span className="text-sm text-gray-900">Light</span>
                            </div>
                            <div className="flex items-center justify-between">
                                <span className="text-sm font-medium text-gray-700">Language</span>
                                <span className="text-sm text-gray-900">English</span>
                            </div>
                        </div>
                    </div>

                    {/* Quick Actions */}
                    <div className="bg-white rounded-lg border border-gray-200 p-6">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
                        <div className="space-y-3">
                            <button className="w-full text-left px-4 py-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors">
                                <div className="flex items-center gap-3">
                                    <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                    </svg>
                                    <div>
                                        <p className="font-medium text-gray-900">Change Password</p>
                                        <p className="text-sm text-gray-500">Update your account password</p>
                                    </div>
                                </div>
                            </button>
                            <button className="w-full text-left px-4 py-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors">
                                <div className="flex items-center gap-3">
                                    <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                    </svg>
                                    <div>
                                        <p className="font-medium text-gray-900">Account Settings</p>
                                        <p className="text-sm text-gray-500">Manage your account preferences</p>
                                    </div>
                                </div>
                            </button>
                            <button className="w-full text-left px-4 py-3 rounded-lg border border-gray-200 hover:bg-gray-50 transition-colors">
                                <div className="flex items-center gap-3">
                                    <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    <div>
                                        <p className="font-medium text-gray-900">Help & Support</p>
                                        <p className="text-sm text-gray-500">Get help with your account</p>
                                    </div>
                                </div>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
} 