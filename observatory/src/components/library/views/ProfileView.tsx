
import { UserProfile } from '../../../UserProfile';

/**
 * ProfileView Component
 * 
 * This component handles the user profile view of the Library app, providing:
 * - A wrapper around the UserProfile component with consistent layout
 * - Proper prop passing from the parent Library component
 * - Responsive design that matches the overall app structure
 * 
 * The ProfileView acts as a container that ensures the UserProfile component
 * is displayed with the correct styling and layout within the Library app's
 * navigation structure.
 * 
 * Features:
 * - Consistent padding and max-width constraints
 * - Responsive design that works across different screen sizes
 * - Clean integration with the Library app's navigation system
 * 
 * Usage Example:
 * ```tsx
 * <ProfileView
 *   repo={repo}
 *   currentUser="user@example.com"
 * />
 * ```
 */

interface ProfileViewProps {
    /** Repository object passed from the parent Library component */
    repo: unknown;
    
    /** Current user's email address for profile display */
    currentUser: string;
}

/**
 * ProfileView Component
 * 
 * Renders the user profile view with proper layout and styling.
 * This component serves as a wrapper around the UserProfile component,
 * ensuring it integrates seamlessly with the Library app's design system.
 */
export function ProfileView({ repo, currentUser }: ProfileViewProps) {
    return (
        <div className="p-6">
            <div className="max-w-6xl mx-auto">
                <UserProfile 
                    repo={repo} 
                    currentUser={currentUser || "alice@example.com"} 
                />
            </div>
        </div>
    );
} 