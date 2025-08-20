# Google OAuth Setup Guide

This guide walks you through setting up Google OAuth authentication for your Library application.

## 1. Create Google OAuth Application

### Step 1: Go to Google Cloud Console

1. Visit the [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account

### Step 2: Create or Select a Project

1. If you don't have a project, click "Create Project"
2. Give it a name like "Library App" or use an existing project
3. Select your project from the dropdown

### Step 3: Enable the Google+ API

1. Go to "APIs & Services" → "Library" from the left sidebar
2. Search for "Google+ API"
3. Click on it and press "Enable"

### Step 4: Configure OAuth Consent Screen

1. Go to "APIs & Services" → "OAuth consent screen"
2. Choose "External" user type (unless you have a Google Workspace)
3. Fill in the required fields:
   - **App name**: Your app name (e.g., "Library App")
   - **User support email**: Your email
   - **Developer contact information**: Your email
4. Click "Save and Continue"
5. Skip the "Scopes" step for now (click "Save and Continue")
6. Add test users if needed, or skip (click "Save and Continue")

### Step 5: Create OAuth Credentials

1. Go to "APIs & Services" → "Credentials"
2. Click "+ Create Credentials" → "OAuth client ID"
3. Choose "Web application"
4. Set the name (e.g., "Library App Web Client")
5. Add authorized origins and redirect URIs:

   **Authorized JavaScript origins:**
   - For development: `http://localhost:3001`
   - For production: `https://your-domain.com`

   **Authorized redirect URIs:**
   - For development: `http://localhost:3001/api/auth/callback/google`
   - For production: `https://your-domain.com/api/auth/callback/google`

6. Click "Create"
7. **IMPORTANT**: Copy the Client ID and Client Secret immediately!

## 2. Update Environment Variables

### Development Setup

1. Open your `.env.local` file
2. Replace the placeholder values:

```bash
# Set to false to enable Google OAuth (true for dev mode with magic links)
DEV_MODE=false

# Replace with your actual Google OAuth credentials
GOOGLE_CLIENT_ID=your_actual_google_client_id_here
GOOGLE_CLIENT_SECRET=your_actual_google_client_secret_here
```

### Production Setup

For production deployment, make sure to:

1. Set `DEV_MODE=false`
2. Add the production callback URL in Google Console: `https://your-domain.com/api/auth/callback/google`
3. Use environment variables or secure secret management for the credentials

## 3. Test the Setup

1. Start your development server:

   ```bash
   pnpm dev
   ```

2. Visit `http://localhost:3001`
3. You should be redirected to the Google OAuth sign-in page
4. Complete the OAuth flow

## 4. Switching Between Dev and Production Auth

### Development Mode (Magic Links)

Set in `.env.local`:

```bash
DEV_MODE=true
```

This uses the fake email provider that logs magic links to the console.

### Production Mode (Google OAuth)

Set in `.env.local`:

```bash
DEV_MODE=false
```

This uses Google OAuth authentication.

## Troubleshooting

### Common Issues

1. **"redirect_uri_mismatch" error**
   - Make sure the callback URL in Google Console exactly matches your application URL
   - Development: `http://localhost:3001/api/auth/callback/google`
   - Check for trailing slashes and protocol (http vs https)

2. **"access_denied" error**
   - Check that your OAuth consent screen is properly configured
   - Make sure you're using the correct Google account
   - Verify the app isn't restricted to specific users only

3. **Environment variables not loading**
   - Restart your development server after changing `.env.local`
   - Make sure there are no extra spaces around the `=` sign
   - Verify the file is named exactly `.env.local`

4. **Console errors about missing scopes**
   - The basic setup only needs profile and email scopes, which are included by default
   - If you need additional scopes, configure them in the OAuth consent screen

### Need Help?

If you run into issues:

1. Check the browser developer console for error messages
2. Check your terminal/server logs for auth-related errors
3. Verify all environment variables are set correctly
4. Make sure your Google Cloud project has the necessary APIs enabled

## Security Notes

- Never commit your `.env.local` file to git (it should be in `.gitignore`)
- Use different OAuth clients for development and production
- Regularly rotate your client secrets
- Consider using OAuth scopes to limit access to only what your app needs
