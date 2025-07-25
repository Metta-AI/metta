# Stage 1: Build the application
FROM node:18 AS builder

# Set working directory
WORKDIR /app

# Copy package.json and package-lock.json for dependency installation
COPY package*.json ./

# Install dependencies
RUN corepack enable
RUN pnpm install

# Copy the rest of the application code
COPY . .

# Build the application for production
RUN pnpm run build

# Stage 2: Serve the app with Nginx
FROM nginx:alpine

# Copy the build output from the previous stage to Nginx's html directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy a custom Nginx configuration file (optional but recommended)
# For example, handle single-page app routing
COPY nginx.conf /etc/nginx/conf.d/default.conf

# COPY docker-fix-env.sh /docker-entrypoint.d/docker-fix-env.sh
# RUN chmod +x /docker-entrypoint.d/docker-fix-env.sh

# Expose port 80
EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
