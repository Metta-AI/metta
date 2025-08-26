# Build stage
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Install js-yaml for nginx config generation
RUN npm install js-yaml

# Copy source code
COPY . .

# Generate nginx.conf from template and links.yaml
RUN node generate-nginx-redirects.js

# Build the application
RUN npm run build

# Runtime stage
FROM nginx:alpine

# Copy built assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy generated nginx configuration from builder stage
COPY --from=builder /app/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port 80
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
