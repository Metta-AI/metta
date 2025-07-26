# This Dockerfile relies on turborepo to optimize the build process.
# Refer to https://turborepo.com/docs/guides/tools/docker for more details.

# Building this image requires the entire metta repo to be used as a build context, due to the global `pnpm-lock.yaml` file.

# ================================ Base image - pnpm and turbo ================================
FROM node:24-bookworm-slim AS base

# Set working directory
WORKDIR /metta

ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"

# Install dependencies
COPY package.json ./
RUN corepack enable pnpm
RUN pnpm install -g turbo

# ================================ Prepare pruned source with turbo ================================
FROM base AS pruned-source
WORKDIR /metta
COPY . .

# /metta/out will contain only the relevant files
RUN turbo prune @softmax/observatory --docker

# =============================== Build from pruned source ================================
FROM base AS builder
WORKDIR /metta

COPY --from=pruned-source /metta/out/json/ .
RUN pnpm install --frozen-lockfile

COPY --from=pruned-source /metta/out/full/ .

WORKDIR /metta/observatory

RUN pnpm run build

# ================================ Serve the static app with Nginx ================================
FROM nginx:alpine

# Copy the build output from the previous stage to Nginx's html directory
COPY --from=builder /metta/observatory/dist /usr/share/nginx/html

# Copy a custom Nginx configuration file (optional but recommended)
# For example, handle single-page app routing
COPY observatory/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

# Start Nginx in the foreground
CMD ["nginx", "-g", "daemon off;"]
