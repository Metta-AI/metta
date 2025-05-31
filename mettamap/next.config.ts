import type { NextConfig } from 'next';

const nextConfig: NextConfig = {
  outputFileTracingIncludes: {
    "/": ["./public/maps/**/*"],
    "/maps/[name]": ["./public/maps/**/*"],
    "**": ["./public/maps/**/*"],
  },
};

export default nextConfig;
