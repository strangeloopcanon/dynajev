import type { NextConfig } from "next";

const api = process.env.READHEAD_API_ORIGIN ?? "http://127.0.0.1:43124";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/readhead-api/:path*",
        destination: `${api}/:path*`,
      },
    ];
  },
};

export default nextConfig;
