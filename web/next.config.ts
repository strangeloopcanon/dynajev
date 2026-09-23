import type { NextConfig } from "next";

const api = process.env.DYNAJEV_API_ORIGIN ?? "http://127.0.0.1:43124";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/dynajev-api/:path*",
        destination: `${api}/:path*`,
      },
    ];
  },
};

export default nextConfig;
