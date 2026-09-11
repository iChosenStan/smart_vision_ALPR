/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // As imagens capturadas (veículo/placa) são servidas pelo backend
  // FastAPI em /captures — ver backend/app/main.py (StaticFiles).
  images: {
    remotePatterns: [
      {
        protocol: "http",
        hostname: "localhost",
        port: "8000",
        pathname: "/captures/**",
      },
    ],
  },
};

export default nextConfig;
