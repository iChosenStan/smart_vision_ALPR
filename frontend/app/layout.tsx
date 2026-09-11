import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "SmartVision ALPR — Estacionamento Inteligente",
  description: "PoC de estacionamento inteligente com reconhecimento automático de placas (ALPR)",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="pt-BR" className="dark">
      <body className="flex min-h-screen bg-[#0b0d13] text-[#e6e8ef] antialiased">
        <Sidebar />
        <main className="flex-1 p-8 overflow-y-auto">{children}</main>
      </body>
    </html>
  );
}
