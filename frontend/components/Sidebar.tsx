"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Car, DoorOpen, History, LayoutDashboard, LogIn } from "lucide-react";

const NAV_ITEMS = [
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/entry", label: "Entrada", icon: LogIn },
  { href: "/exit", label: "Saída", icon: DoorOpen },
  { href: "/history", label: "Histórico", icon: History },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-60 shrink-0 border-r border-border bg-panel flex flex-col">
      <div className="flex items-center gap-2 px-5 py-5 border-b border-border">
        <Car className="text-accent" size={24} />
        <div>
          <p className="text-sm font-semibold text-white leading-tight">SmartVision ALPR</p>
          <p className="text-xs text-gray-500 leading-tight">Estacionamento Inteligente</p>
        </div>
      </div>

      <nav className="flex-1 px-3 py-4 space-y-1">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => {
          const active = pathname?.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm transition-colors ${
                active
                  ? "bg-accent/15 text-accent font-medium"
                  : "text-gray-400 hover:bg-surface hover:text-gray-200"
              }`}
            >
              <Icon size={18} />
              {label}
            </Link>
          );
        })}
      </nav>

      <div className="px-5 py-4 border-t border-border text-xs text-gray-600">
        Prova de conceito — sem hardware real
      </div>
    </aside>
  );
}
