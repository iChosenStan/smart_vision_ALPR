import type {
  DashboardResponse,
  ExitResponse,
  TicketListFilters,
  TicketListResponse,
  TicketRead,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api";

/** Erro tipado da API — carrega o `detail` retornado pelo FastAPI (HTTPException). */
export class ApiError extends Error {
  status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail ?? detail;
    } catch {
      // corpo não era JSON — mantém o statusText
    }
    throw new ApiError(response.status, detail);
  }

  return response.json() as Promise<T>;
}

/** Constrói a URL completa de uma imagem capturada (servida pelo backend em /captures). */
export function captureImageUrl(relativePath: string | null): string | null {
  if (!relativePath) return null;
  const base = API_BASE_URL.replace(/\/api$/, "");
  return `${base}/captures/${relativePath}`;
}

export const api = {
  getDashboard: () => request<DashboardResponse>("/dashboard"),

  registerEntry: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<TicketRead>("/entry", { method: "POST", body: formData });
  },

  registerExit: (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<ExitResponse>("/exit", { method: "POST", body: formData });
  },

  getTicket: (id: number) => request<TicketRead>(`/tickets/${id}`),

  payTicket: (id: number, paymentMethod: string) =>
    request<TicketRead>(`/tickets/${id}/pay`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ payment_method: paymentMethod }),
    }),

  listTickets: (filters: TicketListFilters = {}) => {
    const params = new URLSearchParams();
    if (filters.plate) params.set("plate", filters.plate);
    if (filters.status) params.set("status", filters.status);
    params.set("page", String(filters.page ?? 1));
    params.set("page_size", String(filters.page_size ?? 20));
    return request<TicketListResponse>(`/tickets?${params.toString()}`);
  },

  getHistory: (filters: TicketListFilters = {}) => {
    const params = new URLSearchParams();
    if (filters.plate) params.set("plate", filters.plate);
    if (filters.status) params.set("status", filters.status);
    params.set("page", String(filters.page ?? 1));
    params.set("page_size", String(filters.page_size ?? 20));
    return request<TicketListResponse>(`/history?${params.toString()}`);
  },
};
