// Tipos espelhando exatamente os schemas Pydantic do backend
// (backend/app/schemas/*.py) — mantidos manualmente em sincronia.

export type TicketStatus = "EM_ABERTO" | "PAGO" | "FINALIZADO";

export interface VehicleRead {
  id: number;
  plate: string;
  make: string;
  model: string;
  color: string;
  type: string;
}

export interface TicketRead {
  id: number;
  vehicle: VehicleRead;
  entry_image_path: string | null;
  plate_image_path: string | null;
  entry_at: string;
  exit_at: string | null;
  ocr_confidence: number;
  status: TicketStatus;
  amount: number | null;
  payment_method: string | null;
  paid_at: string | null;
  ticket_number: string;
  duration_minutes: number;
}

export interface TicketListResponse {
  items: TicketRead[];
  total: number;
  page: number;
  page_size: number;
}

export interface DashboardResponse {
  total_vehicles: number;
  vehicles_parked: number;
  tickets_paid: number;
  tickets_pending: number;
  simulated_revenue: number;
  available_spots: number;
  total_spots: number;
}

export interface ExitResponse {
  gate_open: boolean;
  message: string;
  ticket: TicketRead | null;
}

export interface TicketListFilters {
  plate?: string;
  status?: TicketStatus;
  page?: number;
  page_size?: number;
}
