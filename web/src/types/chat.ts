export interface Message {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt?: Date | number;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: Message[];
}

export type BackendStatusCode =
  | "ready"
  | "not_configured"
  | "unreachable"
  | "loading"
  | "weights_missing"
  | "model_mismatch"
  | "prohibited_provider";

export interface BackendStatus {
  configured: boolean;
  status: BackendStatusCode;
  model: string;
  adapter?: string;
  provider: string;
  providerName: string;
  endpoint?: string;
  requiresPassword: boolean;
  rateLimitPerMinute: number;
  maxOutputTokens?: number;
  notice: string;
  privacyNotice: string;
  setupGuide?: string;
  backendError?: string;
  isLoaded?: boolean;
}

export interface WorkspaceFileInfo {
  path: string;
  name: string;
  size_bytes: number;
  extension: string;
}

export interface WorkspaceStatus {
  configured: boolean;
  status: "ready" | "workspace_not_configured" | "error";
  workspace_name?: string;
  file_count?: number;
  read_only: boolean;
  error?: string;
}
