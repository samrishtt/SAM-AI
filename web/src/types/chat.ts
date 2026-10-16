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

export interface BackendStatus {
  configured: boolean;
  status: "ready" | "not_configured" | "missing_api_key";
  model: string;
  provider: string;
  providerName: string;
  endpoint?: string;
  requiresPassword: boolean;
  rateLimitPerMinute: number;
  maxOutputTokens?: number;
  notice: string;
  privacyNotice: string;
  setupGuide?: string;
}

