import { Conversation, Message } from "@/types/chat";

const CONVERSATIONS_KEY = "sam_ai_conversations_v1";
const ACTIVE_CONV_KEY = "sam_ai_active_conversation_v1";
const SITE_PASSWORD_KEY = "sam_ai_site_password_v1";

export function isBrowser(): boolean {
  return typeof window !== "undefined";
}

export function getStoredPassword(): string {
  if (!isBrowser()) return "";
  try {
    return localStorage.getItem(SITE_PASSWORD_KEY) || "";
  } catch {
    return "";
  }
}

export function setStoredPassword(password: string): void {
  if (!isBrowser()) return;
  try {
    localStorage.setItem(SITE_PASSWORD_KEY, password);
  } catch {}
}

export function clearStoredPassword(): void {
  if (!isBrowser()) return;
  try {
    localStorage.removeItem(SITE_PASSWORD_KEY);
  } catch {}
}

export function loadConversations(): Conversation[] {
  if (!isBrowser()) return [];
  try {
    const raw = localStorage.getItem(CONVERSATIONS_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed;
    }
    return [];
  } catch (e) {
    console.error("Failed to load conversations from localStorage:", e);
    return [];
  }
}

export function saveConversations(conversations: Conversation[]): void {
  if (!isBrowser()) return;
  try {
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(conversations));
  } catch (e) {
    console.error("Failed to save conversations to localStorage:", e);
  }
}

export function getActiveConversationId(): string | null {
  if (!isBrowser()) return null;
  try {
    return localStorage.getItem(ACTIVE_CONV_KEY);
  } catch {
    return null;
  }
}

export function setActiveConversationId(id: string | null): void {
  if (!isBrowser()) return;
  try {
    if (id) {
      localStorage.setItem(ACTIVE_CONV_KEY, id);
    } else {
      localStorage.removeItem(ACTIVE_CONV_KEY);
    }
  } catch {}
}

export function createNewConversation(firstMessageText?: string): Conversation {
  const id = `conv_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  const title = firstMessageText
    ? firstMessageText.slice(0, 32).trim() + (firstMessageText.length > 32 ? "..." : "")
    : "New Conversation";

  return {
    id,
    title,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    messages: [],
  };
}

export function deleteConversation(id: string): Conversation[] {
  const convs = loadConversations().filter((c) => c.id !== id);
  saveConversations(convs);
  if (getActiveConversationId() === id) {
    const nextId = convs.length > 0 ? convs[0].id : null;
    setActiveConversationId(nextId);
  }
  return convs;
}
