import React from "react";
import { ChatInterface } from "@/components/ChatInterface";
import { Navbar } from "@/components/Navbar";
import { Metadata } from "next";

export const metadata: Metadata = {
  title: "SAM-AI — Chat & Code Demo",
  description: "Interactive conversational intelligence and sovereign code workspace demonstration.",
};

export default function ChatPage() {
  return (
    <div className="min-h-screen bg-[#FAF8F5] text-[#28231F] flex flex-col font-sans">
      <Navbar />
      <main className="flex-1 flex flex-col w-full h-[calc(100vh-4rem)]">
        <ChatInterface />
      </main>
    </div>
  );
}
