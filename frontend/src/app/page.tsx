"use client";

import { FormEvent, useEffect, useRef, useState } from "react";

type Message = {
  role: "user" | "assistant";
  content: string;
  imagePreview?: string; // data URL shown in the chat bubble
};

type IngestedDocument = {
  source_id: string;
  chunks: number;
  last_ingested_at: string;
};

function renderAssistantMessage(content: string) {
  const cleaned = content
    .replace(/\*\*(.*?)\*\*/g, "$1")
    .replace(/\\\[|\\\]/g, "")
    .replace(/\\\(|\\\)/g, "")
    .replace(/\\text\{([^}]*)\}/g, "$1")
    .replace(/`{1,3}/g, "")
    .replace(/\s{3,}/g, "  ")
    .trim();

  const lines = cleaned
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  const blocks: Array<{ type: "paragraph" | "list"; items: string[] }> = [];

  for (const line of lines) {
    const bullet = line.match(/^[-*•]\s+(.+)/);
    const numbered = line.match(/^\d+[.)]\s+(.+)/);
    const listItem = bullet?.[1] ?? numbered?.[1];

    if (listItem) {
      const last = blocks[blocks.length - 1];
      if (last && last.type === "list") {
        last.items.push(listItem);
      } else {
        blocks.push({ type: "list", items: [listItem] });
      }
      continue;
    }

    blocks.push({ type: "paragraph", items: [line] });
  }

  return (
    <div className="space-y-2">
      {blocks.map((block, index) =>
        block.type === "list" ? (
          <ul key={`block-${index}`} className="list-disc space-y-1 pl-5">
            {block.items.map((item, itemIndex) => (
              <li key={`list-${index}-${itemIndex}`}>{item}</li>
            ))}
          </ul>
        ) : (
          <p key={`block-${index}`}>{block.items[0]}</p>
        )
      )}
    </div>
  );
}

const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/** Convert a File to { base64, mediaType } */
async function fileToBase64(file: File): Promise<{ base64: string; mediaType: string; dataUrl: string }> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      // dataUrl: "data:image/png;base64,iVBOR..."
      const [header, base64] = dataUrl.split(",");
      const mediaType = header.replace("data:", "").replace(";base64", "");
      resolve({ base64, mediaType, dataUrl });
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

export default function Home() {
  const [isLightMode, setIsLightMode] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [documents, setDocuments] = useState<IngestedDocument[]>([]);
  const [isSending, setIsSending] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isLoadingDocs, setIsLoadingDocs] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Chat image attachment state
  const [chatImage, setChatImage] = useState<File | null>(null);
  const [chatImagePreview, setChatImagePreview] = useState<string | null>(null);
  const chatImageInputRef = useRef<HTMLInputElement>(null);

  async function loadDocuments() {
    setIsLoadingDocs(true);
    try {
      const response = await fetch(`${apiBase}/api/v1/documents`);
      if (!response.ok) throw new Error("Documents request failed");
      const data = await response.json();
      setDocuments(data);
    } catch {
      setError("Could not load ingested documents.");
    } finally {
      setIsLoadingDocs(false);
    }
  }

  function handleChatImageSelect(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    if (!file) return;
    setChatImage(file);
    const preview = URL.createObjectURL(file);
    setChatImagePreview(preview);
    // Reset input so the same file can be re-selected
    event.target.value = "";
  }

  function removeChatImage() {
    if (chatImagePreview) URL.revokeObjectURL(chatImagePreview);
    setChatImage(null);
    setChatImagePreview(null);
  }

  async function handleSend(event: FormEvent) {
    event.preventDefault();
    if (!input.trim() || isSending) return;

    const userMessage = input.trim();
    setInput("");
    setError(null);
    setIsSending(true);

    // Capture and clear the attached image before the async work
    const imageFile = chatImage;
    const imagePreviewUrl = chatImagePreview ?? undefined;
    removeChatImage();

    setMessages((prev) => [
      ...prev,
      { role: "user", content: userMessage, imagePreview: imagePreviewUrl },
    ]);

    try {
      let imageBase64: string | null = null;
      let imageMediaType: string | null = null;

      if (imageFile) {
        const result = await fileToBase64(imageFile);
        imageBase64 = result.base64;
        imageMediaType = result.mediaType;
      }

      const response = await fetch(`${apiBase}/api/v1/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          conversation_id: conversationId,
          message: userMessage,
          ...(imageBase64 && { image_base64: imageBase64, image_media_type: imageMediaType }),
        }),
      });

      if (!response.ok) throw new Error("Chat request failed");

      const data = await response.json();
      setConversationId(data.conversation_id);
      setMessages((prev) => [...prev, { role: "assistant", content: data.answer }]);
    } catch {
      setError("Could not reach backend. Check FastAPI server and env config.");
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: "I could not connect to the backend. Please verify the API is running.",
        },
      ]);
    } finally {
      setIsSending(false);
    }
  }

  async function handleIngest() {
    if (selectedFiles.length === 0 || isIngesting) return;

    setError(null);
    setIsIngesting(true);
    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => formData.append("files", file));

      const response = await fetch(`${apiBase}/api/v1/ingest/documents`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Ingest request failed");

      setSelectedFiles([]);
      await loadDocuments();
    } catch {
      setError("Document ingestion failed. Check file type, OpenAI key, and database.");
    } finally {
      setIsIngesting(false);
    }
  }

  useEffect(() => {
    const savedTheme = window.localStorage.getItem("awq-theme");
    if (savedTheme === "light") setIsLightMode(true);
    void loadDocuments();
  }, []);

  function toggleTheme() {
    const next = !isLightMode;
    setIsLightMode(next);
    window.localStorage.setItem("awq-theme", next ? "light" : "dark");
  }

  const dark = !isLightMode;

  return (
    <div
      className={`min-h-screen px-3 py-4 md:px-6 md:py-6 ${
        dark ? "bg-[#050505] text-zinc-100" : "bg-zinc-100 text-zinc-900"
      }`}
    >
      <main
        className={`mx-auto flex h-[calc(100vh-2rem)] w-full max-w-[1500px] min-w-0 flex-col overflow-hidden rounded-3xl border shadow-[0_20px_80px_rgba(0,0,0,0.20)] ${
          dark ? "border-zinc-800/70 bg-[#0b0b0b]" : "border-zinc-300 bg-white"
        }`}
      >
        {/* ── Header ── */}
        <header
          className={`flex items-center justify-between border-b px-6 py-4 ${
            dark ? "border-zinc-800/80" : "border-zinc-200"
          }`}
        >
          <div className="flex items-center gap-3">
            <div className="grid h-8 w-8 place-items-center rounded-lg bg-zinc-100 text-xs font-bold text-zinc-900">
              A
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-tight">Awaqi</h1>
              <p className={`text-xs ${dark ? "text-zinc-500" : "text-zinc-500"}`}>
                Multimodal RAG assistant with conversation memory
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div
              className={`rounded-full border px-3 py-1 text-xs ${
                dark
                  ? "border-zinc-700/80 bg-zinc-900/60 text-zinc-400"
                  : "border-zinc-300 bg-zinc-100 text-zinc-600"
              }`}
            >
              RAG • Vision • Context Aware
            </div>
            <button
              onClick={toggleTheme}
              className={`rounded-full border px-3 py-1 text-xs font-medium ${
                dark
                  ? "border-zinc-700 bg-zinc-100 text-zinc-900"
                  : "border-zinc-300 bg-zinc-900 text-zinc-100"
              }`}
            >
              {dark ? "Light" : "Dark"}
            </button>
          </div>
        </header>

        {/* ── Body ── */}
        <div className="grid min-h-0 flex-1 md:grid-cols-[1fr_370px]">

          {/* ── Chat panel ── */}
          <section className="flex min-h-0 flex-col px-6 py-5">
            {/* Messages */}
            <div className="flex-1 space-y-4 overflow-y-auto pb-4 pr-2">
              {messages.length === 0 && (
                <div className="grid h-full place-items-center">
                  <div
                    className={`max-w-md rounded-2xl border p-8 text-center ${
                      dark ? "border-zinc-800/70 bg-zinc-900/40" : "border-zinc-300 bg-zinc-100"
                    }`}
                  >
                    <div className="mb-3 text-3xl">✦</div>
                    <p className="text-3xl font-semibold tracking-tight">Awaqi</p>
                    <p className="mt-2 text-sm text-zinc-500">
                      Ingest documents or images, then ask questions — or attach an image directly to your message.
                    </p>
                  </div>
                </div>
              )}
              {messages.map((message, index) => (
                <div
                  key={`${message.role}-${index}`}
                  className={`w-fit max-w-[78%] rounded-2xl border border-zinc-200 bg-zinc-100 px-4 py-3 text-sm leading-relaxed text-zinc-950 shadow-sm ${
                    message.role === "user" ? "ml-auto" : "mr-auto"
                  }`}
                >
                  {/* Inline image preview inside user bubble */}
                  {message.imagePreview && (
                    <img
                      src={message.imagePreview}
                      alt="attached"
                      className="mb-2 max-h-48 w-auto rounded-lg object-contain"
                    />
                  )}
                  {message.role === "assistant"
                    ? renderAssistantMessage(message.content)
                    : message.content}
                </div>
              ))}
            </div>

            {/* ── Input area ── */}
            <div className="pb-1">
              {/* Image preview strip above input */}
              {chatImagePreview && (
                <div className="mb-2 flex items-center gap-2">
                  <div className="relative inline-block">
                    <img
                      src={chatImagePreview}
                      alt="preview"
                      className="h-16 w-auto rounded-lg border border-zinc-300 object-contain"
                    />
                    <button
                      onClick={removeChatImage}
                      className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-zinc-800 text-[10px] text-white hover:bg-red-500"
                      title="Remove image"
                    >
                      ✕
                    </button>
                  </div>
                  <span className="text-xs text-zinc-400 truncate max-w-[200px]">
                    {chatImage?.name}
                  </span>
                </div>
              )}

              <form
                className={`flex w-full items-center gap-2 rounded-2xl border px-3 py-2 shadow-[0_8px_30px_rgba(0,0,0,0.10)] ${
                  dark
                    ? "border-zinc-700/90 bg-zinc-900/80"
                    : "border-zinc-300 bg-zinc-50"
                }`}
                onSubmit={handleSend}
              >
                {/* Hidden image file input */}
                <input
                  ref={chatImageInputRef}
                  type="file"
                  accept="image/png,image/jpeg,image/webp,image/gif"
                  className="hidden"
                  onChange={handleChatImageSelect}
                />

                {/* Paperclip button */}
                <button
                  type="button"
                  title="Attach an image"
                  onClick={() => chatImageInputRef.current?.click()}
                  className={`flex-shrink-0 rounded-lg p-1.5 text-sm transition-colors ${
                    chatImage
                      ? "bg-zinc-200 text-zinc-900"
                      : dark
                      ? "text-zinc-500 hover:text-zinc-300"
                      : "text-zinc-400 hover:text-zinc-700"
                  }`}
                >
                  📎
                </button>

                <span className="text-zinc-500">⌁</span>

                <input
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  placeholder={chatImage ? "Ask about this image…" : "Type your question here..."}
                  className="flex-1 bg-transparent px-1 py-2 text-sm outline-none placeholder:text-zinc-500"
                />
                <button
                  type="submit"
                  disabled={isSending}
                  className="rounded-full bg-zinc-100 px-4 py-1.5 text-xs font-semibold text-zinc-950 disabled:opacity-50"
                >
                  {isSending ? "Sending..." : "Ask"}
                </button>
              </form>
            </div>
          </section>

          {/* ── Knowledge base sidebar ── */}
          <section
            className={`hidden min-h-0 border-l p-4 md:flex md:flex-col ${
              dark ? "border-zinc-800/80 bg-[#0d0d0d]" : "border-zinc-200 bg-zinc-50"
            }`}
          >
            <h2 className={`text-sm font-semibold ${dark ? "text-zinc-200" : "text-zinc-900"}`}>
              Knowledge Base
            </h2>
            <p className="mt-1 text-xs text-zinc-500 leading-relaxed">
              Upload documents or images to your local vector DB for RAG retrieval.
            </p>

            <label
              className={`mt-3 rounded-xl border border-dashed p-3 text-xs text-zinc-400 ${
                dark ? "border-zinc-700 bg-zinc-900/70" : "border-zinc-300 bg-white"
              }`}
            >
              <span className={`mb-2 block ${dark ? "text-zinc-300" : "text-zinc-700"}`}>
                Supported: .txt, .md, .pdf, .docx, .png, .jpg, .webp, .gif
              </span>
              <input
                type="file"
                multiple
                accept=".txt,.md,.pdf,.docx,.png,.jpg,.jpeg,.webp,.gif"
                onChange={(event) => {
                  const files = Array.from(event.target.files ?? []);
                  setSelectedFiles(files);
                }}
                className={`block w-full cursor-pointer text-xs text-zinc-400 file:mr-3 file:rounded-md file:border-0 file:px-2 file:py-1 ${
                  dark
                    ? "file:bg-zinc-800 file:text-zinc-200"
                    : "file:bg-zinc-200 file:text-zinc-800"
                }`}
              />
            </label>

            <button
              onClick={handleIngest}
              disabled={isIngesting}
              className="mt-2 rounded-xl bg-zinc-100 px-4 py-2 text-sm font-semibold text-zinc-950 disabled:opacity-50"
            >
              {isIngesting ? "Ingesting…" : "Ingest Documents"}
            </button>

            <div
              className={`mt-2 rounded-lg border p-2 text-xs text-zinc-500 ${
                dark ? "border-zinc-700/80 bg-zinc-900/60" : "border-zinc-300 bg-white"
              }`}
            >
              {selectedFiles.length === 0
                ? "No documents selected"
                : `${selectedFiles.length} file(s) selected`}
            </div>

            {/* Ingested documents list */}
            <div
              className={`mt-4 min-h-0 flex-1 overflow-hidden rounded-xl border ${
                dark ? "border-zinc-700/80 bg-zinc-900/50" : "border-zinc-300 bg-white"
              }`}
            >
              <div
                className={`border-b px-3 py-2 text-xs font-semibold uppercase tracking-wide ${
                  dark ? "border-zinc-700/60 text-zinc-400" : "border-zinc-200 text-zinc-600"
                }`}
              >
                Ingested Documents
              </div>
              <div className="h-full overflow-y-auto px-3 py-2">
                {isLoadingDocs && <p className="text-xs text-zinc-500">Loading documents…</p>}
                {!isLoadingDocs && documents.length === 0 && (
                  <p className="text-xs text-zinc-500">No documents ingested yet.</p>
                )}
                <div className="space-y-2">
                  {documents.map((doc) => {
                    const isImage = /\.(png|jpe?g|webp|gif)$/i.test(doc.source_id);
                    return (
                      <div
                        key={`${doc.source_id}-${doc.last_ingested_at}`}
                        className={`rounded-lg border px-2.5 py-2 ${
                          dark
                            ? "border-zinc-700/60 bg-zinc-900"
                            : "border-zinc-300 bg-zinc-50"
                        }`}
                      >
                        <p
                          className={`truncate text-xs font-medium ${
                            dark ? "text-zinc-200" : "text-zinc-800"
                          }`}
                        >
                          {isImage ? "🖼 " : "📄 "}
                          {doc.source_id}
                        </p>
                        <p className="mt-1 text-[11px] text-zinc-400">
                          {doc.chunks} chunks · {new Date(doc.last_ingested_at).toLocaleString()}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>
        </div>

        {error && <p className="px-6 pb-3 text-sm text-red-400">{error}</p>}
      </main>
    </div>
  );
}
