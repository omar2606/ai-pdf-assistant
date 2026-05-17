from __future__ import annotations

import io
import os
from dataclasses import dataclass

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader


load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MAX_CONTEXT_CHARS = 60_000


@dataclass(frozen=True)
class ParsedPdf:
    name: str
    pages: int
    text: str


def extract_pdf_text(uploaded_file) -> ParsedPdf:
    data = uploaded_file.getvalue()
    reader = PdfReader(io.BytesIO(data))

    page_texts: list[str] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            page_texts.append(f"[{uploaded_file.name}, page {page_number}]\n{text.strip()}")

    return ParsedPdf(
        name=uploaded_file.name,
        pages=len(reader.pages),
        text="\n\n".join(page_texts).strip(),
    )


def build_context(parsed_pdfs: list[ParsedPdf]) -> str:
    context = "\n\n---\n\n".join(pdf.text for pdf in parsed_pdfs if pdf.text)
    if len(context) <= MAX_CONTEXT_CHARS:
        return context

    return (
        context[:MAX_CONTEXT_CHARS]
        + "\n\n[Context truncated because the uploaded PDFs are long. Ask about a specific section for best results.]"
    )


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is missing. Add it to the .env file, then restart Streamlit.")
        st.stop()
    return OpenAI(api_key=api_key)


def ask_ai(client: OpenAI, model: str, pdf_context: str, messages: list[dict[str, str]]) -> str:
    conversation = "\n".join(
        f"{message['role'].title()}: {message['content']}" for message in messages[-10:]
    )

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a helpful PDF assistant. Answer using the provided PDF text. "
                    "If the PDF text does not contain the answer, say that clearly."
                ),
            },
            {
                "role": "user",
                "content": (
                    "PDF text:\n"
                    f"{pdf_context}\n\n"
                    "Conversation so far:\n"
                    f"{conversation}\n\n"
                    "Reply to the latest user message."
                ),
            },
        ],
    )

    return response.output_text.strip()


st.set_page_config(page_title="AI PDF Assistant", page_icon="PDF", layout="wide")

st.title("AI PDF Assistant")
st.caption("Upload a PDF, then ask questions about its contents.")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("OpenAI model", value=DEFAULT_MODEL)

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.parsed_pdfs = []
        st.session_state.uploaded_signature = None
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "parsed_pdfs" not in st.session_state:
    st.session_state.parsed_pdfs = []
if "uploaded_signature" not in st.session_state:
    st.session_state.uploaded_signature = None

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

signature = tuple((file.name, file.size) for file in uploaded_files or [])
if uploaded_files and signature != st.session_state.uploaded_signature:
    parsed_pdfs: list[ParsedPdf] = []
    with st.spinner("Reading PDF text..."):
        for uploaded_file in uploaded_files:
            try:
                parsed_pdf = extract_pdf_text(uploaded_file)
                parsed_pdfs.append(parsed_pdf)
            except Exception as exc:
                st.error(f"Could not read {uploaded_file.name}: {exc}")

    st.session_state.parsed_pdfs = parsed_pdfs
    st.session_state.uploaded_signature = signature
    st.session_state.messages = []

if st.session_state.parsed_pdfs:
    total_pages = sum(pdf.pages for pdf in st.session_state.parsed_pdfs)
    total_chars = sum(len(pdf.text) for pdf in st.session_state.parsed_pdfs)
    st.success(f"Loaded {len(st.session_state.parsed_pdfs)} PDF(s), {total_pages} page(s), {total_chars:,} text characters.")

    empty_text_files = [pdf.name for pdf in st.session_state.parsed_pdfs if not pdf.text]
    if empty_text_files:
        st.warning(
            "Some PDFs had no extractable text: "
            + ", ".join(empty_text_files)
            + ". Scanned image PDFs need OCR before this app can read them."
        )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask about the uploaded PDF...")
if prompt:
    if not st.session_state.parsed_pdfs:
        st.warning("Upload a PDF first, then ask your question.")
        st.stop()

    pdf_context = build_context(st.session_state.parsed_pdfs)
    if not pdf_context:
        st.error("The uploaded PDF did not contain extractable text.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_ai(get_client(), model.strip() or DEFAULT_MODEL, pdf_context, st.session_state.messages)
            except Exception as exc:
                answer = f"OpenAI request failed: {exc}"
                st.error(answer)
            else:
                st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
