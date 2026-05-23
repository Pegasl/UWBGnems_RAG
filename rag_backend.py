from __future__ import annotations

import argparse
import base64
import datetime as dt
import hmac
from http.cookies import SimpleCookie
import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
from pathlib import Path
import secrets
import socket
import sys
import threading
import time
import traceback
from types import SimpleNamespace
from typing import Any, Generator, Iterable, List
from urllib.parse import unquote, urlparse
import uuid

ROOT_DIR = Path(__file__).resolve().parent


def load_project_env() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    candidates: list[Path] = []
    env_from_var = os.getenv("RAG_ENV_FILE")
    if env_from_var:
        candidates.append(Path(env_from_var).expanduser())
    candidates.append(ROOT_DIR / ".env")

    for env_file in candidates:
        if env_file.exists():
            load_dotenv(env_file)
            return


load_project_env()


MODEL_NAME = os.getenv("RAG_LLM_MODEL")
QUESTION_REWRITE_MODEL = os.getenv("RAG_QUESTION_REWRITE_MODEL") or MODEL_NAME or ""
QUESTION_REWRITE_TIMEOUT_SECONDS = float(os.getenv("RAG_QUESTION_REWRITE_TIMEOUT_SECONDS", "30"))

# ================= 路径配置 =================
JSON_DIR = Path(os.getenv("RAG_JSON_DIR")).expanduser()
IMAGE_DIR = Path(os.getenv("RAG_IMAGE_DIR")).expanduser()
CHROMA_PATH = Path(os.getenv("RAG_CHROMA_PATH")).expanduser()
HISTORY_PATH = Path(os.getenv("RAG_HISTORY_PATH")).expanduser()
INDEX_STATE_PATH = Path(
    os.getenv("RAG_INDEX_STATE_PATH", str(ROOT_DIR / "rag_index_state.json"))
).expanduser()

EMBEDDING_MODEL_PATH = os.getenv("RAG_EMBEDDING_MODEL_PATH")
RERANKER_MODEL_PATH = os.getenv("RAG_RERANKER_MODEL_PATH")
EMBEDDING_DEVICE = os.getenv("RAG_EMBEDDING_DEVICE", "cuda:0")
RERANKER_DEVICE = os.getenv("RAG_RERANKER_DEVICE", "cuda:1")
WEB_HOST = os.getenv("RAG_WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.getenv("RAG_WEB_PORT", "7860"))
WEB_API_KEY = os.getenv("RAG_WEB_API_KEY", os.getenv("RAG_WEB_ACCESS_KEY", "")).strip()
WEB_AUTH_COOKIE = "rag_session"
WEB_SESSION_TTL_SECONDS = int(float(os.getenv("RAG_WEB_SESSION_HOURS", "24")) * 3600)
MAX_CONTEXT_MESSAGES = int(os.getenv("RAG_MAX_CONTEXT_MESSAGES", "8"))


class MissingRAGDependency(RuntimeError):
    pass


def import_rag_dependencies() -> SimpleNamespace:
    try:
        import chromadb
        import numpy as np
        import torch
        from docuement_parser import chunk_document, clean_and_extract_doc
        from openai import OpenAI
        from script.qwen3_vl_embedding import Qwen3VLEmbedder
        from script.qwen3_vl_reranker import Qwen3VLReranker
    except Exception as exc:
        raise MissingRAGDependency(str(exc)) from exc

    return SimpleNamespace(
        chromadb=chromadb,
        np=np,
        torch=torch,
        clean_and_extract_doc=clean_and_extract_doc,
        chunk_document=chunk_document,
        OpenAI=OpenAI,
        Qwen3VLEmbedder=Qwen3VLEmbedder,
        Qwen3VLReranker=Qwen3VLReranker,
    )


def build_openai_client(OpenAIClass: Any | None = None) -> Any:
    if OpenAIClass is None:
        try:
            from openai import OpenAI as OpenAIClass
        except Exception as exc:
            raise RuntimeError(f"缺少 OpenAI SDK，无法调用 LLM：{exc}") from exc

    api_key = os.getenv("OPENAI_API_KEY", os.getenv("API_KEY"))
    base_url = os.getenv("OPENAI_BASE_URL", os.getenv("API_URL"))
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY 或 API_KEY，无法调用 LLM。")
    return OpenAIClass(api_key=api_key, base_url=base_url)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def compact_text(value: str, limit: int = 56) -> str:
    value = " ".join((value or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def recent_history_for_rewrite(history: Iterable[dict[str, Any]] | None) -> str:
    lines: list[str] = []
    for message in list(history or [])[-MAX_CONTEXT_MESSAGES:]:
        role = message.get("role")
        content = compact_text(str(message.get("content", "")), 900)
        if role in {"user", "assistant"} and content:
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")
    return "\n".join(lines)


def parse_rewrite_response(content: str, fallback: str) -> tuple[str, str]:
    raw = str(content or "").strip()
    if not raw:
        return fallback, ""

    json_start = raw.find("{")
    json_end = raw.rfind("}")
    if json_start != -1 and json_end != -1 and json_end > json_start:
        try:
            data = json.loads(raw[json_start : json_end + 1])
            rewritten = str(data.get("rewritten_question") or data.get("question") or "").strip()
            rationale = str(data.get("rationale") or "").strip()
            if rewritten:
                return rewritten, rationale
        except Exception:
            pass

    return raw.strip('"').strip(), ""


def rewrite_question_for_retrieval(
    question: str,
    history: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    question = question.strip()
    if not question:
        raise ValueError("问题不能为空。")
    if not QUESTION_REWRITE_MODEL:
        raise RuntimeError("未设置 RAG_LLM_MODEL 或 RAG_QUESTION_REWRITE_MODEL，无法改写问题。")

    client = build_openai_client()
    history_text = recent_history_for_rewrite(history)
    user_prompt = (
        f"原始问题：\n{question}\n\n"
        f"最近对话历史：\n{history_text or '无'}\n\n"
        "请输出 JSON。"
    )
    response = client.chat.completions.create(
        model=QUESTION_REWRITE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是论文 RAG 系统的检索问题改写器。你的任务是把用户问题改写成更适合向量检索"
                    "和重排的查询，不回答问题，不编造论文中未出现的限定条件。保留用户原意，补全省略"
                    "指代，加入必要的中英文术语、缩写、公式名、器件名或方法名；如果原问题已经清晰，"
                    "可以只做轻微规范化。只返回 JSON，格式为 "
                    "{\"rewritten_question\":\"...\", \"rationale\":\"...\"}。"
                ),
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        timeout=QUESTION_REWRITE_TIMEOUT_SECONDS,
    )
    content = response.choices[0].message.content if response.choices else ""
    rewritten, rationale = parse_rewrite_response(content, question)
    rewritten = " ".join(rewritten.split()) or question
    return {
        "question": question,
        "rewritten_question": rewritten,
        "changed": rewritten != question,
        "rationale": rationale,
    }


def is_lan_candidate(ip: str) -> bool:
    try:
        address = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return address.version == 4 and not address.is_loopback and not address.is_unspecified


def discover_lan_addresses() -> list[str]:
    addresses: set[str] = set()

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET, socket.SOCK_DGRAM):
            ip = info[4][0]
            if is_lan_candidate(ip):
                addresses.add(ip)
    except OSError:
        pass

    for target in [("8.8.8.8", 80), ("1.1.1.1", 80), ("192.168.1.1", 80)]:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.connect(target)
            ip = sock.getsockname()[0]
            if is_lan_candidate(ip):
                addresses.add(ip)
        except OSError:
            pass
        finally:
            if sock is not None:
                sock.close()

    return sorted(addresses)


def local_image_to_data_url(img_path: str) -> str:
    path = Path(img_path).expanduser()
    if not path.exists() or not path.is_file():
        return ""
    mime, _ = mimetypes.guess_type(path.name)
    if mime is None:
        mime = "image/png"
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def source_label(source: str) -> str:
    return (source or "Unknown").replace("_content_list.json", "").replace(".json", "")


def compact_sources(txt_results: list[dict[str, Any]], img_results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "texts": [
            {
                "source": source_label(str(item.get("source", "Unknown"))),
                "preview": compact_text(str(item.get("text", "")), 220),
            }
            for item in txt_results[:8]
        ],
        "images": [
            {
                "source": source_label(str(item.get("source", "Unknown"))),
                "page_idx": item.get("page_idx", "Unknown"),
                "caption": compact_text(str(item.get("caption", "")), 180),
            }
            for item in img_results[:8]
        ],
    }


class RAGPipeline:
    def __init__(self, embedding_model: Any, reranker_model: Any, chroma_path: str, deps: SimpleNamespace):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.batch_size = 8
        self.deps = deps
        self.chroma_client = deps.chromadb.PersistentClient(path=chroma_path)
        self._attach_collections()

        self.client = build_openai_client(deps.OpenAI)

    def _attach_collections(self) -> None:
        self.txt_collection = self.chroma_client.get_or_create_collection(
            name="text_collection",
            metadata={"hnsw:space": "cosine"},
        )
        self.img_collection = self.chroma_client.get_or_create_collection(
            name="image_collection",
            metadata={"hnsw:space": "cosine"},
        )

    def reset_vector_store(self) -> None:
        for name in ("text_collection", "image_collection"):
            try:
                self.chroma_client.delete_collection(name=name)
            except Exception:
                pass
        self._attach_collections()

    def delete_source_vectors(self, source_name: str) -> None:
        source_name = str(source_name or "").strip()
        if not source_name:
            return
        try:
            self.txt_collection.delete(where={"source": source_name})
        except Exception:
            pass
        try:
            self.img_collection.delete(where={"source": source_name})
        except Exception:
            pass

    def process_and_store_document(self, json_path: Path, *, replace_existing: bool = False) -> None:
        print(f"正在处理文档: {json_path.name}")
        source_name = str(json_path.name)
        if replace_existing:
            self.delete_source_vectors(source_name)

        full_text, images = self.deps.clean_and_extract_doc(str(json_path))
        document_chunks = self.deps.chunk_document(full_text, chunk_size=800, chunk_overlap=150)

        txt_inputs = [{"text": doc} for doc in document_chunks]
        img_inputs = [
            {"image": img.get("img_path", "").strip(), "text": img.get("caption", "").strip()}
            for img in images
        ]

        txt_vectors = []
        for start in range(0, len(txt_inputs), self.batch_size):
            batch = txt_inputs[start : start + self.batch_size]
            tensors = self.embedding_model.process(batch)
            txt_vectors.extend(tensors.cpu().float().tolist())

        img_vectors = []
        for start in range(0, len(img_inputs), 1):
            batch = img_inputs[start : start + 1]
            try:
                tensors = self.embedding_model.process(batch)
                img_vectors.extend(tensors.cpu().float().tolist())
            except Exception as exc:
                img_path_failed = batch[0].get("image", "未知路径") if batch else "未知"
                print(f"警告：图片处理失败并跳过。图片路径: {img_path_failed} | 错误信息: {exc}")
                img_vectors.append(None)

        valid_img_vectors = []
        valid_images = []
        for index, vec in enumerate(img_vectors):
            if vec is None:
                continue
            img_meta = images[index].copy()
            img_meta["source"] = source_name
            valid_img_vectors.append(vec)
            valid_images.append(img_meta)

        if txt_vectors:
            self.txt_collection.upsert(
                embeddings=txt_vectors,
                documents=document_chunks,
              metadatas=[{"source": source_name} for _ in range(len(document_chunks))],
              ids=[f"txt_{source_name}_{index}" for index in range(len(document_chunks))],
            )

        if valid_img_vectors:
            self.img_collection.upsert(
                embeddings=valid_img_vectors,
                metadatas=valid_images,
            ids=[f"img_{source_name}_{index}" for index in range(len(valid_images))],
            )
        print(f"{json_path.name} 的向量已成功存入 ChromaDB")

    def query(self, question: str, k: int = 100) -> tuple[List[dict], List[dict]]:
        question_vector = self.embedding_model.process([{"text": question}]).cpu().float().tolist()[0]

        txt_results = self.txt_collection.query(
            query_embeddings=[question_vector],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        img_results = self.img_collection.query(
            query_embeddings=[question_vector],
            n_results=50,
            include=["metadatas", "distances"],
        )

        top_k_txt_docs = txt_results["documents"][0] if txt_results["documents"] else []
        top_k_txt_metas = txt_results["metadatas"][0] if txt_results["metadatas"] else []
        top_k_img_metas = img_results["metadatas"][0] if img_results["metadatas"] else []

        txt_sorted_results = []
        if top_k_txt_docs:
            rerank_txt_inputs = [{"text": doc} for doc in top_k_txt_docs]
            txt_rerank_scores = self.reranker_model.process(
                {
                    "query": {"text": question},
                    "documents": rerank_txt_inputs,
                }
            )
            sorted_indices = self.deps.np.argsort(txt_rerank_scores)[::-1][:75]
            txt_sorted_results = [
                {
                    "text": top_k_txt_docs[index],
                    "source": top_k_txt_metas[index].get("source", "Unknown"),
                }
                for index in sorted_indices
            ]

        img_sorted_results = []
        if top_k_img_metas:
            rerank_img_inputs = [
                {"image": meta.get("img_path", ""), "text": meta.get("caption", "")}
                for meta in top_k_img_metas
            ]
            img_rerank_scores = self.reranker_model.process(
                {
                    "query": {"text": question},
                    "documents": rerank_img_inputs,
                }
            )
            sorted_indices = self.deps.np.argsort(img_rerank_scores)[::-1][:25]
            img_sorted_results = [
                {
                    "img_path": top_k_img_metas[index].get("img_path", ""),
                    "caption": top_k_img_metas[index].get("caption", ""),
                    "source": top_k_img_metas[index].get("source", "Unknown"),
                    "page_idx": top_k_img_metas[index].get("page_idx", "Unknown"),
                }
                for index in sorted_indices
            ]

        return txt_sorted_results, img_sorted_results

    def build_llm_messages(
        self,
        question: str,
        txt_results: List[dict],
        img_results: List[dict],
        history: Iterable[dict[str, Any]] | None = None,
        retrieval_question: str | None = None,
    ) -> list[dict[str, Any]]:
        context_text = ""
        for res in txt_results:
            clean_source = source_label(str(res.get("source", "Unknown")))
            context_text += f"[content source: {clean_source}]\n{res.get('text', '')}\n\n"

        img_content_blocks: list[dict[str, Any]] = []
        for index, image in enumerate(img_results):
            img_path = str(image.get("img_path", ""))
            data_url = local_image_to_data_url(img_path)
            clean_source = source_label(str(image.get("source", "Unknown")))
            page_idx = image.get("page_idx", "Unknown")

            img_content_blocks.append(
                {
                    "type": "text",
                    "text": (
                        f"Image {index + 1} [image source: {clean_source}, page {page_idx}] "
                        f"caption: {image.get('caption', '')}"
                    ),
                }
            )
            if data_url:
                img_content_blocks.append({"type": "image_url", "image_url": {"url": data_url}})

        system_prompt = (
            "You are a helpful assistant. Please answer the user's question based on the provided "
            "text context, images, and the recent conversation history. IMPORTANT: When you use "
            "information from the text context or images, you MUST explicitly cite the source "
            "document name provided as [content source: <Document Name>] or "
            "[image source: <Document Name>, page x] in your answer. Format the answer in Markdown "
            "when it improves readability, and use LaTeX for mathematical expressions."
        )

        messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        prior_messages = list(history or [])[-MAX_CONTEXT_MESSAGES:]
        for message in prior_messages:
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role in {"user", "assistant"} and content:
                messages.append({"role": role, "content": content[:8000]})

        retrieval_question = (retrieval_question or question).strip()
        question_block = f"Question: {question}"
        if retrieval_question and retrieval_question != question:
            question_block += (
                "\nRetrieval-optimized question used to select the context "
                f"(do not answer this instead of the user question): {retrieval_question}"
            )

        user_content = [{"type": "text", "text": f"{question_block}\n\nContext:\n{context_text}"}]
        user_content.extend(img_content_blocks)
        messages.append({"role": "user", "content": user_content})
        return messages

    def stream_llm_answer(
        self,
        question: str,
        txt_results: List[dict],
        img_results: List[dict],
        history: Iterable[dict[str, Any]] | None = None,
        retrieval_question: str | None = None,
    ) -> Generator[str, None, None]:
        messages = self.build_llm_messages(
            question,
            txt_results,
            img_results,
            history=history,
            retrieval_question=retrieval_question,
        )
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            stream=True,
        )

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content

    def ask_llm(
        self,
        question: str,
        txt_results: List[dict],
        img_results: List[dict],
        history: Iterable[dict[str, Any]] | None = None,
        retrieval_question: str | None = None,
    ) -> str:
        print("\n\n")

        def spinner_task(stop_event: threading.Event) -> None:
            spinner_chars = ["-", "\\", "|", "/"]
            index = 0
            while not stop_event.is_set():
                sys.stdout.write(f"\rthinking {spinner_chars[index % len(spinner_chars)]}")
                sys.stdout.flush()
                time.sleep(0.1)
                index += 1
            sys.stdout.write("\r" + " " * 30 + "\r")
            sys.stdout.flush()

        stop_event = threading.Event()
        spinner_thread = threading.Thread(target=spinner_task, args=(stop_event,), daemon=True)
        spinner_thread.start()
        content_started = False
        answer_parts: list[str] = []

        try:
            for content in self.stream_llm_answer(
                question,
                txt_results,
                img_results,
                history=history,
                retrieval_question=retrieval_question,
            ):
                if not content_started:
                    stop_event.set()
                    spinner_thread.join()
                    print("\nAgent: ", end="", flush=True)
                    content_started = True
                answer_parts.append(content)
                print(content, end="", flush=True)
        except Exception as exc:
            stop_event.set()
            spinner_thread.join()
            print(f"\nAI 回答中断: {exc}")
        finally:
            stop_event.set()
            if spinner_thread.is_alive():
                spinner_thread.join()

        print("\n=========================================\n")
        return "".join(answer_parts)


class DemoRAGPipeline:
    def __init__(self, reason: str = ""):
        self.reason = reason

    def reset_vector_store(self) -> None:
        print("Demo 模式不会清空向量库。")

    def delete_source_vectors(self, source_name: str) -> None:
        print(f"Demo 模式不会删除向量：{source_name}")

    def process_and_store_document(self, json_path: Path, *, replace_existing: bool = False) -> None:
        print(f"Demo 模式不会构建向量库：{json_path}")

    def query(self, question: str, k: int = 100) -> tuple[List[dict], List[dict]]:
        text = (
            "Demo context for the local web UI. It proves Markdown tables, code blocks, "
            "citations, and LaTeX rendering can all be displayed before the full RAG "
            "runtime is installed."
        )
        return [{"text": text, "source": "demo_context.json"}], []

    def stream_llm_answer(
        self,
        question: str,
        txt_results: List[dict],
        img_results: List[dict],
        history: Iterable[dict[str, Any]] | None = None,
        retrieval_question: str | None = None,
    ) -> Generator[str, None, None]:
        reason = self.reason or "已启用 Demo 模式。"
        answer = f"""### Demo 回答

你问的是：

> {question}

当前网页服务已经启动，历史对话、流式输出、Markdown 和 LaTeX 渲染都可以直接测试。实际 RAG 模型未加载的原因是：

`{reason}`

| 能力 | 状态 |
| --- | --- |
| Markdown | 可渲染 |
| LaTeX | 可渲染 |
| 历史对话 | 已持久化 |

把依赖和模型路径配置好之后，关闭 Demo 模式即可接入真实检索与 LLM。引用示例：[content source: demo_context]
"""
        for index in range(0, len(answer), 18):
            time.sleep(0.035)
            yield answer[index : index + 18]

    def ask_llm(
        self,
        question: str,
        txt_results: List[dict],
        img_results: List[dict],
        history: Iterable[dict[str, Any]] | None = None,
        retrieval_question: str | None = None,
    ) -> str:
        answer = "".join(
            self.stream_llm_answer(
                question,
                txt_results,
                img_results,
                history=history,
                retrieval_question=retrieval_question,
            )
        )
        print(answer)
        return answer


class PipelineManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._pipeline: RAGPipeline | DemoRAGPipeline | None = None

    def get(self) -> RAGPipeline | DemoRAGPipeline:
        with self._lock:
            if self._pipeline is not None:
                return self._pipeline

            if os.getenv("RAG_DEMO", "").lower() in {"1", "true", "yes"}:
                self._pipeline = DemoRAGPipeline("RAG_DEMO=1")
                return self._pipeline

            try:
                deps = import_rag_dependencies()
            except MissingRAGDependency as exc:
                self._pipeline = DemoRAGPipeline(f"缺少 RAG 运行依赖或本地模块：{exc}")
                return self._pipeline

            try:
                print("正在加载 Embedding 和 Reranker 模型...")
                embedder_model = deps.Qwen3VLEmbedder(
                    model_name_or_path=EMBEDDING_MODEL_PATH,
                    max_length=8192,
                    min_pixels=4096,
                    max_pixels=1843200,
                    torch_dtype=deps.torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_name=EMBEDDING_DEVICE,
                )
                reranker_model = deps.Qwen3VLReranker(
                    model_name_or_path=RERANKER_MODEL_PATH,
                    torch_dtype=deps.torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_name=RERANKER_DEVICE,
                )
                self._pipeline = RAGPipeline(
                    embedding_model=embedder_model,
                    reranker_model=reranker_model,
                    chroma_path=str(CHROMA_PATH),
                    deps=deps,
                )
                print("模型加载完成！")
                return self._pipeline
            except Exception as exc:
                if os.getenv("RAG_FALLBACK_DEMO", "").lower() in {"1", "true", "yes"}:
                    self._pipeline = DemoRAGPipeline(f"真实 RAG 初始化失败：{exc}")
                    return self._pipeline
                raise


PIPELINE_MANAGER = PipelineManager()


class ConversationStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"conversations": []}
        try:
            with self.path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except Exception:
            backup = self.path.with_suffix(self.path.suffix + f".broken-{int(time.time())}")
            self.path.replace(backup)
            return {"conversations": []}
        if not isinstance(data, dict) or not isinstance(data.get("conversations"), list):
            return {"conversations": []}
        return data

    def _write_unlocked(self, data: dict[str, Any]) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
        tmp_path.replace(self.path)

    def _find(self, data: dict[str, Any], conversation_id: str) -> dict[str, Any] | None:
        for conversation in data["conversations"]:
            if conversation.get("id") == conversation_id:
                return conversation
        return None

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._read_unlocked()
            conversations = sorted(
                data["conversations"],
                key=lambda item: item.get("updated_at", ""),
                reverse=True,
            )
            return [self._summary(conversation) for conversation in conversations]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._read_unlocked()
            conversation = self._find(data, conversation_id)
            if conversation is None:
                return None
            return self._public(conversation)

    def create_conversation(self, title: str = "New Query") -> dict[str, Any]:
        with self._lock:
            data = self._read_unlocked()
            now = utc_now()
            conversation = {
                "id": uuid.uuid4().hex,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
            data["conversations"].append(conversation)
            self._write_unlocked(data)
            return self._public(conversation)

    def append_message(
        self,
        conversation_id: str | None,
        role: str,
        content: str,
        sources: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            data = self._read_unlocked()
            conversation = self._find(data, conversation_id or "")
            if conversation is None:
                now = utc_now()
                conversation = {
                    "id": uuid.uuid4().hex,
                    "title": "New Query",
                    "created_at": now,
                    "updated_at": now,
                    "messages": [],
                }
                data["conversations"].append(conversation)

            now = utc_now()
            message: dict[str, Any] = {
                "id": uuid.uuid4().hex,
                "role": role,
                "content": content,
                "created_at": now,
            }
            if sources:
                message["sources"] = sources
            if error:
                message["error"] = error
            conversation["messages"].append(message)
            conversation["updated_at"] = now

            if role == "user" and conversation.get("title") in {"New Query", "新对话", ""}:
                conversation["title"] = compact_text(content, 34) or "New Query"

            self._write_unlocked(data)
            return self._public(conversation)

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock:
            data = self._read_unlocked()
            before = len(data["conversations"])
            data["conversations"] = [
                conversation
                for conversation in data["conversations"]
                if conversation.get("id") != conversation_id
            ]
            changed = len(data["conversations"]) != before
            if changed:
                self._write_unlocked(data)
            return changed

    def delete_all(self) -> None:
        with self._lock:
            self._write_unlocked({"conversations": []})

    def _summary(self, conversation: dict[str, Any]) -> dict[str, Any]:
        messages = conversation.get("messages", [])
        preview = ""
        if messages:
            preview = compact_text(str(messages[-1].get("content", "")), 72)
        return {
            "id": conversation.get("id"),
            "title": conversation.get("title") or "New Query",
            "created_at": conversation.get("created_at"),
            "updated_at": conversation.get("updated_at"),
            "message_count": len(messages),
            "preview": preview,
        }

    def _public(self, conversation: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": conversation.get("id"),
            "title": conversation.get("title") or "New Query",
            "created_at": conversation.get("created_at"),
            "updated_at": conversation.get("updated_at"),
            "messages": conversation.get("messages", []),
        }


class AuthSessionStore:
    def __init__(self, ttl_seconds: int):
        self.ttl_seconds = max(300, ttl_seconds)
        self._sessions: dict[str, float] = {}
        self._lock = threading.RLock()

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = time.time() + self.ttl_seconds
        with self._lock:
            self._sessions[token] = expires_at
            self._cleanup_unlocked()
        return token

    def validate(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            expires_at = self._sessions.get(token)
            if expires_at is None:
                return False
            if expires_at < time.time():
                self._sessions.pop(token, None)
                return False
            return True

    def revoke(self, token: str | None) -> None:
        if not token:
            return
        with self._lock:
            self._sessions.pop(token, None)

    def _cleanup_unlocked(self) -> None:
        now = time.time()
        expired = [token for token, expires_at in self._sessions.items() if expires_at < now]
        for token in expired:
            self._sessions.pop(token, None)


def file_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class IndexStateStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _default(self) -> dict[str, Any]:
        return {"version": 1, "updated_at": None, "documents": {}}

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                return self._default()
            try:
                with self.path.open("r", encoding="utf-8") as file:
                    data = json.load(file)
            except Exception:
                backup = self.path.with_suffix(self.path.suffix + f".broken-{int(time.time())}")
                try:
                    self.path.replace(backup)
                except Exception:
                    pass
                return self._default()

            if not isinstance(data, dict):
                return self._default()
            documents = data.get("documents", {})
            if not isinstance(documents, dict):
                documents = {}
            return {
                "version": 1,
                "updated_at": data.get("updated_at"),
                "documents": documents,
            }

    def save(self, documents: dict[str, Any]) -> None:
        with self._lock:
            payload = {
                "version": 1,
                "updated_at": utc_now(),
                "documents": documents,
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
            with tmp_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            tmp_path.replace(self.path)



FRONTEND_DIR = ROOT_DIR / "frontend"

def load_frontend_page(name: str) -> str:
    page_path = FRONTEND_DIR / name
    if not page_path.exists():
        raise RuntimeError(f"页面文件不存在：{page_path}")
    return page_path.read_text(encoding="utf-8")


class RAGWebHandler(BaseHTTPRequestHandler):
    store: ConversationStore
    pipeline_manager: PipelineManager
    auth_sessions: AuthSessionStore
    index_state_store: IndexStateStore = IndexStateStore(INDEX_STATE_PATH)
    index_lock = threading.Lock()
    server_version = "InteractiveRAG/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - - [%s] %s\n" % (self.address_string(), self.log_date_time_string(), format % args))

    def _send_bytes(
        self,
        body: bytes,
        status: int = 200,
        content_type: str = "application/octet-stream",
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for header, value in (extra_headers or {}).items():
            self.send_header(header, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(
        self,
        data: Any,
        status: int = 200,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self._send_bytes(
            body,
            status=status,
            content_type="application/json; charset=utf-8",
            extra_headers=extra_headers,
        )

    def _send_error_json(self, message: str, status: int = 400) -> None:
        self._send_json({"error": message}, status=status)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _auth_required(self) -> bool:
        return bool(WEB_API_KEY)

    def _cookie_token(self) -> str | None:
        raw_cookie = self.headers.get("Cookie", "")
        if not raw_cookie:
            return None
        cookie = SimpleCookie()
        try:
            cookie.load(raw_cookie)
        except Exception:
            return None
        morsel = cookie.get(WEB_AUTH_COOKIE)
        return morsel.value if morsel else None

    def _is_authenticated(self) -> bool:
        if not self._auth_required():
            return True
        return self.auth_sessions.validate(self._cookie_token())

    def _send_unauthorized(self) -> None:
        self._send_json({"error": "请先验证 API Key。"}, status=401)

    def _cookie_header(self, token: str, max_age: int) -> str:
        return f"{WEB_AUTH_COOKIE}={token}; Path=/; HttpOnly; SameSite=Lax; Max-Age={max_age}"

    def _send_stream_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in {"/", "/index.html", "/login"}:
            page = load_frontend_page("index.html") if self._is_authenticated() else load_frontend_page("login.html")
            self._send_bytes(page.encode("utf-8"), content_type="text/html; charset=utf-8")
            return

        if path == "/api/auth":
            self._send_json(
                {
                    "required": self._auth_required(),
                    "authenticated": self._is_authenticated(),
                    "session_hours": round(WEB_SESSION_TTL_SECONDS / 3600, 2),
                }
            )
            return

        if path == "/api/conversations":
            if not self._is_authenticated():
                self._send_unauthorized()
                return
            self._send_json(self.store.list_conversations())
            return

        if path.startswith("/api/conversations/"):
            if not self._is_authenticated():
                self._send_unauthorized()
                return
            conversation_id = path.rsplit("/", 1)[-1]
            conversation = self.store.get_conversation(conversation_id)
            if conversation is None:
                self._send_error_json("对话不存在。", status=404)
                return
            self._send_json(conversation)
            return

        if path == "/health":
            self._send_json({"ok": True})
            return

        self._send_error_json("Not found", status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/login":
            self._handle_login()
            return

        if path == "/api/logout":
            token = self._cookie_token()
            self.auth_sessions.revoke(token)
            self._send_json(
                {"ok": True},
                extra_headers={
                    "Set-Cookie": self._cookie_header("", 0),
                },
            )
            return

        if not self._is_authenticated():
            self._send_unauthorized()
            return

        if path == "/api/conversations":
            payload = self._read_json()
            title = str(payload.get("title") or "New Query")
            self._send_json(self.store.create_conversation(title=title), status=201)
            return

        if path == "/api/rewrite-question":
            self._handle_rewrite_question()
            return

        if path == "/api/chat":
            self._handle_chat_stream()
            return

        if path == "/api/index":
            self._handle_index_stream(mode="update")
            return

        if path == "/api/index/rebuild":
            self._handle_index_stream(mode="rebuild")
            return

        if path == "/api/index/update":
            self._handle_index_stream(mode="update")
            return

        self._send_error_json("Not found", status=404)

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if not self._is_authenticated():
            self._send_unauthorized()
            return

        if path == "/api/conversations":
            self.store.delete_all()
            self._send_json({"ok": True})
            return

        if path.startswith("/api/conversations/"):
            conversation_id = path.rsplit("/", 1)[-1]
            deleted = self.store.delete_conversation(conversation_id)
            self._send_json({"ok": deleted})
            return

        self._send_error_json("Not found", status=404)

    def _write_stream_event(self, event: dict[str, Any]) -> None:
        line = json.dumps(event, ensure_ascii=False) + "\n"
        self.wfile.write(line.encode("utf-8"))
        self.wfile.flush()

    def _handle_login(self) -> None:
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_error_json(f"请求 JSON 无效：{exc}", status=400)
            return

        if not self._auth_required():
            self._send_json({"ok": True, "auth_enabled": False})
            return

        api_key = str(payload.get("api_key", ""))
        if hmac.compare_digest(api_key, WEB_API_KEY):
            token = self.auth_sessions.create()
            self._send_json(
                {"ok": True, "auth_enabled": True},
                extra_headers={
                    "Set-Cookie": self._cookie_header(token, WEB_SESSION_TTL_SECONDS),
                },
            )
            return

        time.sleep(0.25)
        self._send_json({"error": "API Key 不正确。"}, status=401)

    def _handle_rewrite_question(self) -> None:
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_error_json(f"请求 JSON 无效：{exc}", status=400)
            return

        question = str(payload.get("question", "")).strip()
        conversation_id = payload.get("conversation_id") or None
        if not question:
            self._send_error_json("问题不能为空。", status=400)
            return

        history: list[dict[str, Any]] = []
        if conversation_id:
            conversation = self.store.get_conversation(str(conversation_id))
            if conversation:
                history = list(conversation.get("messages", []))

        try:
            self._send_json(rewrite_question_for_retrieval(question, history=history))
        except Exception as exc:
            self._send_json(
                {
                    "question": question,
                    "rewritten_question": question,
                    "changed": False,
                    "rationale": "",
                    "warning": f"问题改写失败，已保留原问题：{exc}",
                }
            )

    def _handle_chat_stream(self) -> None:
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_error_json(f"请求 JSON 无效：{exc}", status=400)
            return

        question = str(payload.get("question", "")).strip()
        retrieval_question = str(payload.get("retrieval_question") or question).strip()
        conversation_id = payload.get("conversation_id") or None
        if not question:
            self._send_error_json("问题不能为空。", status=400)
            return
        if not retrieval_question:
            retrieval_question = question

        self._send_stream_headers()

        conversation = self.store.append_message(conversation_id, "user", question)
        prior_messages = conversation.get("messages", [])[:-1]
        self._write_stream_event({"type": "conversation", "conversation": conversation})

        try:
            self._write_stream_event({"type": "status", "message": "正在加载检索管线..."})
            pipeline = self.pipeline_manager.get()

            self._write_stream_event({"type": "status", "message": "正在用确认后的问题检索相关文本和图片..."})
            txt_results, img_results = pipeline.query(question=retrieval_question, k=200)

            self._write_stream_event(
                {
                    "type": "status",
                    "message": f"找到 {len(txt_results)} 条文本和 {len(img_results)} 张图片，正在组织回答...",
                }
            )

            answer_parts: list[str] = []
            for content in pipeline.stream_llm_answer(
                question,
                txt_results,
                img_results,
                history=prior_messages,
                retrieval_question=retrieval_question,
            ):
                answer_parts.append(content)
                self._write_stream_event({"type": "chunk", "content": content})

            answer = "".join(answer_parts).strip() or "（模型没有返回内容。）"
            conversation = self.store.append_message(
                conversation["id"],
                "assistant",
                answer,
                sources=compact_sources(txt_results, img_results),
            )
            self._write_stream_event({"type": "done", "conversation": conversation})
        except BrokenPipeError:
            raise
        except Exception as exc:
            message = f"{exc}"
            traceback.print_exc()
            conversation = self.store.append_message(
                conversation["id"],
                "assistant",
                f"**出错了。**\n\n{message}",
                error=message,
            )
            try:
                self._write_stream_event({"type": "error", "message": message, "conversation": conversation})
            except BrokenPipeError:
                pass

    def _handle_index_stream(self, mode: str = "update") -> None:
        mode = "rebuild" if mode == "rebuild" else "update"
        mode_label = "重构" if mode == "rebuild" else "更新"
        self._send_stream_headers()

        if not self.index_lock.acquire(blocking=False):
            self._write_stream_event(
                {
                    "type": "error",
                    "message": "已有向量库任务正在运行。",
                    "mode": mode,
                }
            )
            return

        processed = 0
        try:
            self._write_stream_event(
                {
                    "type": "status",
                    "message": "正在加载检索管线...",
                    "mode": mode,
                }
            )
            pipeline = self.pipeline_manager.get()

            json_dir = JSON_DIR
            if not json_dir.exists():
                self._write_stream_event(
                    {
                        "type": "error",
                        "message": f"JSON 文件夹不存在：{json_dir}",
                        "mode": mode,
                    }
                )
                return

            json_files = sorted(json_dir.glob("*.json"))
            if not json_files:
                self._write_stream_event(
                    {
                        "type": "error",
                        "message": f"未在 {json_dir} 找到 JSON 文件。",
                        "mode": mode,
                    }
                )
                return

            signatures: dict[str, dict[str, int]] = {}
            for json_file in json_files:
                try:
                    signatures[json_file.name] = file_signature(json_file)
                except OSError:
                    self._write_stream_event(
                        {
                            "type": "status",
                            "message": f"跳过无法访问的文件：{json_file.name}",
                            "mode": mode,
                        }
                    )

            state = self.index_state_store.load()
            state_documents = state.get("documents", {})
            if not isinstance(state_documents, dict):
                state_documents = {}
            state_documents = {
                str(name): metadata
                for name, metadata in state_documents.items()
                if isinstance(metadata, dict)
            }

            removed_sources = [name for name in list(state_documents.keys()) if name not in signatures]

            if mode == "rebuild":
                self._write_stream_event(
                    {
                        "type": "status",
                        "message": "正在清空现有向量库...",
                        "mode": mode,
                    }
                )
                pipeline.reset_vector_store()
                state_documents = {}
                files_to_process = [json_file for json_file in json_files if json_file.name in signatures]
                start_message = f"找到 {len(files_to_process)} 个 JSON 文件，开始重构向量库。"
            else:
                if removed_sources:
                    self._write_stream_event(
                        {
                            "type": "status",
                            "message": f"检测到 {len(removed_sources)} 个已删除文档，正在清理旧向量...",
                            "mode": mode,
                        }
                    )
                    for source_name in removed_sources:
                        pipeline.delete_source_vectors(source_name)
                        state_documents.pop(source_name, None)
                    self.index_state_store.save(state_documents)

                files_to_process = [
                    json_file
                    for json_file in json_files
                    if json_file.name in signatures and state_documents.get(json_file.name) != signatures[json_file.name]
                ]
                start_message = f"检测到 {len(files_to_process)} 个 JSON 文件需要更新（共 {len(signatures)} 个）。"

            total = len(files_to_process)
            if total == 0:
                self.index_state_store.save(state_documents)
                self._write_stream_event(
                    {
                        "type": "done",
                        "processed": 0,
                        "total": 0,
                        "mode": mode,
                        "message": "没有检测到需要处理的文件。",
                    }
                )
                return

            self._write_stream_event(
                {
                    "type": "status",
                    "message": start_message,
                    "processed": 0,
                    "total": total,
                    "mode": mode,
                }
            )

            for index, json_file in enumerate(files_to_process, 1):
                self._write_stream_event(
                    {
                        "type": "file",
                        "name": json_file.name,
                        "path": str(json_file),
                        "index": index,
                        "total": total,
                        "mode": mode,
                    }
                )
                pipeline.process_and_store_document(
                    json_file,
                    replace_existing=(mode == "update"),
                )
                processed = index
                state_documents[json_file.name] = signatures[json_file.name]
                self.index_state_store.save(state_documents)
                self._write_stream_event(
                    {
                        "type": "status",
                        "message": f"已完成 {processed}/{total}: {json_file.name}",
                        "processed": processed,
                        "total": total,
                        "mode": mode,
                    }
                )

            self._write_stream_event({"type": "done", "processed": processed, "total": total, "mode": mode})
        except BrokenPipeError:
            raise
        except Exception as exc:
            traceback.print_exc()
            self._write_stream_event(
                {
                    "type": "error",
                    "message": f"{mode_label}向量库失败：{exc}",
                    "mode": mode,
                }
            )
        finally:
            self.index_lock.release()


def build_server(host: str, port: int) -> tuple[ThreadingHTTPServer, int]:
    RAGWebHandler.store = ConversationStore(HISTORY_PATH)
    RAGWebHandler.pipeline_manager = PIPELINE_MANAGER
    RAGWebHandler.auth_sessions = AuthSessionStore(WEB_SESSION_TTL_SECONDS)
    RAGWebHandler.index_state_store = IndexStateStore(INDEX_STATE_PATH)

    last_error: OSError | None = None
    for candidate in range(port, port + 50):
        try:
            return ThreadingHTTPServer((host, candidate), RAGWebHandler), candidate
        except OSError as exc:
            last_error = exc
    raise RuntimeError(f"无法在 {host}:{port}-{port + 49} 找到可用端口。") from last_error


def run_web_server(host: str = WEB_HOST, port: int = WEB_PORT) -> None:
    server, actual_port = build_server(host, port)
    display_host = "127.0.0.1" if host in {"0.0.0.0", ""} else host
    print("Interactive RAG Web 已启动")
    print(f"本机访问地址: http://{display_host}:{actual_port}")
    if WEB_API_KEY:
        print(f"访问验证: 已启用，session 有效期约 {round(WEB_SESSION_TTL_SECONDS / 3600, 2)} 小时")
    else:
        print("访问验证: 未启用。请在 .env 设置 RAG_WEB_API_KEY 后重启。")
    if host in {"0.0.0.0", ""}:
        lan_addresses = discover_lan_addresses()
        if lan_addresses:
            print("局域网访问地址:")
            for ip in lan_addresses:
                print(f"  http://{ip}:{actual_port}")
        else:
            print("局域网地址未能自动识别；请在本机网络设置中查看 IPv4 地址后访问对应端口。")
    elif display_host in {"127.0.0.1", "localhost"}:
        print("当前仅允许本机访问；同一局域网访问请使用 --lan 或 --host 0.0.0.0。")
    print(f"历史记录: {HISTORY_PATH}")
    print("按 Ctrl+C 停止服务。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在关闭服务...")
    finally:
        server.server_close()


def interactive_menu() -> None:
    pipeline = PIPELINE_MANAGER.get()
    history: list[dict[str, Any]] = []

    while True:
        print("====== RAG 交互系统 ======")
        print("1. 从 JSON 文件夹构建/更新向量数据库")
        print("2. 进行问答查询")
        print("3. 退出系统")
        choice = input("请输入选项 (1/2/3): ").strip()

        if choice == "1":
            if not JSON_DIR.exists():
                print(f"错误: JSON 文件夹 {JSON_DIR} 不存在！")
                continue

            json_files = list(JSON_DIR.glob("*.json"))
            if not json_files:
                print(f"未在 {JSON_DIR} 找到任何 JSON 文件。")
                continue

            for json_file in json_files:
                pipeline.process_and_store_document(json_file)
            print("所有文档构建完毕！\n")

        elif choice == "2":
            question = input("\n请输入您的问题: ").strip()
            if not question:
                continue

            retrieval_question = question
            try:
                rewrite = rewrite_question_for_retrieval(question, history=history)
                suggested_question = str(rewrite.get("rewritten_question") or question).strip() or question
                print("\n建议用于检索的问题：")
                print(suggested_question)
                confirm = input("\n使用该问题检索？(Y=确认 / n=使用原问题 / e=编辑 / c=取消): ").strip().lower()
                if confirm == "c":
                    print("已取消本次查询。\n")
                    continue
                if confirm == "e":
                    edited = input("请输入确认后的检索问题: ").strip()
                    retrieval_question = edited or suggested_question
                elif confirm == "n":
                    retrieval_question = question
                else:
                    retrieval_question = suggested_question
            except Exception as exc:
                print(f"问题改写失败，使用原问题继续：{exc}")

            print("\n正在检索相关文档和图片...")
            txt_results, img_results = pipeline.query(question=retrieval_question, k=200)
            print(f"检索到 {len(txt_results)} 条相关文本，{len(img_results)} 张相关图片。")

            answer = pipeline.ask_llm(
                question,
                txt_results,
                img_results,
                history=history,
                retrieval_question=retrieval_question,
            )
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

        elif choice == "3":
            print("退出系统，再见！")
            break
        else:
            print("无效输入，请重新选择。\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive RAG web app")
    parser.add_argument("--cli", action="store_true", help="运行原命令行交互菜单")
    parser.add_argument("--host", default=WEB_HOST, help="Web 服务监听地址")
    parser.add_argument("--port", type=int, default=WEB_PORT, help="Web 服务端口")
    parser.add_argument("--lan", action="store_true", help="允许同一局域网的其他设备访问网页")
    parser.add_argument("--demo", action="store_true", help="强制使用 Demo 管线测试前端")
    args = parser.parse_args()

    if args.demo:
        os.environ["RAG_DEMO"] = "1"
    if args.lan:
        args.host = "0.0.0.0"

    if args.cli:
        interactive_menu()
    else:
        run_web_server(args.host, args.port)


if __name__ == "__main__":
    main()
