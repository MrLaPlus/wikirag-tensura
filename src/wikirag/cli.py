import json
import os
import sys
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import typer

from wikirag.chunking.chunker import SectionAwareChunker
from wikirag.chunking.embedder import LocalSentenceTransformerEmbedder
from wikirag.config import load_project_config
from wikirag.connectors.mediawiki import MediaWikiConnector
from wikirag.generation.generator import GroundedAnswerGenerator
from wikirag.llm import get_llm_provider
from wikirag.parser.wikitext import WikitextParser
from wikirag.retrieval.pipeline import RetrievalPipeline
from wikirag.utils.logging import get_logger
from wikirag.vectorstore.lancedb_store import LanceDBStore

# Load environment variables from .env if present
load_dotenv()

app = typer.Typer(
    name="wikirag",
    help="Production-quality local RAG platform for fandom and knowledge wikis.",
    add_completion=False,
)
console = Console()
logger = get_logger("wikirag.cli")


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the project to initialize (e.g. tensura)"),
    api_url: str = typer.Option("https://tensura.fandom.com/api.php", help="MediaWiki api.php URL"),
):
    """Initializes a new project configuration file."""
    proj_file = Path("projects") / f"{project_name}.yaml"
    if proj_file.exists():
        console.print(f"[bold red]Project config '{proj_file}' already exists![/bold red]")
        raise typer.Exit(1)

    proj_file.parent.mkdir(parents=True, exist_ok=True)
    template = f"""project:
  name: {project_name}
  title: "{project_name.capitalize()} Wiki"
  description: "Knowledge base for {project_name}"
  license: "CC BY-SA 3.0"
  attribution_required: true

source:
  type: mediawiki
  api_url: "{api_url}"
  base_url: "{api_url.replace('/api.php', '/wiki')}"
  namespace: 0
  user_agent: "WikiRAG/0.1.0"
  request_delay_seconds: 0.5
  batch_size: 50
  checkpoint_interval: 100

storage:
  data_dir: "./data/{project_name}"
  raw_dir: "./data/{project_name}/raw"
  parsed_dir: "./data/{project_name}/parsed"
  embeddings_cache_dir: "./data/{project_name}/embeddings"
  vectordb_dir: "./data/{project_name}/vectordb"
  alias_map_path: "./data/{project_name}/aliases.json"
  state_file: "./data/{project_name}/state.json"
  failed_pages_path: "./data/{project_name}/failed_pages.jsonl"
"""
    with open(proj_file, "w", encoding="utf-8") as f:
        f.write(template)

    console.print(f"[bold green]Created project config at: {proj_file}[/bold green]")


@app.command()
def crawl(
    project: str = typer.Option("tensura", "--project", "-p", help="Project name"),
    aliases_only: bool = typer.Option(False, "--aliases-only", help="Build alias map only"),
    no_resume: bool = typer.Option(False, "--no-resume", help="Ignore checkpoint and restart"),
):
    """Crawls pages from the wiki and writes raw records to disk."""
    cfg = load_project_config(project)
    connector = MediaWikiConnector(cfg)

    # 1. Build alias map
    alias_map = connector.build_alias_map()
    if aliases_only:
        console.print(f"[green]Alias map built with {len(alias_map)} entries.[/green]")
        return

    # 2. Crawl articles
    raw_dir = Path(cfg.storage.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_file = raw_dir / "pages.jsonl"

    mode = "a" if not no_resume else "w"
    count = 0
    console.print(f"[cyan]Starting crawl for project '{project}'...[/cyan]")

    with open(raw_file, mode, encoding="utf-8") as f:
        for page in connector.crawl_all(resume=not no_resume):
            f.write(json.dumps(page, ensure_ascii=False) + "\n")
            count += 1
            if count % 100 == 0:
                console.print(f"Fetched {count} articles...")

    console.print(f"[bold green]Crawl complete! Total pages saved to {raw_file}: {count}[/bold green]")


@app.command()
def parse(
    project: str = typer.Option("tensura", "--project", "-p", help="Project name"),
):
    """Parses raw fetched wikitext into structured clean documents and chunks."""
    cfg = load_project_config(project)
    raw_file = Path(cfg.storage.raw_dir) / "pages.jsonl"
    if not raw_file.exists():
        console.print(f"[bold red]Raw data file {raw_file} not found. Run 'wikirag crawl' first.[/bold red]")
        raise typer.Exit(1)

    parsed_dir = Path(cfg.storage.parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    parsed_file = parsed_dir / "parsed_pages.jsonl"
    chunks_file = parsed_dir / "chunks.jsonl"

    # Load alias map if available
    alias_map = {}
    if Path(cfg.storage.alias_map_path).exists():
        with open(cfg.storage.alias_map_path, "r", encoding="utf-8") as f:
            alias_map = json.load(f)

    parser = WikitextParser(failed_pages_path=cfg.storage.failed_pages_path)
    chunker = SectionAwareChunker(cfg.chunking, project_name=project)

    page_count = 0
    chunk_count = 0

    console.print(f"[cyan]Parsing raw wikitext from {raw_file}...[/cyan]")
    with open(raw_file, "r", encoding="utf-8") as rf, \
         open(parsed_file, "w", encoding="utf-8") as pf, \
         open(chunks_file, "w", encoding="utf-8") as cf:

        for line in rf:
            if not line.strip():
                continue
            raw_rec = json.loads(line)
            parsed_page = parser.parse_page(raw_rec, alias_map=alias_map)
            if not parsed_page:
                continue

            pf.write(parsed_page.model_dump_json() + "\n")
            page_count += 1

            for chunk in chunker.chunk_page(parsed_page):
                cf.write(chunk.model_dump_json() + "\n")
                chunk_count += 1

    console.print(
        f"[bold green]Parsing complete! Successfully structured {page_count} pages into {chunk_count} chunks.[/bold green]"
    )


@app.command()
def embed(
    project: str = typer.Option("tensura", "--project", "-p", help="Project name"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", help="Override embedding batch size"),
):
    """Embeds parsed chunks and indexes them into LanceDB."""
    cfg = load_project_config(project)
    chunks_file = Path(cfg.storage.parsed_dir) / "chunks.jsonl"
    if not chunks_file.exists():
        console.print(f"[bold red]Chunks file {chunks_file} not found. Run 'wikirag parse' first.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Loading chunks from {chunks_file}...[/cyan]")
    chunks = []
    texts = []
    with open(chunks_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                c = json.loads(line)
                chunks.append(c)
                texts.append(c["chunk_text"])

    console.print(f"Total chunks to embed: {len(chunks)}")

    # Initialize embedder
    bs = batch_size or cfg.embedding.batch_size
    embedder = LocalSentenceTransformerEmbedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
        backend=cfg.embedding.backend,
        quantization=cfg.embedding.quantization,
        batch_size=bs,
        cache_dir=cfg.storage.embeddings_cache_dir,
        normalize=cfg.embedding.normalize,
    )

    # Initialize vector store
    store = LanceDBStore(
        db_path=cfg.storage.vectordb_dir,
        table_name=cfg.vectorstore.table_name,
        dimension=embedder.dimension,
    )

    # Embed and upsert in batches
    console.print("[cyan]Generating embeddings with disk cache enabled...[/cyan]")
    vectors = embedder.embed_texts(texts, show_progress=True)

    console.print("[cyan]Upserting vectors into LanceDB...[/cyan]")
    inserted = store.upsert_chunks(chunks, vectors)
    console.print(
        f"[bold green]Embedding and indexing complete! Total vectors in table: {store.count()} (New: {inserted})[/bold green]"
    )


@app.command()
def sync(
    project: str = typer.Option("tensura", "--project", "-p", help="Project name"),
    incremental: bool = typer.Option(True, "--incremental/--full", help="Incremental or full sync"),
):
    """Syncs wiki updates, parses changed pages, and updates LanceDB index."""
    cfg = load_project_config(project)
    connector = MediaWikiConnector(cfg)

    console.print(f"[cyan]Syncing project '{project}' (incremental={incremental})...[/cyan]")
    records = list(connector.sync_incremental() if incremental else connector.crawl_all(resume=False))

    if not records:
        console.print("[green]Index is already up to date! Zero changed pages found.[/green]")
        return

    console.print(f"Found {len(records)} updated pages. Parsing and embedding...")
    alias_map = {}
    if Path(cfg.storage.alias_map_path).exists():
        with open(cfg.storage.alias_map_path, "r", encoding="utf-8") as f:
            alias_map = json.load(f)

    parser = WikitextParser(failed_pages_path=cfg.storage.failed_pages_path)
    chunker = SectionAwareChunker(cfg.chunking, project_name=project)

    new_chunks = []
    for raw_rec in records:
        parsed_page = parser.parse_page(raw_rec, alias_map=alias_map)
        if parsed_page:
            for chunk in chunker.chunk_page(parsed_page):
                new_chunks.append(chunk.model_dump())

    if not new_chunks:
        console.print("[yellow]No new valid text chunks extracted.[/yellow]")
        return

    embedder = LocalSentenceTransformerEmbedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
        backend=cfg.embedding.backend,
        quantization=cfg.embedding.quantization,
        batch_size=cfg.embedding.batch_size,
        cache_dir=cfg.storage.embeddings_cache_dir,
    )
    store = LanceDBStore(
        db_path=cfg.storage.vectordb_dir,
        table_name=cfg.vectorstore.table_name,
        dimension=embedder.dimension,
    )

    texts = [c["chunk_text"] for c in new_chunks]
    vectors = embedder.embed_texts(texts, show_progress=True)
    inserted = store.upsert_chunks(new_chunks, vectors)
    console.print(f"[bold green]Sync completed! Upserted {inserted} updated chunks.[/bold green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Search query or question (Thai or English)"),
    project: str = typer.Option("tensura", "--project", "-p", help="Project name"),
    llm: Optional[str] = typer.Option(None, "--llm", help="Provider and model override, e.g. 'ollama:llama3.1' or 'gemini:gemini-2.5-flash'"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of chunks to retrieve"),
    stream: bool = typer.Option(True, "--stream/--no-stream", help="Stream response tokens"),
):
    """Asks a question against the wiki knowledge base using dense retrieval + LLM."""
    cfg = load_project_config(project)

    # 1. Parse LLM provider and model
    provider_name = None
    model_name = None
    if llm:
        if ":" in llm:
            provider_name, model_name = llm.split(":", 1)
        else:
            provider_name = llm
    else:
        provider_name = cfg.llm.default_provider
        model_name = cfg.llm.default_model

    llm_instance = get_llm_provider(provider_name=provider_name, model_name=model_name)

    # 2. Setup embedder & vector store
    embedder = LocalSentenceTransformerEmbedder(
        model_name=cfg.embedding.model_name,
        device=cfg.embedding.device,
        backend=cfg.embedding.backend,
        quantization=cfg.embedding.quantization,
        cache_dir=cfg.storage.embeddings_cache_dir,
    )
    store = LanceDBStore(
        db_path=cfg.storage.vectordb_dir,
        table_name=cfg.vectorstore.table_name,
        dimension=embedder.dimension,
    )

    # 3. Retrieve chunks
    pipeline = RetrievalPipeline(config=cfg, embedder=embedder, vectorstore=store)
    console.print(f"[cyan]Searching for: '{question}'...[/cyan]")
    chunks = pipeline.retrieve(question, top_k=top_k)

    if not chunks:
        console.print("[bold red]No matching content found in the knowledge base.[/bold red]")
        raise typer.Exit(0)

    # 4. Show Sources Summary Table
    table = Table(title="Retrieved Sources", show_header=True, header_style="bold magenta")
    table.add_column("#", width=3)
    table.add_column("Entity", width=25)
    table.add_column("Section", width=25)
    table.add_column("Score", width=8)

    for i, c in enumerate(chunks, 1):
        table.add_row(
            str(i),
            c.get("entity", "Unknown"),
            c.get("section_path", "General"),
            f"{c.get('score', 0.0):.3f}",
        )
    console.print(table)

    # 5. Generate Grounded Answer
    generator = GroundedAnswerGenerator(config=cfg, llm_provider=llm_instance)
    console.print(f"\n[bold green]Answer ({provider_name}:{model_name}):[/bold green]\n")

    if stream:
        for token in generator.stream_answer(question, chunks):
            sys.stdout.write(token)
            sys.stdout.flush()
        print("\n")
    else:
        res = generator.generate_answer(question, chunks)
        console.print(Panel(res.answer, title="WikiRAG Answer", border_style="green"))


@app.command()
def stats(
    project: str = typer.Option("tensura", "--project", "-p", help="Project name"),
):
    """Displays ingestion and index statistics for the project."""
    cfg = load_project_config(project)
    raw_file = Path(cfg.storage.raw_dir) / "pages.jsonl"
    chunks_file = Path(cfg.storage.parsed_dir) / "chunks.jsonl"
    alias_file = Path(cfg.storage.alias_map_path)
    failed_file = Path(cfg.storage.failed_pages_path)

    raw_count = sum(1 for _ in open(raw_file, "r", encoding="utf-8")) if raw_file.exists() else 0
    chunk_count = sum(1 for _ in open(chunks_file, "r", encoding="utf-8")) if chunks_file.exists() else 0
    alias_count = len(json.load(open(alias_file, "r", encoding="utf-8"))) if alias_file.exists() else 0
    failed_count = sum(1 for _ in open(failed_file, "r", encoding="utf-8")) if failed_file.exists() else 0

    table = Table(title=f"WikiRAG Project Statistics: {project}", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row("Raw Articles Crawled", str(raw_count))
    table.add_row("Parsed Chunks", str(chunk_count))
    table.add_row("Alias Map Entities", str(alias_count))
    table.add_row("Quarantined Failed Pages", str(failed_count))
    table.add_row("Default LLM Provider", f"{cfg.llm.default_provider} ({cfg.llm.default_model})")

    console.print(table)


@app.command()
def eval(
    project: str = typer.Option("tensura", "--project", "-p", help="Project config name"),
    golden: str = typer.Option("eval/golden_qa.json", "--golden", "-g", help="Path to golden Q&A dataset"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Top-K retrieval depth to evaluate"),
):
    """Runs automated evaluation benchmark against golden Q&A dataset."""
    from wikirag.api.app import get_pipeline
    from wikirag.eval.runner import EvalRunner

    console.print(f"[bold cyan]Running WikiRAG Evaluation for project '{project}'...[/bold cyan]")
    pipeline = get_pipeline(project)
    runner = EvalRunner(golden_path=golden)
    runner.run_eval(pipeline=pipeline, top_k=top_k)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host interface to bind"),
    port: int = typer.Option(8000, "--port", help="Port number"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Starts the WikiRAG FastAPI backend server (supports SSE streaming & Web UI)."""
    import uvicorn
    console.print(f"[bold green]Starting WikiRAG API Server at http://{host}:{port}...[/bold green]")
    console.print(f"[cyan]API Documentation available at: http://localhost:{port}/docs[/cyan]")
    uvicorn.run("wikirag.api.app:app", host=host, port=port, reload=reload)


def main():

    app()


if __name__ == "__main__":
    main()
