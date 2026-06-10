"""
knowledge_updater.py — Research paper crawler for ai-job-search-enhanced.
Sources: ArXiv cs.CL + cs.IR, Semantic Scholar, Papers with Code.
Schedule: Weekly Sunday 02:00 (APScheduler). Updates SECOND-KNOWLEDGE-BRAIN.md.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

BRAIN_FILE = Path(__file__).parent.parent / "SECOND-KNOWLEDGE-BRAIN.md"
DATA_DIR = Path(__file__).parent.parent / "data"

ARXIV_QUERIES = [
    ("cs.CL", "job OR resume OR career OR matching OR recommendation"),
    ("cs.IR", "job OR resume OR matching OR information retrieval"),
    ("cs.LG", "salary prediction OR job recommendation"),
]

SEMANTIC_SCHOLAR_QUERIES = [
    "semantic job matching transformer",
    "resume parsing skill extraction",
    "salary prediction job posting NLP",
    "career recommendation system deep learning",
]

PAPERS_WITH_CODE_TASKS = [
    "Information Retrieval",
    "Sentence Embeddings",
    "Document Similarity",
]

DOMAIN_KEYWORDS = [
    "job", "resume", "career", "salary", "matching", "recommendation",
    "skill extraction", "sentence-transformer", "dense-retrieval",
    "person-job fit", "job posting", "talent acquisition",
]


class KnowledgeUpdater:
    """Crawl research papers and update SECOND-KNOWLEDGE-BRAIN.md."""

    def __init__(self):
        self._memory: Optional[object] = None
        DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def memory(self):
        if self._memory is None:
            from agent.memory.memory_manager import MemoryManager
            self._memory = MemoryManager()
        return self._memory

    async def run(self) -> dict:
        logger.info("Knowledge update started")
        papers = []

        arxiv_papers = await self._crawl_arxiv()
        papers.extend(arxiv_papers)

        scholar_papers = await self._crawl_semantic_scholar()
        papers.extend(scholar_papers)

        pwc_papers = await self._crawl_papers_with_code()
        papers.extend(pwc_papers)

        new_papers = self._deduplicate(papers)
        scored = self._score_papers(new_papers)
        top_papers = scored[:10]

        self._append_to_brain(top_papers)

        for p in top_papers:
            self.memory.mark_paper_known(p["title"], p.get("url", ""))

        logger.info("Knowledge update complete: %d new entries added", len(top_papers))
        return {"new_entries": len(top_papers), "total_candidates": len(papers)}

    async def _crawl_arxiv(self) -> list[dict]:
        papers = []
        for category, terms in ARXIV_QUERIES:
            query = f"cat:{category}+AND+({terms.replace(' ', '+')})"
            url = (
                f"https://export.arxiv.org/api/query"
                f"?search_query={query}"
                f"&max_results=15"
                f"&sortBy=submittedDate&sortOrder=descending"
            )
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ai-job-search-enhanced/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    xml_data = resp.read()
                root = ET.fromstring(xml_data)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    title_el = entry.find("atom:title", ns)
                    abstract_el = entry.find("atom:summary", ns)
                    link_el = entry.find("atom:id", ns)
                    published_el = entry.find("atom:published", ns)
                    authors_els = entry.findall("atom:author/atom:name", ns)

                    if title_el is None:
                        continue

                    papers.append({
                        "title": title_el.text.strip().replace("\n", " "),
                        "abstract": (abstract_el.text or "").strip()[:500],
                        "url": (link_el.text or "").strip(),
                        "published": (published_el.text or "")[:10],
                        "authors": ", ".join(a.text for a in authors_els[:3] if a.text),
                        "source": f"ArXiv:{category}",
                    })
            except Exception as exc:
                logger.warning("ArXiv %s query failed: %s", category, exc)
        return papers

    async def _crawl_semantic_scholar(self) -> list[dict]:
        papers = []
        base = "https://api.semanticscholar.org/graph/v1/paper/search"
        fields = "title,authors,year,venue,externalIds,abstract"
        for q in SEMANTIC_SCHOLAR_QUERIES:
            url = f"{base}?query={q.replace(' ', '+')}&fields={fields}&limit=8"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ai-job-search-enhanced/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                for paper in data.get("data", []):
                    title = paper.get("title") or ""
                    ext_ids = paper.get("externalIds") or {}
                    doi = ext_ids.get("DOI") or ext_ids.get("ArXiv") or ""
                    papers.append({
                        "title": title.strip(),
                        "abstract": (paper.get("abstract") or "")[:500],
                        "url": f"https://doi.org/{doi}" if doi else "",
                        "published": str(paper.get("year") or ""),
                        "authors": ", ".join(
                            a["name"] for a in (paper.get("authors") or [])[:3]
                        ),
                        "source": f"SemanticScholar:{q[:30]}",
                    })
            except Exception as exc:
                logger.warning("Semantic Scholar query '%s' failed: %s", q, exc)
        return papers

    async def _crawl_papers_with_code(self) -> list[dict]:
        papers = []
        for task in PAPERS_WITH_CODE_TASKS:
            url = (
                f"https://paperswithcode.com/api/v1/search/"
                f"?q={task.replace(' ', '+')}&limit=8"
            )
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "ai-job-search-enhanced/1.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                for result in data.get("results", []):
                    paper = result.get("paper", {})
                    title = paper.get("title", "")
                    if not title:
                        continue
                    papers.append({
                        "title": title.strip(),
                        "abstract": (paper.get("abstract") or "")[:500],
                        "url": paper.get("url", ""),
                        "published": (paper.get("published") or "")[:10],
                        "authors": "",
                        "source": f"PapersWithCode:{task}",
                    })
            except Exception as exc:
                logger.warning("Papers with Code '%s' query failed: %s", task, exc)
        return papers

    def _deduplicate(self, papers: list[dict]) -> list[dict]:
        seen = set()
        unique = []
        for p in papers:
            title = p.get("title", "").lower().strip()
            if not title or len(title) < 10:
                continue
            h = hashlib.sha256(title.encode()).hexdigest()
            if h in seen:
                continue
            if self.memory.is_known_paper(title, p.get("url", "")):
                continue
            seen.add(h)
            unique.append(p)
        return unique

    def _score_papers(self, papers: list[dict]) -> list[dict]:
        scored = []
        today = datetime.now()
        for p in papers:
            text = f"{p.get('title','')} {p.get('abstract','')}".lower()
            relevance = sum(1 for kw in DOMAIN_KEYWORDS if kw in text) / len(DOMAIN_KEYWORDS)

            try:
                pub_date = datetime.strptime(p.get("published", "2020-01-01")[:10], "%Y-%m-%d")
                days_old = (today - pub_date).days
                recency = max(0.0, 1.0 - days_old / 365)
            except Exception:
                recency = 0.5

            p["_score"] = 0.6 * recency + 0.4 * relevance
            scored.append(p)

        scored.sort(key=lambda x: x["_score"], reverse=True)
        return scored

    def _append_to_brain(self, papers: list[dict]):
        if not papers:
            return

        today = datetime.now().strftime("%Y-%m-%d")
        table_rows = []
        for p in papers:
            title = p.get("title", "Unknown")[:80]
            authors = p.get("authors", "N/A")[:40]
            year = p.get("published", "")[:4]
            url = p.get("url", "N/A")
            abstract = p.get("abstract", "")[:120].replace("\n", " ") + "..."
            source = p.get("source", "")
            table_rows.append(
                f"| {title} | {authors} | {year} | {source} | {url} | {abstract} |"
            )

        log_entry = f"\n### {today} — Scheduled Update\n"
        log_entry += f"- **Added:** {len(papers)} new papers\n"
        log_entry += "- **Sources:** ArXiv (cs.CL, cs.IR, cs.LG), Semantic Scholar, Papers with Code\n"
        log_entry += f"- **Next run:** {(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')} 02:00\n"

        if papers:
            log_entry += "\n**New papers added this run:**\n"
            log_entry += "| Title | Authors | Year | Source | Link | Key Finding |\n"
            log_entry += "|-------|---------|------|--------|------|-------------|\n"
            log_entry += "\n".join(table_rows) + "\n"

        brain_text = BRAIN_FILE.read_text(encoding="utf-8") if BRAIN_FILE.exists() else ""
        log_section_marker = "## Knowledge Update Log"

        if log_section_marker in brain_text:
            insert_pos = brain_text.index(log_section_marker) + len(log_section_marker)
            new_content = brain_text[:insert_pos] + "\n" + log_entry + brain_text[insert_pos:]
        else:
            new_content = brain_text + f"\n\n{log_section_marker}\n" + log_entry

        self._atomic_write(BRAIN_FILE, new_content)

    def _atomic_write(self, path: Path, content: str):
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".md", dir=str(path.parent))
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(content)
            Path(tmp_path).replace(path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    def schedule(self):
        """Start APScheduler for weekly runs at Sunday 02:00."""
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
            scheduler = BackgroundScheduler()
            scheduler.add_job(
                lambda: asyncio.run(self.run()),
                "cron",
                day_of_week="sun",
                hour=2,
                minute=0,
                id="weekly_knowledge_update",
            )
            scheduler.start()
            logger.info("Knowledge updater scheduled: weekly Sunday 02:00")
            return scheduler
        except ImportError:
            logger.warning("APScheduler not installed; manual updates only")
            return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    stats = asyncio.run(KnowledgeUpdater().run())
    print(f"Done: {stats}")
