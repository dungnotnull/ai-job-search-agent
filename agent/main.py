"""
ai-job-search-enhanced — entry point
CLI + FastAPI server for the Career Intelligence Agent.
"""

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import typer
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from agent.orchestrator import JobSearchOrchestrator

app_cli = typer.Typer(name="ai-job-search", help="AI-powered career intelligence agent")
_orchestrator: Optional[JobSearchOrchestrator] = None

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def get_orchestrator() -> JobSearchOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = JobSearchOrchestrator()
    return _orchestrator


@asynccontextmanager
async def lifespan(app: FastAPI):
    missing = []
    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        if os.getenv("PRIVACY_MODE", "false").lower() != "true":
            missing.append("ANTHROPIC_API_KEY or OPENAI_API_KEY")
    if missing:
        logging.warning("Missing env vars (some features will use fallbacks): %s", ", ".join(missing))
    get_orchestrator()
    logging.info("ai-job-search-enhanced API started")
    yield
    logging.info("ai-job-search-enhanced API shutting down")


app_api = FastAPI(
    title="ai-job-search-enhanced API",
    version="1.0.0",
    lifespan=lifespan,
)

app_api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app_api.middleware("http")
async def request_logging(request: Request, call_next):
    logging.info("%s %s", request.method, request.url.path)
    response = await call_next(request)
    return response


@app_api.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


# -- Pydantic schemas --

class SearchRequest(BaseModel):
    resume_text: str = Field(..., min_length=10, description="Resume text content")
    query: str = Field(..., min_length=2, description="Job search query")
    n_jobs: int = Field(default=20, ge=1, le=100, description="Number of jobs to fetch")
    tone: str = Field(default="formal", description="Cover letter tone: formal|casual|technical|creative")
    precision_mode: bool = False

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        if v not in ("formal", "casual", "technical", "creative"):
            raise ValueError("tone must be one of: formal, casual, technical, creative")
        return v


class MatchRequest(BaseModel):
    resume_text: str = Field(..., min_length=10)
    job_descriptions: list[str] = Field(..., min_length=1, max_length=50)
    precision_mode: bool = False


class CoverLetterRequest(BaseModel):
    resume_text: str = Field(..., min_length=10)
    job_description: str = Field(..., min_length=10)
    company_name: str = Field(default="", max_length=200)
    tone: str = Field(default="formal")

    @field_validator("tone")
    @classmethod
    def validate_tone(cls, v: str) -> str:
        if v not in ("formal", "casual", "technical", "creative"):
            raise ValueError("tone must be one of: formal, casual, technical, creative")
        return v


class SalaryRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=200)
    location: str = Field(..., min_length=2, max_length=200)
    company_size: str = Field(default="mid", pattern="^(startup|small|mid|large|enterprise)$")
    seniority: str = Field(default="mid", pattern="^(junior|mid|senior|staff|principal)$")
    industry: str = Field(default="technology", max_length=100)
    remote: bool = False
    years_required: int = Field(default=3, ge=0, le=30)


class SentimentRequest(BaseModel):
    company_name: str = Field(..., min_length=1, max_length=200)
    reviews: list[str] = Field(default_factory=list, max_length=500)


class GapAnalysisRequest(BaseModel):
    resume_text: str = Field(..., min_length=10)
    job_descriptions: list[str] = Field(..., min_length=1, max_length=10)


# -- REST API endpoints --

@app_api.get("/health")
async def health():
    return {"status": "ok", "agent": "ai-job-search-enhanced", "version": "1.0.0"}


@app_api.post("/search")
async def search_jobs(req: SearchRequest):
    orc = get_orchestrator()
    try:
        result = await orc.full_search(
            resume_text=req.resume_text,
            query=req.query,
            n_jobs=req.n_jobs,
            tone=req.tone,
            precision_mode=req.precision_mode,
        )
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app_api.post("/match")
async def match_jobs(req: MatchRequest):
    orc = get_orchestrator()
    results = await orc.match_only(
        resume_text=req.resume_text,
        job_descriptions=req.job_descriptions,
        precision_mode=req.precision_mode,
    )
    return {"matches": [r.__dict__ for r in results]}


@app_api.post("/cover-letter")
async def generate_cover_letter(req: CoverLetterRequest):
    orc = get_orchestrator()
    letter = await orc.generate_cover_letter(
        resume_text=req.resume_text,
        job_description=req.job_description,
        company_name=req.company_name,
        tone=req.tone,
    )
    return letter.__dict__


@app_api.post("/salary")
async def predict_salary(req: SalaryRequest):
    orc = get_orchestrator()
    pred = orc.salary_predictor.predict({
        "title": req.title,
        "location": req.location,
        "company_size": req.company_size,
        "seniority": req.seniority,
        "industry": req.industry,
        "remote": req.remote,
        "years_required": req.years_required,
    })
    return pred.__dict__


@app_api.post("/sentiment")
async def company_sentiment(req: SentimentRequest):
    orc = get_orchestrator()
    score = orc.company_sentiment.score(req.company_name, req.reviews)
    return score.__dict__


@app_api.post("/gap-analysis")
async def gap_analysis(req: GapAnalysisRequest):
    orc = get_orchestrator()
    analysis = await orc.analyze_gaps(
        resume_text=req.resume_text,
        job_descriptions=req.job_descriptions,
    )
    return analysis.__dict__


@app_api.post("/knowledge/update")
async def trigger_knowledge_update(background_tasks: BackgroundTasks):
    from tools.knowledge_updater import KnowledgeUpdater
    updater = KnowledgeUpdater()
    background_tasks.add_task(updater.run)
    return {"status": "knowledge update started in background"}


@app_api.get("/metrics")
async def metrics():
    orc = get_orchestrator()
    return orc.memory.get_session_stats()


# -- CLI commands --

@app_cli.command()
def search(
    resume: Path = typer.Argument(..., help="Path to resume file (PDF or .txt)"),
    query: str = typer.Argument(..., help="Job search query"),
    n_jobs: int = typer.Option(20, "--n-jobs", "-n", help="Number of jobs to fetch"),
    tone: str = typer.Option("formal", "--tone", "-t", help="Cover letter tone"),
    precision: bool = typer.Option(False, "--precision", "-p", help="Use BGE-large (slower, more accurate)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Save report to file"),
):
    """Run a full career intelligence search."""
    resume_text = _read_resume(resume)
    orc = get_orchestrator()

    typer.echo(f"Searching for: {query}")
    typer.echo(f"Resume loaded: {len(resume_text)} chars")

    result = asyncio.run(orc.full_search(
        resume_text=resume_text,
        query=query,
        n_jobs=n_jobs,
        tone=tone,
        precision_mode=precision,
    ))

    report_md = result.get("report_markdown", "No report generated.")
    if output:
        output.write_text(report_md, encoding="utf-8")
        typer.echo(f"Report saved to {output}")
    else:
        typer.echo("\n" + report_md)


@app_cli.command()
def match(
    resume: Path = typer.Argument(..., help="Resume file"),
    jobs_file: Path = typer.Argument(..., help="JSON file with job descriptions"),
    precision: bool = typer.Option(False, "--precision", "-p"),
):
    """Match a resume against a list of job descriptions."""
    resume_text = _read_resume(resume)
    job_descs = json.loads(jobs_file.read_text(encoding="utf-8"))
    orc = get_orchestrator()
    results = asyncio.run(orc.match_only(resume_text, job_descs, precision))
    for i, r in enumerate(results, 1):
        typer.echo(f"{i}. [{r.score:.3f}] {r.title}")


@app_cli.command()
def cover_letter(
    resume: Path = typer.Argument(..., help="Resume file"),
    jd: Path = typer.Argument(..., help="Job description file (.txt)"),
    tone: str = typer.Option("formal", "--tone", "-t"),
    company: str = typer.Option("", "--company", "-c"),
):
    """Generate a personalized cover letter."""
    resume_text = _read_resume(resume)
    jd_text = jd.read_text(encoding="utf-8")
    orc = get_orchestrator()
    letter = asyncio.run(orc.generate_cover_letter(resume_text, jd_text, company, tone))
    typer.echo(f"\nSubject: {letter.subject_line}\n")
    typer.echo(letter.body)


@app_cli.command()
def salary(
    title: str = typer.Argument(...),
    location: str = typer.Argument(...),
    seniority: str = typer.Option("mid", "--seniority", "-s"),
    industry: str = typer.Option("technology", "--industry"),
    remote: bool = typer.Option(False, "--remote"),
):
    """Predict salary range for a job."""
    orc = get_orchestrator()
    pred = orc.salary_predictor.predict({
        "title": title, "location": location, "company_size": "mid",
        "seniority": seniority, "industry": industry, "remote": remote,
        "years_required": 3,
    })
    typer.echo(f"Predicted salary: ${pred.low:,} - ${pred.median:,} - ${pred.high:,} (confidence: {pred.confidence:.0%})")


@app_cli.command()
def gap_analysis(
    resume: Path = typer.Argument(..., help="Resume file"),
    jd: Path = typer.Argument(..., help="Job description file (.txt)"),
):
    """Analyze skill gaps between a resume and a target role."""
    resume_text = _read_resume(resume)
    jd_text = jd.read_text(encoding="utf-8")
    orc = get_orchestrator()
    analysis = asyncio.run(orc.analyze_gaps(resume_text, [jd_text]))
    typer.echo(f"\nMatch: {analysis.match_percentage}%")
    typer.echo(f"Time to close gaps: {analysis.time_estimate}\n")
    typer.echo("Missing skills (priority ranked):")
    for gap in analysis.missing_skills:
        typer.echo(f"  [{gap['priority']}] {gap['skill']} - {gap['reason']}")
    typer.echo("\nLearning path:")
    for step in analysis.learning_path:
        typer.echo(f"  Step {step['step']}: {step['action']} ({step['weeks']}w) - {step.get('resource','')}")


@app_cli.command()
def update_knowledge():
    """Trigger a manual knowledge base update."""
    from tools.knowledge_updater import KnowledgeUpdater
    updater = KnowledgeUpdater()
    stats = asyncio.run(updater.run())
    typer.echo(f"Knowledge update complete: {stats.get('new_entries', 0)} new entries added.")


@app_cli.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8017, "--port"),
):
    """Start the FastAPI REST server."""
    _start_server(host=host, port=port)


@app_cli.command(name="cost-report")
def cost_report():
    """Show LLM API cost summary."""
    orc = get_orchestrator()
    report = orc.memory.get_cost_summary()
    typer.echo(json.dumps(report, indent=2))


def _read_resume(path: Path) -> str:
    if not path.exists():
        typer.echo(f"Resume file not found: {path}", err=True)
        raise typer.Exit(1)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            typer.echo("pdfplumber not installed; treating as plain text", err=True)
    return path.read_text(encoding="utf-8", errors="replace")


def _start_server(host: str = "0.0.0.0", port: int = 8017):
    typer.echo(f"Starting server at http://{host}:{port}")
    uvicorn.run(app_api, host=host, port=port)


if __name__ == "__main__":
    app_cli()
