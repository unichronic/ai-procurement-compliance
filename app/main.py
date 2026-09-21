import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.db.session import init_db
from app.db.neo4j_client import get_neo4j_client
from app.api.router import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup sequence
    logger.info("Initializing relational database tables...")
    init_db()

    logger.info("Setting up Neo4j graph constraints and indexes...")
    get_neo4j_client().create_constraints()

    logger.info("FastAPI application startup complete.")
    yield
    # Shutdown sequence
    logger.info("Shutting down application resources...")
    get_neo4j_client().close()

app = FastAPI(
    title=settings.APP_NAME,
    description="""
    ## AI-Powered Indian Standards (IS) Procurement Compliance Engine API
    
    This service manages:
    - **Data Ingestion Pipeline**: Fan-out writes to PostgreSQL (SSOT), Neo4j (Graph), ChromaDB (Vector DB), and BM25 Sparse Keyword Index.
    - **Derived Stores Rebuild**: Automated resync to keep vector, graph, and keyword stores consistent with PostgreSQL.
    - **Corpus Statistics**: Real-time stats across all 4 storage layers.
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS for Frontend and Extension Integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to AI Procurement Compliance Recommendation Engine API",
        "docs_url": "/docs",
        "health_check": "/api/v1/health"
    }
