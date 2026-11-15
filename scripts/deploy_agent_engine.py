"""Deploy the ADK multi-agent advisor to Vertex AI Agent Engine.

    pip install "google-cloud-aiplatform[agent_engines,adk]"
    python scripts/deploy_agent_engine.py            # create
    python scripts/deploy_agent_engine.py --test     # create + send one query

(Equivalent CLI: `adk deploy agent_engine --project $GOOGLE_CLOUD_PROJECT --region us-central1
 --staging_bucket $STAGING_BUCKET advisor/adk_agent`.)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import vertexai  # noqa: E402
from vertexai import agent_engines  # noqa: E402

from advisor.adk_agent.agent import root_agent  # noqa: E402
from advisor.config import settings  # noqa: E402
from advisor.rag.retriever import build_index  # noqa: E402

s = settings()
vertexai.init(project=s.project, location=s.location, staging_bucket=os.environ["STAGING_BUCKET"])
build_index()  # ship a pre-built index with the package so cold starts don't re-embed

app = agent_engines.AdkApp(agent=root_agent, enable_tracing=True)
remote = agent_engines.create(
    agent_engine=app,
    display_name="agentic-financial-advisor",
    description="Multi-agent financial advisor (ADK + Gemini + RAG)",
    requirements=[
        "google-cloud-aiplatform[agent_engines,adk]",
        "google-genai",
        "numpy",
        "python-dotenv",
        "mcp>=1.9,<2",
        "langgraph",
    ],
    extra_packages=["advisor", "data"],
    env_vars={"ADVISOR_MODEL": s.model, "EMBEDDING_MODEL": s.embedding_model},
)
print("Deployed:", remote.resource_name)

if "--test" in sys.argv:
    for ev in remote.stream_query(user_id="smoke", message="How long until $10,000 doubles at 6%?"):
        print(ev)
