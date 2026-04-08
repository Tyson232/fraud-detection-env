"""
FastAPI application exposing the FraudDetectionEnvironment over HTTP and WebSocket.
Each WebSocket connection gets its own isolated environment instance.
"""
from __future__ import annotations

import json
import sys
import os
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Allow import of top-level modules when running from project root or server/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import FraudAction, FraudObservation, FraudState
from server.environment import FraudDetectionEnvironment

app = FastAPI(
    title="Financial Fraud Detection Environment",
    description="OpenEnv-compliant fraud detection RL environment.",
    version="1.0.0",
)

# ── Global HTTP environment (one per server process) ─────────────────────────
_http_env = FraudDetectionEnvironment()


# ── Request / Response helpers ────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_level: Optional[str] = "easy"


# ══════════════════════════════════════════════════════════════════════════════
#  HTTP Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health() -> dict:
    return {"status": "healthy", "environment": "fraud_detection"}


@app.post("/reset", response_model=FraudObservation)
def reset(body: ResetRequest = ResetRequest()) -> FraudObservation:
    """Start a fresh episode. task_level: 'easy' | 'medium' | 'hard'."""
    level = body.task_level or "easy"
    if level not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=422, detail=f"Invalid task_level: {level}")
    return _http_env.reset(task_level=level)


@app.post("/step", response_model=FraudObservation)
def step(action: FraudAction) -> FraudObservation:
    """Submit one FraudAction; receive next observation + reward."""
    try:
        return _http_env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=FraudState)
def state() -> FraudState:
    """Return current episode metadata."""
    return _http_env.state


@app.get("/web", response_class=HTMLResponse)
def web() -> str:
    """Simple browser dashboard showing live environment state."""
    s = _http_env.state
    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <meta http-equiv="refresh" content="3" />
      <title>Fraud Detection Environment</title>
      <style>
        body {{ font-family: monospace; background: #0d1117; color: #c9d1d9; padding: 2rem; }}
        h1 {{ color: #58a6ff; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px;
                 padding: 1.5rem; margin: 1rem 0; max-width: 680px; }}
        .label {{ color: #8b949e; font-size: 0.85em; text-transform: uppercase; }}
        .value {{ font-size: 1.1em; color: #e6edf3; }}
        .good {{ color: #3fb950; }}
        .bad  {{ color: #f85149; }}
        .warn {{ color: #d29922; }}
        table {{ border-collapse: collapse; width: 100%; }}
        td, th {{ padding: 0.4rem 0.8rem; border: 1px solid #30363d; text-align: left; }}
        th {{ background: #21262d; }}
      </style>
    </head>
    <body>
      <h1>&#x1F4B3; Fraud Detection Environment</h1>
      <div class="card">
        <table>
          <tr><th>Field</th><th>Value</th></tr>
          <tr><td class="label">Episode ID</td>
              <td class="value">{s.episode_id or "—"}</td></tr>
          <tr><td class="label">Task Level</td>
              <td class="value">{s.task_level}</td></tr>
          <tr><td class="label">Step</td>
              <td class="value">{s.step_count} / {s.total_transactions}</td></tr>
          <tr><td class="label">Current Score</td>
              <td class="value {'good' if s.current_score >= 0.6 else 'warn' if s.current_score >= 0.3 else 'bad'}">{s.current_score:.4f}</td></tr>
          <tr><td class="label">Correct Classifications</td>
              <td class="value good">{s.correct_classifications}</td></tr>
          <tr><td class="label">False Positives</td>
              <td class="value {'bad' if s.false_positives > 0 else 'good'}">{s.false_positives}</td></tr>
          <tr><td class="label">False Negatives</td>
              <td class="value {'bad' if s.false_negatives > 0 else 'good'}">{s.false_negatives}</td></tr>
          <tr><td class="label">Pattern Complexity</td>
              <td class="value">{s.pattern_complexity}</td></tr>
        </table>
      </div>
      <p style="color:#484f58; font-size:0.8em;">Auto-refreshes every 3 seconds.</p>
    </body>
    </html>
    """
    return html


# ══════════════════════════════════════════════════════════════════════════════
#  WebSocket Endpoint — each connection gets its own isolated environment
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    Persistent WebSocket session.

    Message format (JSON):
      {"type": "reset", "task_level": "easy"}
      {"type": "step", "action": { ...FraudAction fields... }}
      {"type": "state"}

    Response is always a JSON object:
      reset → FraudObservation
      step  → FraudObservation (includes reward)
      state → FraudState
      error → {"error": "message"}
    """
    await websocket.accept()
    # Each WebSocket connection owns its own isolated environment
    ws_env = FraudDetectionEnvironment()

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            msg_type = msg.get("type")

            if msg_type == "reset":
                level = msg.get("task_level", "easy")
                try:
                    obs = ws_env.reset(task_level=level)
                    await websocket.send_text(obs.model_dump_json())
                except (ValueError, RuntimeError) as exc:
                    await websocket.send_text(json.dumps({"error": str(exc)}))

            elif msg_type == "step":
                action_data = msg.get("action")
                if not action_data:
                    await websocket.send_text(json.dumps({"error": "Missing 'action' field"}))
                    continue
                try:
                    action = FraudAction(**action_data)
                    obs = ws_env.step(action)
                    await websocket.send_text(obs.model_dump_json())
                except (ValueError, RuntimeError, TypeError) as exc:
                    await websocket.send_text(json.dumps({"error": str(exc)}))

            elif msg_type == "state":
                await websocket.send_text(ws_env.state.model_dump_json())

            else:
                await websocket.send_text(
                    json.dumps({"error": f"Unknown message type: {msg_type}"})
                )

    except WebSocketDisconnect:
        pass


def main() -> None:
    """Entry point for openenv-core and pyproject.toml [project.scripts]."""
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
