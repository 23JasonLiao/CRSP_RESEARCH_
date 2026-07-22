from __future__ import annotations

import argparse
import os

import uvicorn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Part1-Part7 evidence-audit research server.")
    parser.add_argument("--host", default=os.getenv("APP_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("APP_PORT", "8000")))
    parser.add_argument("--reload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-level", choices=("critical", "error", "warning", "info", "debug", "trace"), default="info")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        f"Evidence-audit UI: http://{args.host}:{args.port} | "
        f"Part7 API key={'configured' if os.getenv('OPENAI_API_KEY') else 'preview mode'}"
    )
    uvicorn.run("api_server:app", host=args.host, port=args.port, reload=args.reload, log_level=args.log_level)
