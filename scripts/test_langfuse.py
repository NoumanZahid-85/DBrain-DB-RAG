import argparse
import logging
import os
import sys
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test LLM call with Langfuse tracing")
    parser.add_argument("--prompt", "-p", default="Hello, who is the best athlete in the world?",
                        help="Prompt to send to the model")
    parser.add_argument("--model", "-m", default="gemini-2.5-flash", help="Model name")
    parser.add_argument("--temperature", "-t", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--env-file", "-e", default=None,
                        help="Path to a .env file to load (optional). If omitted, defaults are used")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    if args.env_file:
        from pathlib import Path

        env_path = Path(args.env_file).expanduser()
        if not env_path.exists():
            logging.error("Specified env file not found: %s", env_path)
            return 3
        load_dotenv(dotenv_path=str(env_path))
        logging.info("Loaded environment from %s", env_path)
    else:
        load_dotenv()
        logging.info("Loaded environment from default locations (if present)")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logging.error("Environment variable GOOGLE_API_KEY is not set. Set it and retry.")
        return 2

    handler = CallbackHandler()

    llm = ChatGoogleGenerativeAI(
        model=args.model,
        google_api_key=api_key,
        temperature=args.temperature,
    )

    try:
        logging.info("Sending prompt to model: %s", args.prompt)
        response = llm.invoke([
            HumanMessage(content=args.prompt)
        ], config={"callbacks": [handler]})
        content = getattr(response, "content", None) or str(response)
        print(content)
        return 0

    except Exception as exc:
        logging.exception("LLM call failed: %s", exc)
        return 1

    finally:
        # Best-effort flush of Langfuse client if present
        client = getattr(handler, "_langfuse_client", None)
        try:
            if client is not None:
                client.flush()
                logging.info("Flushed Langfuse client events.")
        except Exception:
            logging.debug("Failed to flush Langfuse client", exc_info=True)


if __name__ == "__main__":
    raise SystemExit(main())