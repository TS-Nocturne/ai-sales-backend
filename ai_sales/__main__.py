"""Entry point for `python -m ai_sales`."""

import argparse
import sys

from ai_sales.cli import run_chat
from ai_sales.main import run_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ตัวแทนฝ่ายขาย AI และการให้คะแนนลีด",
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("demo", help="เรียกใช้การสาธิตแบบอัตโนมัติที่มี HITL")
    chat_parser = subparsers.add_parser("chat", help="เริ่มการสนทนาแบบโต้ตอบ")
    chat_parser.add_argument(
        "--thread-id",
        help="ทำต่อจากการสนทนาที่มีอยู่แล้ว",
    )

    serve_parser = subparsers.add_parser(
        "serve",
        help="เปิดเซิร์ฟเวอร์ FastAPI ให้ Next.js เรียกใช้ผ่าน HTTP",
    )
    serve_parser.add_argument("--host", default="127.0.0.1", help="โฮสต์ (ค่าเริ่มต้น 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="พอร์ต (ค่าเริ่มต้น 8000)")
    serve_parser.add_argument("--reload", action="store_true", help="รีโหลดอัตโนมัติขณะพัฒนา")

    args = parser.parse_args()

    if args.command == "chat":
        run_chat(thread_id=args.thread_id)
    elif args.command == "demo":
        run_demo()
    elif args.command == "serve":
        import uvicorn

        uvicorn.run(
            "ai_sales.api.server:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
        )
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
