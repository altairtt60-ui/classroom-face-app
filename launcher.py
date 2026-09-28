"""Double-click entry point: starts the server and opens the dashboard in the browser."""
import threading
import time
import webbrowser

import uvicorn

from backend.app.main import app

HOST, PORT = "127.0.0.1", 8000


def open_browser() -> None:
    time.sleep(2)
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
