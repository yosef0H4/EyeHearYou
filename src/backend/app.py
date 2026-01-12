"""FastAPI application setup"""
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import config, screenshot, detection, extraction


app = FastAPI()

# Get frontend directory path
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the single page UI"""
    ui_path = FRONTEND_DIR / "index.html"
    try:
        with open(ui_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse("<h1>Error: index.html not found</h1>", status_code=404)


# Serve static files (CSS, JS)
@app.get("/css/{filename}")
async def serve_css(filename: str):
    """Serve CSS files"""
    css_path = FRONTEND_DIR / "css" / filename
    if css_path.exists():
        return FileResponse(css_path, media_type="text/css")
    return {"error": "File not found"}, 404


@app.get("/dist/{filename:path}")
async def serve_dist(filename: str):
    """Serve compiled TypeScript/JavaScript files from dist folder"""
    # Ensure filename has .js extension
    if not filename.endswith('.js'):
        filename = filename + '.js'
    
    js_path = FRONTEND_DIR / "dist" / filename
    if js_path.exists() and js_path.is_file():
        return FileResponse(js_path, media_type="application/javascript")
    return {"error": "File not found"}, 404


# Register API routes
app.get("/config")(config.get_config)
app.post("/config")(config.save_config)

app.post("/capture")(screenshot.capture)
app.get("/screenshot")(screenshot.get_screenshot)
app.get("/screenshot/events")(screenshot.screenshot_events)

app.post("/detect_preview")(detection.detect_preview)

app.post("/run_extraction")(extraction.run_extraction)
app.get("/last_extraction")(extraction.get_last_extraction)

