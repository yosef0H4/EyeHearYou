# Refactored Project Structure

The project has been refactored into a clean structure with separated backend (Python) and frontend (TypeScript).

## Project Structure

```
src/
  backend/
    __init__.py
    main.py          # Entry point
    app.py           # FastAPI app setup
    state.py         # Application state
    hotkey.py        # Keyboard hotkey management
    routes/
      __init__.py
      config.py      # Configuration routes
      screenshot.py  # Screenshot routes
      detection.py   # Text detection routes
      extraction.py  # Text extraction routes
  frontend/
    index.html       # Main HTML file
    css/
      styles.css     # All CSS styles
    ts/
      main.ts        # Main entry point
      types.ts       # TypeScript type definitions
      config.ts      # Configuration management
      sse.ts         # Server-Sent Events
      detection.ts   # Text detection
      merge.ts       # Merge logic
      ui.ts          # UI rendering
```

## Setup

### Backend

The backend is in Python and uses the existing structure. Run:

```bash
uv run python -m src.backend.main
```

### Frontend

The frontend is in TypeScript. To compile:

1. Install TypeScript (if not already installed):
   ```bash
   npm install
   ```

2. Compile TypeScript to JavaScript:
   ```bash
   npm run build
   ```

   Or watch for changes:
   ```bash
   npm run watch
   ```

3. The compiled JavaScript files will be in `src/frontend/ts/` (same directory as source)

## Development

- Backend: Python modules in `src/backend/`
- Frontend: TypeScript files in `src/frontend/ts/` compile to `.js` files in the same directory
- The server serves static files from `src/frontend/`

## Notes

- TypeScript files use ES modules
- The HTML file loads the compiled JavaScript as modules
- All routes are properly separated by functionality
- State management is centralized in `state.py`

