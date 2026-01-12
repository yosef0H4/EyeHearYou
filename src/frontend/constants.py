"""Constants and optional imports for the GUI"""
# Try to import keyboard for hotkeys
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    keyboard = None



