"""
Runtime patch for torchvision compatibility with PyTorch 2.7.0

This patch fixes the RuntimeError: operator torchvision::nms does not exist
by monkey-patching torch.library.register_fake before torchvision imports.

CRITICAL: This must be imported and applied BEFORE importing torch or anything
that imports torch (including kokoro, transformers, etc.)
"""
# Import torch - this is the first torch import in our codebase
import torch

# Track if patch is applied
_patch_applied = False


def apply_torchvision_patch():
    """
    Apply runtime monkey-patch to handle torchvision nms operator registration
    compatibility issue with PyTorch 2.7.0.
    
    This patches torch.library.register_fake to gracefully handle the missing
    torchvision::nms operator when torchvision tries to register it.
    """
    global _patch_applied
    
    if _patch_applied:
        return
    
    try:
        # Store original function
        if not hasattr(torch.library, '_original_register_fake'):
            torch.library._original_register_fake = torch.library.register_fake
        
        original_register_fake = torch.library._original_register_fake
        
        def patched_register_fake(name, func=None):
            """
            Wrapped register_fake that catches RuntimeError for torchvision::nms.
            
            The decorator @torch.library.register_fake("torchvision::nms") calls
            this with name="torchvision::nms" and func=None, expecting a decorator back.
            """
            if name == "torchvision::nms":
                # For nms operator, always return a no-op decorator to avoid the error
                if func is None:
                    # Called as decorator: @register_fake("name")
                    def noop_decorator(f):
                        return f
                    return noop_decorator
                else:
                    # Called directly: register_fake("name", func)
                    return func
            else:
                # Not the nms operator - use original (may still raise, but that's OK)
                try:
                    return original_register_fake(name, func)
                except RuntimeError as e:
                    # Re-raise for non-nms operators
                    raise
        
        # Apply the monkey-patch
        torch.library.register_fake = patched_register_fake
        _patch_applied = True
        
    except Exception as e:
        print(f"[Torchvision Patch] Warning: Could not apply patch: {e}")
        import traceback
        traceback.print_exc()
        print("[Torchvision Patch] Continuing without patch - torchvision may fail to import")


# Auto-apply patch on import
apply_torchvision_patch()
