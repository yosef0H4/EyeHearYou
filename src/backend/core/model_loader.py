"""
Logic for loading and running the local H2OVL model.
Implements a singleton pattern to avoid reloading the model on every request.
"""
import torch
import warnings
import tempfile
import os
from pathlib import Path
from transformers import AutoConfig, AutoModel, AutoTokenizer
from PIL import Image

# Suppress warnings
warnings.filterwarnings("ignore", message=".*torch_dtype.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")
warnings.filterwarnings("ignore", message=".*Flash Attention 2 only supports.*")
warnings.filterwarnings("ignore", message=".*Flash Attention is not available.*")
warnings.filterwarnings("ignore", message=".*FlashAttention is not installed.*")


class H2OVLModel:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if self._model is not None:
            return
            
        self.model_path = 'h2oai/h2ovl-mississippi-800m'
        print(f"[Model] Initializing H2OVL from {self.model_path}...")
        self._load_model()

    def _load_model(self):
        try:
            # Check Flash Attention
            use_flash_attention = False
            try:
                import flash_attn
                use_flash_attention = True
                print("[Model] Flash Attention 2 available")
            except (ImportError, OSError):
                print("[Model] Flash Attention 2 not available, using SDPA fallback")

            # Load Config
            config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)
            if use_flash_attention:
                config.llm_config._attn_implementation = 'flash_attention_2'
            else:
                config.llm_config._attn_implementation = 'sdpa'

            # Load Model
            self._model = AutoModel.from_pretrained(
                self.model_path,
                dtype=torch.bfloat16,
                config=config,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            ).eval()

            # Device Selection
            if torch.cuda.is_available():
                self._device = 'cuda'
                self._model = self._model.cuda()
                print(f"[Model] Loaded on GPU: {torch.cuda.get_device_name(0)}")
            else:
                self._device = 'cpu'
                print("[Model] Loaded on CPU (Warning: Slow)")

            # Load Tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True, 
                use_fast=False
            )
            
            print("[Model] Model loaded successfully!")
            
        except Exception as e:
            print(f"[Model] Critical Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def test_text_only(self, prompt: str = "Hello") -> str:
        """Test the model with a simple text-only prompt (no image)
        
        Args:
            prompt: Text prompt to test with
            
        Returns:
            Model response string
        """
        if self._model is None:
            self._load_model()
        
        generation_config = dict(max_new_tokens=128, do_sample=False)
        
        try:
            with torch.autocast(device_type=self._device, dtype=torch.bfloat16):
                # Pass None for image_files to do text-only inference
                response, _ = self._model.chat(
                    self._tokenizer,
                    None,  # No image
                    prompt,
                    generation_config,
                    history=None,
                    return_history=True
                )
            return response
        except Exception as e:
            print(f"[Model] Text-only test error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def predict(self, image: Image.Image) -> str:
        """Run inference on a single image
        
        Args:
            image: PIL Image object to process
            
        Returns:
            Extracted text string
        """
        if self._model is None:
            self._load_model()

        # Input Prompt
        question = '<image>\nExtract all text from this image. Return only the extracted text, no additional commentary.'
        
        # Configuration - use greedy decoding for OCR accuracy
        generation_config = dict(max_new_tokens=2048, do_sample=False)

        # H2OVL model expects a file path string, not a PIL Image
        # Save PIL Image to temporary file
        temp_file = None
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.png')
            temp_file = temp_path
            os.close(temp_fd)  # Close file descriptor, we'll use the path
            
            # Save PIL Image to temp file
            image.save(temp_path, format='PNG')
            
            # Autocast for performance
            with torch.autocast(device_type=self._device, dtype=torch.bfloat16):
                response, _ = self._model.chat(
                    self._tokenizer,
                    temp_path,  # Pass file path string
                    question,
                    generation_config,
                    history=None,
                    return_history=True
                )
            return response
        except Exception as e:
            print(f"[Model] Inference error: {e}")
            import traceback
            traceback.print_exc()
            return ""
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    print(f"[Model] Warning: Could not delete temp file {temp_file}: {e}")


# Global accessor
def get_model():
    """Get the singleton H2OVL model instance"""
    return H2OVLModel.get_instance()


def preload_model(test=True):
    """
    Preload the model at startup and optionally test it with a simple text prompt.
    
    Args:
        test: If True, run a simple text-only test inference to verify the model works
        
    Returns:
        True if model loaded (and tested) successfully, False otherwise
    """
    try:
        print("[Model] Preloading H2OVL model...")
        model = get_model()
        
        if test:
            print("[Model] Running startup test...")
            # Run a simple text-only inference to verify the model works
            result = model.test_text_only("Hello")
            if result is not None and len(result.strip()) > 0:
                print("[Model] ✓ Model loaded and tested successfully!")
                return True
            else:
                print("[Model] ⚠ Model loaded but test returned empty result")
                return True  # Still return True, model is loaded
        else:
            print("[Model] ✓ Model loaded successfully!")
            return True
            
    except Exception as e:
        print(f"[Model] ✗ Error preloading model: {e}")
        import traceback
        traceback.print_exc()
        return False

