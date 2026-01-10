import torch
from transformers import AutoModel, AutoTokenizer
import os

class DeepSeekOCR:
    """Singleton wrapper for the DeepSeek OCR model."""
    
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            # We disable quantization because you have 24GB VRAM (Model takes ~15GB)
            cls._instance = DeepSeekOCR(load_in_4bit=False)
        return cls._instance

    def __init__(self, model_name="deepseek-ai/DeepSeek-OCR", load_in_4bit=False, offload_to_cpu=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading DeepSeek-OCR ({self.device})...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Strategy: Run in native BFloat16. 
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        # Force model to GPU 0 relative to visible devices
        device_map = {"": 0} if self.device == "cuda" else None

        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            # No quantization_config needed -> Runs pure BFloat16
            torch_dtype=dtype,
            device_map=device_map
        )
        
        self.model.eval()
        print(f"DeepSeek-OCR Loaded in {dtype}.")

    def process_image(self, image_path: str, prompt="<image>\n<|grounding|>Convert to markdown.") -> str:
        if not os.path.exists(image_path):
            return f"Error: File {image_path} not found."

        debug_dir = "data/ocr_debug"
        os.makedirs(debug_dir, exist_ok=True)

        try:
            with torch.no_grad():
                result = self.model.infer(
                    self.tokenizer,
                    prompt=prompt,
                    image_file=image_path,
                    output_path=debug_dir,
                    base_size=1024,
                    image_size=640,
                    crop_mode=True,
                    save_results=False,
                    eval_mode=True
                )
            return result
        except Exception as e:
            return f"OCR Failed: {str(e)}"