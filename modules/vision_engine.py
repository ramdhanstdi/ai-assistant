import cv2
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class VisionManager:
    def __init__(self, model_id="vikhyatk/moondream2"):
        # Detect the best available device: DirectML, OpenVINO, or fall back to CPU.
        # Note: True OpenVINO or DirectML support in HF sometimes requires specific Optimum
        # packages or custom integrations. Here we'll configure standard device fallbacks.
        # For optimum-intel/openvino, you'd typically use OVModelForCausalLM.
        # For standard torch, DirectML can be accessed if torch-directml is installed.
        
        self.device = "cpu"
        
        try:
            # Try DirectML if available
            import torch_directml
            if torch_directml.is_available():
                self.device = torch_directml.device()
                print(f"Using DirectML device: {self.device}")
        except ImportError:
            pass

        if self.device == "cpu":
            # Just defaulting to CPU
            print("Using CPU device fallback")

        print(f"Loading vision model {model_id} on {self.device}...")
        
        # Initialize moondream2 model
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision="2024-05-08")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            revision="2024-05-08"
        ).to(self.device)
        # Note: For production use with moondream2, it's highly recommended to use the appropriate revision, 
        # as the model gets updated frequently. Revision may change over time.

    def capture_frame(self, output_path="captured_frame.jpg") -> str:
        """
        Membuka webcam, mengambil 1 gambar, menyimpan ke file, lalu menutup webcam.
        Returns the path to the saved image.
        """
        # Open default camera (index 0)
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            raise Exception("Tidak dapat membuka webcam")
            
        print("📸 Menangkap gambar dari webcam...")
        ret, frame = cap.read()
        
        if ret:
            # Save the captured frame
            cv2.imwrite(output_path, frame)
            print(f"✅ Gambar disimban ke {output_path}")
        else:
            cap.release()
            raise Exception("Gagal membaca frame dari webcam")
            
        # Release camera immediately after capturing 1 frame
        cap.release()
        
        return output_path
        
    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Menganalisis gambar menggunakan model moondream2 berdasarkan prompt yang diberikan.
        """
        print(f"🔍 Menganalisis gambar {image_path}...")
        
        # Load the image using PIL (required by moondream2)
        try:
            image = Image.open(image_path)
        except Exception as e:
            return f"Error membuka gambar: {e}"

        # Get features from the image using the model's specialized method
        enc_image = self.model.encode_image(image)
        
        # Generate the response
        answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
        
        return answer
