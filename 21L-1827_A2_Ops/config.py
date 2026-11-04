import os                                                
from dotenv import load_dotenv                           

load_dotenv()                                            

MODEL_DIR = os.environ.get("VISIONGUARD_MODEL_DIR", "serving_model")
MODEL_NAME = os.environ.get("VISIONGUARD_MODEL_NAME", "VisionGuard-classifier")
THRESHOLD = float(os.environ.get("VISIONGUARD_THRESHOLD", "0.5"))             
DATABASE_URL = os.environ.get("VISIONGUARD_DATABASE_URL", "sqlite:///visionguard.db")  
MAX_UPLOAD_MB = int(os.environ.get("VISIONGUARD_MAX_UPLOAD_MB", "5"))         
HOST = os.environ.get("VISIONGUARD_HOST", "127.0.0.1")                        
PORT = int(os.environ.get("VISIONGUARD_PORT", "5000"))                        
DEBUG = os.environ.get("VISIONGUARD_DEBUG", "0") == "1"                       