"""
Qwen 2.5-VL REST API Service

Provides a FastAPI-based REST service for Qwen 2.5-VL inference.
"""

from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import asyncio
import logging
from pathlib import Path
import yaml
import time
import base64
from io import BytesIO
from PIL import Image

from .model_manager import QwenVLModelManager

logger = logging.getLogger(__name__)

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: List[Dict[str, Any]]

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    max_new_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False

class ChatResponse(BaseModel):
    response: str
    usage: Dict[str, int]
    model: str
    created: int

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]

class QwenVLService:
    """FastAPI service for Qwen 2.5-VL inference."""
    
    def __init__(self, config_path: str = "config/service_config.yaml"):
        """Initialize the service."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.model_manager = QwenVLModelManager()
        self.app = self._create_app()
        
    def _load_config(self) -> Dict:
        """Load service configuration."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_app(self) -> FastAPI:
        """Create and configure FastAPI application."""
        app = FastAPI(
            title="Qwen 2.5-VL Service",
            description="REST API for Qwen 2.5-VL Vision-Language Model",
            version="1.0.0"
        )
        
        # Add CORS middleware
        if self.config['api']['cors_enabled']:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Setup routes
        self._setup_routes(app)
        
        return app
    
    def _setup_routes(self, app: FastAPI) -> None:
        """Setup API routes."""
        
        security = HTTPBearer() if self.config['authentication']['api_key_required'] else None
        
        def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
            """Verify API key if authentication is enabled."""
            if self.config['authentication']['api_key_required']:
                # In production, verify against actual API keys
                if credentials.credentials != "your-api-key-here":
                    raise HTTPException(status_code=401, detail="Invalid API key")
            return True
        
        @app.post("/v1/chat/completions", response_model=ChatResponse)
        async def chat_completions(
            request: ChatRequest,
            authenticated: bool = Depends(verify_api_key)
        ):
            """Handle chat completion requests."""
            try:
                start_time = time.time()
                
                # Convert request to internal format
                messages = [msg.dict() for msg in request.messages]
                
                # Generate response
                response_text = self.model_manager.generate(
                    messages=messages,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p
                )
                
                # Calculate usage (simplified)
                usage = {
                    "prompt_tokens": 100,  # Would need actual tokenization
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": 100 + len(response_text.split())
                }
                
                return ChatResponse(
                    response=response_text,
                    usage=usage,
                    model="qwen-2.5-vl",
                    created=int(time.time())
                )
                
            except Exception as e:
                logger.error(f"Error in chat completion: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/v1/analyze/image")
        async def analyze_image(
            file: UploadFile = File(...),
            prompt: str = "Describe this image in detail.",
            authenticated: bool = Depends(verify_api_key)
        ):
            """Analyze uploaded image."""
            try:
                # Read and process image
                image_data = await file.read()
                image = Image.open(BytesIO(image_data))
                
                # Convert to base64 for processing
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                # Create message format
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_base64}"}
                        }
                    ]
                }]
                
                # Generate response
                response_text = self.model_manager.generate(messages)
                
                return {"analysis": response_text}
                
            except Exception as e:
                logger.error(f"Error in image analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            model_info = self.model_manager.get_model_info()
            
            return HealthResponse(
                status="healthy" if self.model_manager.model else "no_model",
                model_loaded=self.model_manager.model is not None,
                model_info=model_info
            )
        
        @app.post("/load_model")
        async def load_model(
            variant: str = "qwen-2.5-vl-7b",
            authenticated: bool = Depends(verify_api_key)
        ):
            """Load a specific model variant."""
            try:
                self.model_manager.load_model(variant)
                return {"status": "success", "model": variant}
            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def run(self, host: str = None, port: int = None):
        """Run the service."""
        import uvicorn
        
        host = host or self.config['api']['host']
        port = port or self.config['api']['port']
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=self.config['api']['max_workers']
        )
