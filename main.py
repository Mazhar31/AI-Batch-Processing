from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
import socketio
import asyncio
import json
import csv
import io
import time
import zipfile
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
import aiofiles
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

# Configuration constants
DEFAULT_RATE_LIMIT = 10
DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.7
DEFAULT_RETRY_ATTEMPTS = 3
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
SUPPORTED_EXTENSIONS = ['.csv', '.json', '.txt']
RATE_LIMIT_RANGE = (1, 60)
TOKEN_RANGE = (1, 4000)
TEMPERATURE_RANGE = (0.0, 2.0)

# Pydantic models
class AIConfig(BaseModel):
    service: str
    model: str
    api_key: str
    params: Dict[str, Any]
    rate_limit: int = DEFAULT_RATE_LIMIT

class MappingConfig(BaseModel):
    group_by: Optional[str] = None
    main_content: str

class PromptTemplate(BaseModel):
    system: Optional[str] = None
    main: str

class OutputConfig(BaseModel):
    format: str
    directory: str = "./outputs/"
    include_prompt: bool = False
    timestamp_files: bool = True
    
    class Config:
        validate_assignment = True

class ProcessingConfig(BaseModel):
    data_file: str
    ai_config: AIConfig
    mapping: MappingConfig
    prompt_template: PromptTemplate
    output: OutputConfig
    advanced_config: Optional[Dict[str, Any]] = {}

# Global state
uploaded_files: Dict[str, Dict] = {}
processing_jobs: Dict[str, Dict] = {}
results_storage: Dict[str, List] = {}

# Initialize FastAPI and SocketIO
app = FastAPI()
sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
socket_app = socketio.ASGIApp(sio, app)

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the HTML frontend"""
    with open("flask_template.html", "r") as f:
        return HTMLResponse(f.read())

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Handle file upload and parsing"""
    # Validate file extension
    file_ext = '.' + file.filename.split('.')[-1].lower() if '.' in file.filename else ''
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
    
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    try:
        if file.filename.endswith('.csv'):
            data = parse_csv(content)
        elif file.filename.endswith('.json'):
            data = parse_json(content)
        else:  # .txt
            data = parse_txt(content)
        
        file_info = {
            "filename": file.filename,
            "rows": len(data["rows"]),
            "columns": data["columns"],
            "data": data["rows"]
        }
        
        uploaded_files[file.filename] = file_info
        return file_info
        
    except Exception as e:
        raise HTTPException(400, f"Error parsing file: {str(e)}")

def parse_csv(content: bytes) -> Dict:
    """Parse CSV content"""
    text = content.decode('utf-8')
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    columns = list(rows[0].keys()) if rows else []
    return {"columns": columns, "rows": rows}

def parse_json(content: bytes) -> Dict:
    """Parse JSON content"""
    data = json.loads(content.decode('utf-8'))
    if isinstance(data, list) and data:
        columns = list(data[0].keys())
        return {"columns": columns, "rows": data}
    raise ValueError("JSON must be an array of objects")

def parse_txt(content: bytes) -> Dict:
    """Parse TXT content"""
    lines = content.decode('utf-8').strip().split('\n')
    rows = [{"content": line} for line in lines if line.strip()]
    return {"columns": ["content"], "rows": rows}

@app.post("/start_processing")
async def start_processing(config: ProcessingConfig, background_tasks: BackgroundTasks):
    """Start batch processing"""
    if config.data_file not in uploaded_files:
        raise HTTPException(400, "File not found")
    
    job_id = f"job_{int(time.time())}"
    processing_jobs[job_id] = {
        "status": "running",
        "config": config,
        "start_time": time.time(),
        "current": 0,
        "total": uploaded_files[config.data_file]["rows"],
        "completed": 0,
        "errors": 0,
        "paused": False
    }
    
    background_tasks.add_task(process_batch, job_id)
    return {"job_id": job_id, "status": "started"}

async def process_batch(job_id: str):
    """Main batch processing function"""
    job = processing_jobs[job_id]
    config = job["config"]
    file_data = uploaded_files[config.data_file]
    
    # Initialize AI client
    if config.ai_config.service == "openai":
        client = AsyncOpenAI(api_key=config.ai_config.api_key)
    else:
        client = AsyncAnthropic(api_key=config.ai_config.api_key)
    
    # Group data if needed
    if config.mapping.group_by and config.mapping.group_by != "None":
        groups = group_data(file_data["data"], config.mapping.group_by)
    else:
        groups = {f"row_{i}": [row] for i, row in enumerate(file_data["data"])}
    
    results = []
    conversation_history = {}
    rate_limiter = RateLimiter(config.ai_config.rate_limit)
    
    # Log configuration if enabled
    if config.advanced_config.get("log_requests", False):
        print(f"Starting batch processing with config: {config.ai_config.model}, rate_limit: {config.ai_config.rate_limit}")
    
    for group_name, group_rows in groups.items():
        if job["status"] == "stopped":
            break
            
        while job["paused"]:
            await asyncio.sleep(1)
        
        try:
            # Process group
            for row in group_rows:
                if job["status"] == "stopped":
                    break
                    
                while job["paused"]:
                    await asyncio.sleep(1)
                
                try:
                    await rate_limiter.wait()
                    
                    # Build prompt
                    prompt = build_prompt(config.prompt_template, row)
                    
                    # Get or create conversation history
                    if group_name not in conversation_history:
                        conversation_history[group_name] = []
                        if config.prompt_template.system:
                            conversation_history[group_name].append({
                                "role": "system", 
                                "content": config.prompt_template.system
                            })
                    
                    conversation_history[group_name].append({
                        "role": "user", 
                        "content": prompt
                    })
                    
                    # Log request if enabled
                    if config.advanced_config.get("log_requests", False):
                        print(f"API Request for group {group_name}: {prompt[:100]}...")
                    
                    # Call AI API with retry logic
                    response = await call_ai_api(
                        client, 
                        config.ai_config, 
                        conversation_history[group_name]
                    )
                    
                    conversation_history[group_name].append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                    # Store result
                    result = {
                        "group": group_name,
                        "input": row,
                        "prompt": prompt,
                        "response": response,
                        "timestamp": datetime.now().isoformat()
                    }
                    results.append(result)
                    
                    job["current"] += 1
                    job["completed"] += 1
                    
                    # Emit progress
                    await sio.emit("progress_update", {
                        "current": job["current"],
                        "total": job["total"],
                        "group": group_name
                    })
                    
                    await sio.emit("item_completed", {
                        "index": job["current"] - 1,
                        "group": group_name,
                        "conversation_length": len(conversation_history.get(group_name, []))
                    })
                    
                except Exception as e:
                    job["errors"] += 1
                    job["current"] += 1
                    error_msg = str(e)
                    
                    # Log error if enabled
                    if config.advanced_config.get("log_requests", False):
                        print(f"API Error for group {group_name}: {error_msg}")
                    
                    await sio.emit("item_error", {
                        "index": job["current"] - 1,
                        "error": error_msg
                    })
                
        except Exception as e:
            # Handle group-level errors
            print(f"Group processing error: {str(e)}")
            job["errors"] += len(group_rows)
            await sio.emit("item_error", {
                "index": job["current"],
                "error": f"Group error: {str(e)}"
            })
    
    # Store results
    results_storage[job_id] = results
    job["status"] = "completed"
    
    await sio.emit("batch_completed", {
        "total_processed": job["completed"],
        "total_errors": job["errors"]
    })

def group_data(data: List[Dict], group_by: str) -> Dict[str, List[Dict]]:
    """Group data by specified column"""
    groups = {}
    for row in data:
        key = str(row.get(group_by, "unknown"))
        if key not in groups:
            groups[key] = []
        groups[key].append(row)
    return groups

def build_prompt(template: PromptTemplate, row: Dict[str, Any]) -> str:
    """Build prompt from template and row data"""
    prompt = template.main
    for key, value in row.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    return prompt

async def call_ai_api(client, config: AIConfig, messages: List[Dict], retry_count: int = 0) -> str:
    """Call AI API (OpenAI or Anthropic) with retry logic"""
    max_retries = config.params.get("retry_attempts", DEFAULT_RETRY_ATTEMPTS)
    
    try:
        if config.service == "openai":
            response = await client.chat.completions.create(
                model=config.model,
                messages=messages,
                temperature=config.params.get("temperature", DEFAULT_TEMPERATURE),
                max_tokens=config.params.get("max_tokens", DEFAULT_MAX_TOKENS)
            )
            return response.choices[0].message.content
        else:  # Anthropic
            # Convert messages for Anthropic format
            system_msg = None
            user_messages = []
            
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)
            
            kwargs = {
                "model": config.model,
                "messages": user_messages,
                "temperature": config.params.get("temperature", DEFAULT_TEMPERATURE),
                "max_tokens": config.params.get("max_tokens", DEFAULT_MAX_TOKENS)
            }
            
            if system_msg:
                kwargs["system"] = system_msg
            
            response = await client.messages.create(**kwargs)
            return response.content[0].text
            
    except Exception as e:
        error_msg = str(e).lower()
        
        # Handle specific API errors
        if "rate limit" in error_msg or "429" in error_msg:
            if retry_count < max_retries:
                wait_time = (2 ** retry_count) * 2  # Exponential backoff
                await sio.emit("rate_limit_wait", {"wait_time": wait_time})
                await asyncio.sleep(wait_time)
                return await call_ai_api(client, config, messages, retry_count + 1)
            else:
                raise Exception(f"Rate limit exceeded after {max_retries} retries")
        
        elif "invalid" in error_msg and "key" in error_msg:
            raise Exception("Invalid API key. Please check your credentials.")
        
        elif "quota" in error_msg or "billing" in error_msg or "credits" in error_msg:
            if config.service == "anthropic":
                raise Exception("Anthropic API credits exhausted. Free tier has limited credits. Please check your usage or upgrade your plan.")
            else:
                raise Exception("API quota exceeded or billing issue. Please check your account.")
        
        elif "model" in error_msg and "not found" in error_msg:
            if config.service == "anthropic":
                raise Exception(f"Model '{config.model}' not available. Free tier users should use 'claude-3-haiku-20240307'. Upgrade your plan for access to other models.")
            else:
                raise Exception(f"Model '{config.model}' not available. Please select a different model.")
        
        else:
            if retry_count < max_retries:
                wait_time = (2 ** retry_count) * 1  # Shorter wait for general errors
                await asyncio.sleep(wait_time)
                return await call_ai_api(client, config, messages, retry_count + 1)
            else:
                raise Exception(f"API error after {max_retries} retries: {str(e)}")

class RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.requests_per_minute = max(1, requests_per_minute)
        self.interval = 60.0 / self.requests_per_minute
        self.last_request_times = []
    
    async def wait(self):
        current_time = time.time()
        
        # Remove requests older than 1 minute
        self.last_request_times = [t for t in self.last_request_times if current_time - t < 60]
        
        # If we've hit the rate limit, wait
        if len(self.last_request_times) >= self.requests_per_minute:
            oldest_request = min(self.last_request_times)
            wait_time = 60 - (current_time - oldest_request)
            
            if wait_time > 0:
                await sio.emit("rate_limit_wait", {"wait_time": wait_time})
                await asyncio.sleep(wait_time)
        
        # Record this request
        self.last_request_times.append(time.time())

@app.post("/pause_processing")
async def pause_processing():
    """Pause/resume processing"""
    for job in processing_jobs.values():
        if job["status"] == "running":
            job["paused"] = not job["paused"]
            return {"status": "paused" if job["paused"] else "running"}
    return {"status": "no_active_job"}

@app.post("/stop_processing")
async def stop_processing():
    """Stop processing"""
    for job in processing_jobs.values():
        if job["status"] == "running":
            job["status"] = "stopped"
            return {"status": "stopped"}
    return {"status": "no_active_job"}

@app.get("/get_status")
async def get_status():
    """Get current processing status"""
    for job_id, job in processing_jobs.items():
        if job["status"] == "running":
            elapsed = time.time() - job["start_time"]
            rate = job["completed"] / (elapsed / 60) if elapsed > 0 else 0
            remaining = job["total"] - job["current"]
            eta = (remaining / rate * 60) if rate > 0 else 0
            
            return {
                "job_id": job_id,
                "current": job["current"],
                "total": job["total"],
                "completed": job["completed"],
                "errors": job["errors"],
                "rate": rate,
                "eta": eta,
                "status": job["status"]
            }
    return {"status": "no_active_job"}

@app.post("/reset_system")
async def reset_system():
    """Reset system - clear all data for fresh start"""
    uploaded_files.clear()
    processing_jobs.clear()
    results_storage.clear()
    return {"status": "reset_complete"}

@app.get("/export_results")
async def export_results():
    """Export processing results"""
    # Find the most recent completed job
    latest_job = None
    for job_id, job in processing_jobs.items():
        if job_id in results_storage:
            latest_job = job_id
            break
    
    if not latest_job or latest_job not in results_storage:
        raise HTTPException(404, "No results to export")
    
    results = results_storage[latest_job]
    job = processing_jobs[latest_job]
    config = job["config"]
    
    # Create export based on format
    config = job["config"]
    output_config = config["output"] if isinstance(config, dict) else config.output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if (output_config["timestamp_files"] if isinstance(output_config, dict) else output_config.timestamp_files) else ""
    
    # No need to create directories - streaming directly to user
    
    format_type = output_config["format"] if isinstance(output_config, dict) else output_config.format
    include_prompt = output_config["include_prompt"] if isinstance(output_config, dict) else output_config.include_prompt
    
    if format_type == "json":
        filename = f"results_{timestamp}.json" if timestamp else "results.json"
        filepath = os.path.join(output_dir, filename)
        
        export_data = []
        for result in results:
            item = {
                "group": result["group"],
                "response": result["response"],
                "timestamp": result["timestamp"]
            }
            if include_prompt:
                item["prompt"] = result["prompt"]
                item["input"] = result["input"]
            export_data.append(item)
        
        json_content = json.dumps(export_data, indent=2)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    elif format_type == "csv":
        filename = f"results_{timestamp}.csv" if timestamp else "results.csv"
        filepath = os.path.join(output_dir, filename)
        
        csv_content = io.StringIO()
        if results:
            fieldnames = ["group", "response", "timestamp"]
            if include_prompt:
                fieldnames = ["group", "prompt", "input", "response", "timestamp"]
            
            writer = csv.DictWriter(csv_content, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    "group": result["group"], 
                    "response": result["response"],
                    "timestamp": result["timestamp"]
                }
                if include_prompt:
                    row["prompt"] = result["prompt"]
                    row["input"] = json.dumps(result["input"]) if isinstance(result["input"], dict) else str(result["input"])
                writer.writerow(row)
        
        return Response(
            content=csv_content.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    elif format_type == "individual":
        # Create individual files and zip them
        zip_filename = f"results_{timestamp}.zip" if timestamp else "results.zip"

        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for i, result in enumerate(results):
                file_content = result["response"]
                if include_prompt:
                    input_str = json.dumps(result["input"], indent=2) if isinstance(result["input"], dict) else str(result["input"])
                    file_content = f"INPUT:\n{input_str}\n\nPROMPT:\n{result['prompt']}\n\nRESPONSE:\n{file_content}\n\nTIMESTAMP: {result['timestamp']}"
                
                safe_group = "".join(c for c in str(result['group']) if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"result_{i+1:03d}_{safe_group}_{timestamp}.txt" if timestamp else f"result_{i+1:03d}_{safe_group}.txt"
                zipf.writestr(filename, file_content)
        
        zip_buffer.seek(0)
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )
    
    else:  # both
        zip_filename = f"results_{timestamp}.zip" if timestamp else "results.zip"
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add individual files
            for i, result in enumerate(results):
                file_content = result["response"]
                if include_prompt:
                    input_str = json.dumps(result["input"], indent=2) if isinstance(result["input"], dict) else str(result["input"])
                    file_content = f"INPUT:\n{input_str}\n\nPROMPT:\n{result['prompt']}\n\nRESPONSE:\n{file_content}\n\nTIMESTAMP: {result['timestamp']}"
                
                safe_group = "".join(c for c in str(result['group']) if c.isalnum() or c in (' ', '-', '_')).strip()
                filename = f"individual/result_{i+1:03d}_{safe_group}_{timestamp}.txt" if timestamp else f"individual/result_{i+1:03d}_{safe_group}.txt"
                zipf.writestr(filename, file_content)
            
            # Add consolidated JSON
            export_data = []
            for result in results:
                item = {
                    "group": result["group"],
                    "response": result["response"],
                    "timestamp": result["timestamp"]
                }
                if include_prompt:
                    item["prompt"] = result["prompt"]
                    item["input"] = result["input"]
                export_data.append(item)
            
            json_filename = f"results_{timestamp}.json" if timestamp else "results.json"
            zipf.writestr(f"consolidated/{json_filename}", json.dumps(export_data, indent=2))
            
            # Add consolidated CSV
            csv_content = io.StringIO()
            fieldnames = ["group", "response", "timestamp"]
            if include_prompt:
                fieldnames = ["group", "prompt", "input", "response", "timestamp"]
            
            writer = csv.DictWriter(csv_content, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    "group": result["group"], 
                    "response": result["response"],
                    "timestamp": result["timestamp"]
                }
                if include_prompt:
                    row["prompt"] = result["prompt"]
                    row["input"] = json.dumps(result["input"]) if isinstance(result["input"], dict) else str(result["input"])
                writer.writerow(row)
            
            csv_filename = f"results_{timestamp}.csv" if timestamp else "results.csv"
            zipf.writestr(f"consolidated/{csv_filename}", csv_content.getvalue())
        
        zip_buffer.seek(0)
        return Response(
            content=zip_buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
        )

# SocketIO events
@sio.event
async def connect(sid, environ):
    print(f"Client {sid} connected")

@sio.event
async def disconnect(sid):
    print(f"Client {sid} disconnected")

# Create the final app
final_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:final_app", host="0.0.0.0", port=8000, reload=True)