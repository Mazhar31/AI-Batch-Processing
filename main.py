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
        
        # Clear previous data when new file is uploaded
        uploaded_files.clear()
        processing_jobs.clear()
        results_storage.clear()
        
        uploaded_files[file.filename] = file_info
        return file_info
        
    except Exception as e:
        raise HTTPException(400, f"Error parsing file: {str(e)}")

def parse_csv(content: bytes) -> Dict:
    """Parse CSV content with enhanced error handling"""
    try:
        text = content.decode('utf-8')
        
        # Check if file is empty
        if not text.strip():
            raise ValueError("CSV file is empty")
        
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        
        if not rows:
            raise ValueError("CSV file contains no data rows. Ensure it has a header row and at least one data row.")
        
        columns = list(rows[0].keys())
        
        # Validate columns
        if not columns or columns == [None] or '' in columns:
            raise ValueError("CSV file has invalid or missing column headers. Ensure the first row contains column names.")
        
        # Check for empty rows
        valid_rows = []
        for i, row in enumerate(rows, 2):
            if any(value.strip() for value in row.values()):
                valid_rows.append(row)
            elif all(not value.strip() for value in row.values()):
                continue  # Skip completely empty rows
        
        if not valid_rows:
            raise ValueError("CSV file contains no valid data rows")
        
        return {"columns": columns, "rows": valid_rows}
        
    except UnicodeDecodeError:
        raise ValueError("CSV file contains invalid characters. Please ensure it's saved as UTF-8.")
    except csv.Error as e:
        raise ValueError(f"Invalid CSV format: {str(e)}")
    except Exception as e:
        if "CSV file" in str(e):
            raise
        raise ValueError(f"Error parsing CSV file: {str(e)}")

def parse_json(content: bytes) -> Dict:
    """Parse JSON content with enhanced error handling"""
    try:
        text = content.decode('utf-8').strip()
        
        if not text:
            raise ValueError("JSON file is empty")
        
        data = json.loads(text)
        
        if not isinstance(data, list):
            raise ValueError("JSON must be an array of objects. Example: [{\"column1\": \"value1\", \"column2\": \"value2\"}]")
        
        if not data:
            raise ValueError("JSON array is empty. Please include at least one object.")
        
        # Validate all items are objects
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i+1} in JSON array is not an object. All items must be objects with key-value pairs.")
        
        # Get columns from first object
        columns = list(data[0].keys())
        
        if not columns:
            raise ValueError("JSON objects have no properties. Each object must have at least one key-value pair.")
        
        # Validate all objects have consistent structure
        for i, item in enumerate(data[1:], 2):
            item_keys = set(item.keys())
            expected_keys = set(columns)
            
            if item_keys != expected_keys:
                missing = expected_keys - item_keys
                extra = item_keys - expected_keys
                error_msg = f"Object {i} has inconsistent structure."
                if missing:
                    error_msg += f" Missing keys: {list(missing)}."
                if extra:
                    error_msg += f" Extra keys: {list(extra)}."
                raise ValueError(error_msg)
        
        return {"columns": columns, "rows": data}
        
    except UnicodeDecodeError:
        raise ValueError("JSON file contains invalid characters. Please ensure it's saved as UTF-8.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {str(e)}. Please check your JSON syntax.")
    except Exception as e:
        if "JSON" in str(e):
            raise
        raise ValueError(f"Error parsing JSON file: {str(e)}")

def parse_txt(content: bytes) -> Dict:
    """Parse TXT content - supports both structured and simple formats"""
    try:
        text = content.decode('utf-8').strip()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            raise ValueError("TXT file is empty")
        
        # Check if first line looks like headers (contains delimiters)
        first_line = lines[0]
        
        # Try to detect structured format (comma is primary, then pipe, then tab)
        if ',' in first_line and len(lines) > 1:
            # Check if it looks like CSV headers (not natural language)
            words = first_line.split(',')
            if len(words) >= 2 and all(len(word.strip()) < 30 and not ' ' in word.strip() for word in words[:3]):
                # Comma-separated format: column1,column2,column3
                return parse_txt_structured(lines, ',')
        
        if '|' in first_line and len(lines) > 1:
            # Pipe-separated format: column1|column2|column3
            return parse_txt_structured(lines, '|')
        elif '\t' in first_line and len(lines) > 1:
            # Tab-separated format
            return parse_txt_structured(lines, '\t')
        else:
            # Simple format: one item per line
            if len(lines) < 1:
                raise ValueError("TXT file must contain at least one line of content")
            
            rows = [{"content": line} for line in lines]
            return {"columns": ["content"], "rows": rows}
            
    except UnicodeDecodeError:
        raise ValueError("TXT file contains invalid characters. Please ensure it's saved as UTF-8 text.")
    except Exception as e:
        raise ValueError(f"Error parsing TXT file: {str(e)}")

def parse_txt_structured(lines: list, delimiter: str) -> Dict:
    """Parse structured TXT with delimiters"""
    try:
        # First line as headers
        headers = [col.strip() for col in lines[0].split(delimiter)]
        
        if len(headers) < 1:
            raise ValueError(f"Structured TXT must have at least 1 column separated by '{delimiter}'")
        
        # Clean headers
        headers = [header.strip() for header in headers if header.strip()]
        
        # Validate headers
        for header in headers:
            if not header or not header.replace('_', '').isalnum():
                raise ValueError(f"Invalid column name '{header}'. Use alphanumeric characters and underscores only.")
        
        rows = []
        for i, line in enumerate(lines[1:], 2):
            values = [val.strip() for val in line.split(delimiter)]
            
            if len(values) != len(headers):
                raise ValueError(f"Line {i} has {len(values)} values but expected {len(headers)} columns")
            
            row = dict(zip(headers, values))
            rows.append(row)
        
        if not rows:
            raise ValueError("No data rows found after header line")
        
        return {"columns": headers, "rows": rows}
        
    except Exception as e:
        raise ValueError(f"Error parsing structured TXT: {str(e)}")

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
    

    
    # Create semaphore for concurrent processing - use user's rate limit setting
    max_concurrent = min(config.ai_config.rate_limit, 10)  # Limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Log configuration if enabled
    if config.advanced_config.get("log_requests", False):
        print(f"Starting batch processing: {len(groups)} groups, max concurrent: {max_concurrent}")
    
    # Process groups in parallel while maintaining conversation context
    group_tasks = []
    for group_name, group_rows in groups.items():
        task = process_group_batch(semaphore, group_name, group_rows, conversation_history, 
                                 client, config, job, rate_limiter, results)
        group_tasks.append(task)
    
    # Wait for all groups to complete
    await asyncio.gather(*group_tasks, return_exceptions=True)
    
    # Store results
    results_storage[job_id] = results
    job["status"] = "completed"
    
    await sio.emit("batch_completed", {
        "total_processed": job["completed"],
        "total_errors": job["errors"]
    })

def group_data(data: List[Dict], group_by: str) -> Dict[str, List[Dict]]:
    """Group data by specified column with row index tracking"""
    groups = {}
    for row_index, row in enumerate(data):
        key = str(row.get(group_by, "unknown"))
        if key not in groups:
            groups[key] = []
        # Add row index for maintaining original order
        row_with_index = row.copy()
        row_with_index['_row_index'] = row_index
        groups[key].append(row_with_index)
    return groups

def build_prompt(template: PromptTemplate, row: Dict[str, Any]) -> str:
    """Build prompt from template and row data"""
    prompt = template.main
    for key, value in row.items():
        prompt = prompt.replace(f"{{{key}}}", str(value))
    return prompt

# Thread-safe locks for shared resources
import threading
_job_lock = threading.Lock()
_results_lock = threading.Lock()
_conversation_lock = threading.Lock()

async def process_single_item(semaphore, group_name, row, conversation_history, 
                               client, config, job, rate_limiter, results, row_index):
    """Process a single item with conversation context - THREAD SAFE"""
    if job["status"] == "stopped":
        return
        
    while job["paused"]:
        await asyncio.sleep(1)
    
    async with semaphore:  # Limit concurrent requests
        try:
            await rate_limiter.wait()
            
            # Build prompt
            prompt = build_prompt(config.prompt_template, row)
            
            # Handle conversation context with thread safety
            with _conversation_lock:
                is_new_conversation = group_name not in conversation_history
                if is_new_conversation:
                    conversation_history[group_name] = []
                    if config.prompt_template.system:
                        conversation_history[group_name].append({
                            "role": "system", 
                            "content": config.prompt_template.system
                        })
            
            # For grouped conversations, we need to maintain order
            if config.mapping.group_by and config.mapping.group_by != "None":
                # Thread-safe conversation handling
                with _conversation_lock:
                    conversation_history[group_name].append({
                        "role": "user", 
                        "content": prompt
                    })
                    # Make a copy for API call to avoid race conditions
                    messages_copy = conversation_history[group_name].copy()
                
                # Call AI API with conversation history copy
                response = await call_ai_api(client, config.ai_config, messages_copy)
                
                # Thread-safe update of conversation history
                with _conversation_lock:
                    conversation_history[group_name].append({
                        "role": "assistant", 
                        "content": response
                    })
            else:
                # Independent processing - no conversation context needed
                messages = []
                if config.prompt_template.system:
                    messages.append({"role": "system", "content": config.prompt_template.system})
                messages.append({"role": "user", "content": prompt})
                
                response = await call_ai_api(client, config.ai_config, messages)
            
            # Log request if enabled
            if config.advanced_config.get("log_requests", False):
                conversation_type = "New conversation" if is_new_conversation else "Continuing conversation"
                print(f"[{datetime.now().strftime('%H:%M:%S')}] API Request for group {group_name} ({conversation_type}): {prompt[:100]}...")
            
            # Thread-safe result storage with all original fields preserved
            main_content_value = row.get(config.mapping.main_content, "") if config.mapping.main_content else ""
            
            # Create result with all original file columns preserved
            result = {
                "group": group_name,
                "main_content": main_content_value,
                "response": response,
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt,
                "row_index": row.get('_row_index', 0)
            }
            
            # Add all original file columns (excluding internal _row_index)
            for key, value in row.items():
                if key != '_row_index' and key not in result:
                    result[key] = value
            
            with _results_lock:
                results.append(result)
            
            # Thread-safe job counter updates
            with _job_lock:
                job["current"] += 1
                job["completed"] += 1
                current_count = job["current"]
                total_count = job["total"]
            
            # Emit progress
            await sio.emit("progress_update", {
                "current": current_count,
                "total": total_count,
                "group": group_name
            })
            
            await sio.emit("item_completed", {
                "index": current_count - 1,
                "group": group_name,
                "conversation_length": len(conversation_history.get(group_name, []))
            })
            
        except Exception as e:
            with _job_lock:
                job["errors"] += 1
                job["current"] += 1
                current_count = job["current"]
            
            error_msg = str(e)
            
            # Log error if enabled
            if config.advanced_config.get("log_requests", False):
                print(f"[{datetime.now().strftime('%H:%M:%S')}] API Error for group {group_name}: {error_msg}")
            
            await sio.emit("item_error", {
                "index": current_count - 1,
                "error": error_msg
            })

async def process_group_batch(semaphore, group_name, group_rows, conversation_history, 
                            client, config, job, rate_limiter, results):
    """Process a group of rows in TRUE batch (parallel) while handling conversation context"""
    try:
        if config.mapping.group_by and config.mapping.group_by != "None":
            # For grouped conversations, process sequentially to maintain context
            for i, row in enumerate(group_rows):
                await process_single_item(semaphore, group_name, row, conversation_history, 
                                         client, config, job, rate_limiter, results, i)
        else:
            # For independent processing, TRUE parallel batch processing
            tasks = []
            for i, row in enumerate(group_rows):
                task = process_single_item(semaphore, group_name, row, conversation_history, 
                                         client, config, job, rate_limiter, results, i)
                tasks.append(task)
            
            # Process all items in parallel
            await asyncio.gather(*tasks, return_exceptions=True)
                
    except Exception as e:
        # Handle group-level errors
        print(f"Group processing error for {group_name}: {str(e)}")
        job["errors"] += len(group_rows)
        await sio.emit("item_error", {
            "index": job["current"],
            "error": f"Group {group_name} error: {str(e)}"
        })

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
        self.requests_per_minute = max(1, min(requests_per_minute, 60))  # Enforce limits
        self.interval = 60.0 / self.requests_per_minute
        self.last_request_times = []
        print(f"Rate limiter initialized: {self.requests_per_minute} requests/minute (interval: {self.interval:.2f}s)")
    
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
async def export_results(format_type: str, include_prompt: bool = False, timestamp_files: bool = True):
    """Export processing results in specified format"""
    # Find the most recent completed job
    latest_job = None
    for job_id, job in processing_jobs.items():
        if job_id in results_storage and job.get("status") == "completed":
            latest_job = job_id
            break
    
    if not latest_job or latest_job not in results_storage:
        raise HTTPException(404, "No results to export")
    
    results = results_storage[latest_job]
    
    # Sort results by original row order
    results.sort(key=lambda x: x.get('row_index', 0))
    
    # Generate timestamp if requested
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if timestamp_files else ""
    
    # Get original file columns for consistent output structure
    original_columns = set()
    for result in results:
        for key in result.keys():
            if key not in ['group', 'main_content', 'response', 'timestamp', 'prompt', 'row_index']:
                original_columns.add(key)
    original_columns = sorted(list(original_columns))
    
    if format_type == "json":
        filename = f"results_{timestamp}.json" if timestamp else "results.json"
        
        export_data = []
        for result in results:
            # Build consistent output structure
            item = {
                "group": result.get("group", ""),
                "main_content": result.get("main_content", "")
            }
            
            # Add all original file columns in consistent order
            for col in original_columns:
                item[col] = result.get(col, "")
            
            # Add prompt if requested
            if include_prompt:
                item["prompt"] = result.get("prompt", "")
            
            # Add response and timestamp last
            item["response"] = result.get("response", "")
            item["timestamp"] = result.get("timestamp", "")
            
            export_data.append(item)
        
        json_content = json.dumps(export_data, indent=2)
        return Response(
            content=json_content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    
    elif format_type == "csv":
        filename = f"results_{timestamp}.csv" if timestamp else "results.csv"
        
        csv_content = io.StringIO()
        if results:
            # Build consistent fieldnames in proper order
            fieldnames = ["group", "main_content"] + original_columns
            if include_prompt:
                fieldnames.append("prompt")
            fieldnames.extend(["response", "timestamp"])
            
            writer = csv.DictWriter(csv_content, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    "group": result.get("group", ""),
                    "main_content": result.get("main_content", "")
                }
                
                # Add all original file columns
                for col in original_columns:
                    row[col] = result.get(col, "")
                
                # Add prompt if requested
                if include_prompt:
                    row["prompt"] = result.get("prompt", "")
                
                # Add response and timestamp
                row["response"] = result.get("response", "")
                row["timestamp"] = result.get("timestamp", "")
                
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
                file_content = result.get("response", "")
                
                if include_prompt:
                    # Build structured input data with all original columns
                    input_sections = []
                    input_sections.append(f"GROUP: {result.get('group', '')}")
                    input_sections.append(f"MAIN_CONTENT: {result.get('main_content', '')}")
                    
                    # Add all original file columns
                    for col in original_columns:
                        input_sections.append(f"{col.upper()}: {result.get(col, '')}")
                    
                    input_str = "\n".join(input_sections)
                    prompt_text = result.get("prompt", "")
                    timestamp_text = result.get("timestamp", "")
                    file_content = f"INPUT:\n{input_str}\n\nPROMPT:\n{prompt_text}\n\nRESPONSE:\n{file_content}\n\nTIMESTAMP: {timestamp_text}"
                
                group_name = result.get("group", "unknown")
                safe_group = "".join(c for c in str(group_name) if c.isalnum() or c in (' ', '-', '_')).strip()
                if not safe_group:
                    safe_group = "unknown"
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
                file_content = result.get("response", "")
                
                if include_prompt:
                    # Build structured input data with all original columns
                    input_sections = []
                    input_sections.append(f"GROUP: {result.get('group', '')}")
                    input_sections.append(f"MAIN_CONTENT: {result.get('main_content', '')}")
                    
                    # Add all original file columns
                    for col in original_columns:
                        input_sections.append(f"{col.upper()}: {result.get(col, '')}")
                    
                    input_str = "\n".join(input_sections)
                    prompt_text = result.get("prompt", "")
                    timestamp_text = result.get("timestamp", "")
                    file_content = f"INPUT:\n{input_str}\n\nPROMPT:\n{prompt_text}\n\nRESPONSE:\n{file_content}\n\nTIMESTAMP: {timestamp_text}"
                
                group_name = result.get("group", "unknown")
                safe_group = "".join(c for c in str(group_name) if c.isalnum() or c in (' ', '-', '_')).strip()
                if not safe_group:
                    safe_group = "unknown"
                filename = f"individual/result_{i+1:03d}_{safe_group}_{timestamp}.txt" if timestamp else f"individual/result_{i+1:03d}_{safe_group}.txt"
                zipf.writestr(filename, file_content)
            
            # Add consolidated JSON with consistent structure
            export_data = []
            for result in results:
                item = {
                    "group": result.get("group", ""),
                    "main_content": result.get("main_content", "")
                }
                
                # Add all original file columns
                for col in original_columns:
                    item[col] = result.get(col, "")
                
                # Add prompt if requested
                if include_prompt:
                    item["prompt"] = result.get("prompt", "")
                
                # Add response and timestamp
                item["response"] = result.get("response", "")
                item["timestamp"] = result.get("timestamp", "")
                
                export_data.append(item)
            
            json_filename = f"results_{timestamp}.json" if timestamp else "results.json"
            zipf.writestr(f"consolidated/{json_filename}", json.dumps(export_data, indent=2))
            
            # Add consolidated CSV with consistent structure
            csv_content = io.StringIO()
            fieldnames = ["group", "main_content"] + original_columns
            if include_prompt:
                fieldnames.append("prompt")
            fieldnames.extend(["response", "timestamp"])
            
            writer = csv.DictWriter(csv_content, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                row = {
                    "group": result.get("group", ""),
                    "main_content": result.get("main_content", "")
                }
                
                # Add all original file columns
                for col in original_columns:
                    row[col] = result.get(col, "")
                
                # Add prompt if requested
                if include_prompt:
                    row["prompt"] = result.get("prompt", "")
                
                # Add response and timestamp
                row["response"] = result.get("response", "")
                row["timestamp"] = result.get("timestamp", "")
                
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