#!/usr/bin/env python3
"""
Simple startup script for the AI Batch Processor
"""
import uvicorn
import sys
import os

def main():
    port = int(os.environ.get("PORT", 8000))
    print("🚀 Starting AI Batch Processor...")
    print(f"📁 Frontend will be available at: http://localhost:{port}")
    print(f"🔧 API docs available at: http://localhost:{port}/docs")
    print("⚡ Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "main:final_app", 
            host="0.0.0.0", 
            port=port, 
            reload=False,  # Disable reload in production
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()