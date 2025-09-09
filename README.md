# AI Batch Processor

A powerful web application for batch processing datasets with AI models (OpenAI and Anthropic). Upload your data, configure AI parameters, and process multiple items with intelligent conversation grouping.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python run.py
   ```

3. **Open your browser:**
   - Frontend: http://localhost:8000
   - API docs: http://localhost:8000/docs

## 📋 Features

- **File Upload**: Support for CSV, JSON, and TXT files
- **AI Integration**: Works with OpenAI (GPT-4, GPT-3.5) and Anthropic (Claude) models
- **Conversation Grouping**: Maintain context across related items
- **Real-time Progress**: Live updates via WebSocket
- **Batch Controls**: Pause, resume, and stop processing
- **Flexible Output**: Individual files, consolidated CSV/JSON, or both
- **Rate Limiting**: Respect API limits with configurable rates

## 🛠 Usage

1. **Upload Data**: Drag & drop or select your CSV/JSON/TXT file
2. **Configure AI**: Choose service (OpenAI/Anthropic), model, and API key
3. **Map Columns**: Select grouping and content columns
4. **Design Prompts**: Create templates with {column_name} placeholders
5. **Set Output**: Choose format and options
6. **Process**: Start batch processing with real-time monitoring

## 📁 File Formats

- **CSV**: Header row defines columns
- **JSON**: Array of objects, keys become columns  
- **TXT**: Each line becomes a "content" column

## 🔧 Configuration

### AI Services
- **OpenAI**: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
- **Anthropic**: claude-3-5-sonnet, claude-3-opus, claude-3-sonnet, claude-3-haiku

### Parameters
- Temperature (0.0-2.0)
- Max tokens (1-4000)
- Rate limit (requests/minute)
- Retry attempts

## 📤 Output Options

- **Individual**: Separate text file per item
- **CSV**: Consolidated spreadsheet format
- **JSON**: Structured data format
- **Both**: Individual + consolidated in ZIP

## 🔒 Security

- API keys are never stored or logged
- All processing happens locally
- No data sent to external services except AI APIs

## 🐛 Troubleshooting

- Ensure your API key is valid and has sufficient credits
- Check file format matches expected structure
- Monitor rate limits to avoid API errors
- Use smaller batches for testing

## 📝 Example

**CSV Input:**
```csv
topic,audience,priority
AI,Teachers,High
Python,Students,Medium
```

**Prompt Template:**
```
Write an article about {topic} for {audience} with {priority} priority.
```

**Result:**
Two AI-generated articles customized for each row.