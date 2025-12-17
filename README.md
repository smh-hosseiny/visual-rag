# VisualRAG: Agentic RAG for Competitive Intelligence

<p align="center">
  <img src="https://img.shields.io/badge/Vision_Model-DeepSeek_OCR-blue" alt="DeepSeek-OCR">
  <img src="https://img.shields.io/badge/LLM-Gemini_1.5_Flash-orange" alt="Gemini">
  <img src="https://img.shields.io/badge/Orchestration-LangGraph-red" alt="LangGraph">
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688" alt="FastAPI">
  <br>
  <img src="https://img.shields.io/badge/Framework-LangChain-green" alt="LangChain">
  <img src="https://img.shields.io/badge/Vector_DB-ChromaDB-purple" alt="ChromaDB">
  <img src="https://img.shields.io/badge/Accelerator-CUDA-76B900" alt="CUDA">
</p>

<p align="center">
  <video src="assets/demo.mp4" controls width="85%">
    Your browser does not support the video tag.
  </video>
  <br>
  <em>(Watch the autonomous agent analyze H100 GPU prices)</em>
</p>

---

**VisualRAG** is an autonomous Agentic RAG system designed to automate the heavy lifting of market research. Unlike traditional scrapers that fail on complex documents, VisualRAG treats research as a visual task, reading PDFs and charts pixel-by-pixel using **DeepSeek-OCR** to extract structured data that standard LLMs miss.

It autonomously plans research strategies, finds high-value whitepapers, and compresses hours of analysis into a concise, executive-grade report.

---

## ✨ Key Features
- **Autonomous orchestration**  
  A self-correcting workflow that handles search failures.
- **Hybrid compute setup**  
  Designed for both cost and performance: GPU-heavy visual processing runs locally (CUDA), while higher-level reasoning is handled in the cloud.
- **Visual-first data extraction**  
  Able to pull useful information from “unreadable” PDFs, where important market details often hide.
- **Clean, structured outputs**  
  Generates well-formatted Markdown reports with clear sections and direct citations, ready to share with decision-makers.

---

## 📂 Project Structure

```plaintext
visual-rag/
├── app/
│   ├── agent.py       # LangGraph orchestration logic
│   ├── api.py         # FastAPI backend
│   ├── ocr.py         # DeepSeek-OCR local inference
│   ├── rag.py         # Vector DB (Chroma) & Embeddings
│   └── tools.py       # Search & PDF processing tools
├── assets/            # Demo videos and screenshots
├── data/              # Local storage for PDFs/Reports
├── environment.yml    # Conda environment definition
├── index.html         # Frontend UI
├── pyproject.toml     # Project dependencies & configuration
└── run_server.sh      # Startup script
```
---

## ⚙️ The Pipeline

VisualRAG operates as a fully autonomous directed graph (DAG) orchestrated by **LangGraph**. Here is the lifecycle of a single research task:

### 1. **Plan & Search** 🧭
The agent receives a high-level objective (e.g., *"Analyze NVIDIA H100 pricing vs. AMD MI300"*). It autonomously generates search queries using **Tavily API**, specifically targeting high-signal sources like investor reports, whitepapers, and technical data sheets (filtering for pdf and image assets).

### 2. **Visual Ingestion** 👁️
Standard text scrapers often destroy the context of tables and charts. VisualRAG takes a different approach:
* It downloads retrieved documents locally.
* It converts PDFs into high-resolution images using `pdf2image`.
* This ensures that layout, row/column alignment, and visual hierarchies are preserved for the next step.

### 3. **DeepSeek-OCR (Local Vision)** 🧠
The core of the system is a local instance of **DeepSeek-OCR** running on an **RTX 3090**.
* The model "reads" the document images, performing vision-language extraction.
* It parses complex pricing tables and technical specifications that standard OCR tools miss.
* This runs entirely on-premise for speed and data privacy.

### 4. **RAG & Synthesis** 📊
* **Embeddings:** The extracted text is chunked and embedded using `all-MiniLM-L6-v2` (running locally via HuggingFace) into a persistent **ChromaDB** vector store.
* **Reasoning:** The system retrieves the most relevant chunks and feeds them into **Gemini 2.5 Flash**. The LLM synthesizes the disparate data points into a structured Markdown report with citations, SWOT analysis, and exact pricing data.

---

## 🛠️ Environment Setup

This project uses Conda to manage dependencies and CUDA environments.

### 1. Prerequisites
* **OS:** Linux (Ubuntu 20.04+) or Windows WSL2
* **GPU:** NVIDIA GPU with at least 12GB VRAM (RTX 3090 recommended)
* **Drivers:** NVIDIA Drivers & CUDA 11.8+

### 2. Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/smh-hosseiny/visual-rag.git
    cd visual-rag
    ```

2.  **Create the Conda Environment:**
    Use the provided `environment.yml` to set up Python 3.10 and CUDA toolkit.
    ```bash
    conda env create -f environment.yml
    conda activate llm-ocr
    ```

3.  **Configure API Keys:**
    Create a `.env` file in the root directory:
    ```bash
    GOOGLE_API_KEY=your_gemini_key
    TAVILY_API_KEY=your_tavily_key
    ```

---

## 🚀 Running the Agent

### Option 1: Using the Startup Script

```bash
bash run_server.sh
```

### Option 2: Manual Start

```bash
uvicorn app.api:app --host 0.0.0.0 --port 8001 --reload
```

The server will start at **http://localhost:8001**


### Accessing the Frontend

**Option A: Direct File Access** 
1. Open `index.html` in your browser
2. The interface will connect to `http://localhost:8001`

**Option B: Using Live Server** (VS Code)
1. Install "Live Server" extension in VS Code
2. Right-click `index.html` → "Open with Live Server"
3. Opens at `http://127.0.0.1:5500`

---

## 📊 Usage Examples

### Example 1: Competitive Analysis
```
Topic: "NVIDIA H100 vs AMD MI300X pricing comparison"
```

**Expected Output:**
- Executive summary of GPU market positioning
- Detailed pricing breakdown with exact figures
- SWOT analysis for both products
- Strategic recommendations

---

## 🔧 Configuration

### Adjusting Processing Limits

Edit `app/agent.py`for number of web sources to process:

```python
limit = 3  # In ocr_node()
```

### Changing LLM Model

Edit `app/agent.py` to change the model:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  
    temperature=0,
    max_retries=2
)
```


### 📉 Handling OOM Errors (Lowering Resolution)

If you encounter CUDA Out-Of-Memory errors on smaller GPUs (e.g., <12GB VRAM), reduce the inference resolution in `app/ocr.py`:

```python
# In app/ocr.py

result = self.model.infer(
    ...
    base_size=512,   # 📉 Reduce this (Default: 1024)
    image_size=480,  # 📉 Reduce this (Default: 640)
    ...
)
```

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **DeepSeek AI** for the DeepSeek-OCR model
- **Google** for Gemini API
- **Tavily** for search API
- **LangChain** team for LangGraph framework

---

## 📧 Contact

For questions or support, please open an issue on GitHub or contact [hosseiny290@gmail.com]

---