# Retrieval Grounded LLM

A Retrieval-Augmented Generation (RAG) knowledge pipeline that ingests MDN documentation, generates embeddings, and stores them in a vector database for semantic retrieval.

This repository builds the knowledge layer used by the runtime system to ground LLM responses in authoritative documentation.

# What This Repository Does

`retrieval-grounded-llm` builds the **knowledge layer** used by the AI runtime.

It ingests MDN documentation, transforms it into semantically searchable chunks, generates embeddings, and stores them in a vector database so that relevant documentation can be retrieved at runtime.

This enables the system to answer developer questions using **retrieved documentation rather than relying solely on model knowledge**.

Core responsibilities of this repository:

- **Documentation ingestion**  
  Crawl and normalize MDN documentation for downstream processing.

- **Semantic chunking**  
  Split documentation into retrieval-optimized chunks while preserving context.

- **Embedding generation**  
  Convert documentation chunks into vector embeddings.

- **Vector storage**  
  Store embeddings in PostgreSQL using `pgvector`.

- **Semantic retrieval support**  
  Provide the searchable knowledge base used by the runtime system.

At runtime, the system retrieves the most relevant documentation chunks and injects them into the LLM prompt to generate **grounded answers with source citations**.

# System Architecture

This repository provides the **knowledge ingestion and embedding pipeline** used by the runtime system.

It prepares documentation so that it can be retrieved and used to ground LLM responses.

                     USER
                      │
                      ▼
                ┌─────────────┐
                │  runtime-ui │
                │ React Chat  │
                └─────────────┘
                      │
                      │  POST /chat
                      ▼
           ┌──────────────────────┐
           │   ai-runtime-server   │
           │   RAG Runtime API     │
           │                       │
           │ • Query Embedding     │
           │ • Vector Search       │
           │ • Context Assembly    │
           │ • LLM Generation      │
           └──────────────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │  control-plane  │
             │ Agent Runtime   │
             │ Architecture    │
             │                 │
             │ • Orchestrator  │
             │ • Policy Layer  │
             │ • Kernel        │
             │ • Tool System   │
             └─────────────────┘
                      │
                      ▼
               ┌──────────────┐
               │   rag-mdn     │
               │ Knowledge     │
               │ Ingestion     │
               │               │
               │ • Crawl MDN   │
               │ • Chunk Docs  │
               │ • Embeddings  │
               └──────────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Postgres +      │
             │ pgvector        │
             │ Vector Database │
             └─────────────────┘

## Retrieval Pipeline

The system follows a retrieval-first architecture.

```
Documentation
   │
   ▼
Parsing
   │
   ▼
Chunking
   │
   ▼
Embedding Generation
   │
   ▼
Vector Storage
   │
   ▼
Semantic Retrieval
   │
   ▼
Context Injection
   │
   ▼
LLM Response
```

This ensures responses are grounded in retrieved documentation rather than model hallucination.

## Documentation Ingestion

Documentation is sourced from [MDN Web Docs](https://developer.mozilla.org/).

**Processing steps:**

- Markdown parsing
- structural normalization
- heading extraction
- semantic block identification

**Goal:** preserve the original document structure before chunking.

## Semantic Chunking

Documentation is divided into semantic chunks optimized for retrieval.

**Strategy:**

- token-based chunk sizing
- controlled overlap between chunks
- chunk boundary optimization

**Benefits:**

- prevents semantic fragmentation
- improves retrieval precision
- preserves contextual meaning

## Embedding Generation

Each documentation chunk is converted into a vector embedding.

Embeddings enable semantic search by mapping text into high-dimensional vector space.

**Example pipeline:**

```
Markdown chunk
   │
   ▼
Embedding model
   │
   ▼
Vector representation
```

## Vector Storage

Embeddings are stored in PostgreSQL using [pgvector](https://github.com/pgvector/pgvector).

**Example table:** `document_embeddings`

**Columns:**

| Column    | Description      |
| --------- | ---------------- |
| text      | Chunk content    |
| source    | Document source  |
| embedding | Vector embedding |

Vector similarity search allows the system to retrieve relevant documentation for user queries.

## Retrieval

When a query is received:

```
User Query
   │
   ▼
Query Embedding
   │
   ▼
Vector Similarity Search
   │
   ▼
Top-K Documentation Chunks
```

These chunks are returned to the runtime server to construct the LLM prompt.

## Evaluation

Retrieval performance is evaluated using [Promptfoo](https://www.promptfoo.dev/).

**Evaluation focuses on:**

- retrieval accuracy
- grounding fidelity
- hallucination reduction
- citation correctness

Retrieval parameters are iteratively tuned based on evaluation results.

## Design Principles

This system follows several design principles:

- retrieval-first architecture
- deterministic context selection
- transparent citation mapping
- evaluation-driven refinement
- production-oriented embedding strategy

## Repository Structure

```
rag-mdn
├─ ingestion
├─ embeddings
├─ postgres
├─ scripts
└─ evaluation
```

Each component corresponds to a stage of the RAG pipeline.

# System Repositories

This project is part of a modular AI system composed of several repositories.

| Repository                                                            | Description                                                      |
| --------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [control-plane](https://github.com/johntmunger/ai-control-plane)      | Agent runtime architecture (orchestrator, policy, kernel, tools) |
| [ai-runtime-server](https://github.com/johntmunger/ai-runtime-server) | Retrieval-Augmented Generation (RAG) runtime and chat API        |
| [rag-mdn](https://github.com/johntmunger/retrieval-grounded-llm)      | Documentation ingestion and embedding pipeline                   |
| [runtime-ui](https://github.com/johntmunger/ai-runtime-ui)            | Chat interface for interacting with the runtime                  |

## Summary

**rag-mdn** provides the knowledge ingestion and retrieval infrastructure used by the runtime system.

**Key capabilities:**

- structured documentation ingestion
- semantic chunking
- embedding generation
- vector similarity search
- retrieval for RAG pipelines

This repository forms the knowledge backbone of the AI system.
