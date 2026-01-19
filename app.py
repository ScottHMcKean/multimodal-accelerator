"""
Multimodal Document Chat App

This app showcases simplified document processing using:
- Databricks Managed MCP (Model Context Protocol) for chat completions  
- Delta tables for document storage and retrieval
- Direct foundation model integration (no custom agent code)

The app demonstrates the three core processing paths:
1. AI_PARSE - Databricks native document parsing
2. Docling + Ray - Parallel processing with native Docling
3. Docling + Serving - Serving endpoint-based processing
"""

import logging
import os
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import ExecuteStatementRequest
import pandas as pd
from typing import Dict, List, Optional

# Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get workspace client
workspace_client = WorkspaceClient()

# Configuration from environment variables
CATALOG = os.getenv("CATALOG", "main")
SCHEMA = os.getenv("SCHEMA", "default") 
DOCUMENTS_TABLE = f"{CATALOG}.{SCHEMA}.{os.getenv('DOCUMENTS_TABLE', 'processed_documents')}"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.{os.getenv('CHUNKS_TABLE', 'document_chunks')}"

# App configuration
st.set_page_config(
    page_title="Document Chat - Simplified",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Simplified
st.markdown("""
<style>
    .main {
        background-color: #F5F6FA;
    }
    .block-container {
        padding-top: 2rem;
    }
    .doc-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e0e0e0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .stChatMessage {
        background: #fff;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        padding: 1rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


class DocumentRetriever:
    """Simple document retriever using SQL queries on tables."""
    
    def __init__(self):
        self.workspace_client = workspace_client
    
    def search_documents(self, query: str, limit: int = 5) -> List[Dict]:
        """Search documents using SQL similarity search."""
        
        # Simple text search query - can be enhanced with vector search
        sql_query = f"""
        SELECT 
            doc_path,
            chunk_text,
            page_number,
            chunk_index,
            similarity_score
        FROM (
            SELECT 
                doc_path,
                chunk_text,
                page_number, 
                chunk_index,
                -- Simple text similarity (can be replaced with vector similarity)
                CASE 
                    WHEN LOWER(chunk_text) LIKE LOWER('%{query}%') THEN 0.9
                    WHEN LOWER(chunk_text) LIKE LOWER('%{query.split()[0]}%') THEN 0.7
                    ELSE 0.0
                END as similarity_score
            FROM {CHUNKS_TABLE}
            WHERE LENGTH(chunk_text) > 50
        )
        WHERE similarity_score > 0.0
        ORDER BY similarity_score DESC, LENGTH(chunk_text) DESC
        LIMIT {limit}
        """
        
        try:
            # Execute query using Databricks SQL
            response = self.workspace_client.statement_execution.execute_statement(
                warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),
                statement=sql_query,
                wait_timeout="30s"
            )
            
            results = []
            if response.result and response.result.data_array:
                for row in response.result.data_array:
                    results.append({
                        "doc_path": row[0],
                        "chunk_text": row[1],
                        "page_number": row[2],
                        "chunk_index": row[3],
                        "similarity_score": float(row[4])
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Document search failed: {e}")
            return []
    
    def get_document_info(self, doc_path: str) -> Optional[Dict]:
        """Get document metadata from the processed documents table."""
        
        sql_query = f"""
        SELECT 
            doc_path,
            pages,
            pictures,
            tables,
            processing_method,
            created_at
        FROM {DOCUMENTS_TABLE}
        WHERE doc_path = '{doc_path}'
        LIMIT 1
        """
        
        try:
            response = self.workspace_client.statement_execution.execute_statement(
                warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),
                statement=sql_query,
                wait_timeout="10s"
            )
            
            if response.result and response.result.data_array:
                row = response.result.data_array[0]
                return {
                    "doc_path": row[0],
                    "pages": int(row[1]) if row[1] else 0,
                    "pictures": int(row[2]) if row[2] else 0,
                    "tables": int(row[3]) if row[3] else 0,
                    "processing_method": row[4],
                    "created_at": row[5]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Document info retrieval failed: {e}")
            return None


def get_databricks_mcp_response(query: str, context: List[Dict]) -> str:
    """
    Use Databricks Managed MCP (Model Context Protocol) for chat completions.
    MCP is a Databricks-managed service that provides standardized access to foundation models.
    """
    
    # Format context from retrieved documents
    context_text = ""
    if context:
        context_text = "\n\n**Relevant Document Context:**\n"
        for i, doc in enumerate(context, 1):
            doc_name = doc['doc_path'].split('/')[-1]
            context_text += f"\n{i}. From {doc_name} (Page {doc.get('page_number', 'Unknown')}):\n"
            context_text += f"   {doc['chunk_text'][:300]}...\n"
    
    # Create prompt for Databricks Foundation Model
    prompt = f"""You are a helpful assistant that answers questions about documents.

{context_text}

User Question: {query}

Instructions:
- Answer based on the document context provided above
- Be specific and cite which document your information comes from
- If the context doesn't contain enough information, say so
- Keep your response clear and concise"""

    try:
        # Use Databricks Foundation Model API directly
        # This could be replaced with any Databricks-managed model endpoint
        from databricks_langchain import ChatDatabricks
        
        # Use Databricks Managed MCP to access foundation models
        chat_model = ChatDatabricks(
            endpoint=os.getenv("DATABRICKS_LLM_ENDPOINT", "databricks-meta-llama-3-1-405b-instruct")
        )
        
        response = chat_model.invoke(prompt)
        return response.content
        
    except Exception as e:
        logger.error(f"MCP response failed: {e}")
        return "I apologize, but I'm having trouble processing your request right now. Please try again."


def main():
    """Main Streamlit app."""
    
    st.title("📄 Multimodal Document Chat")
    st.caption("Powered by Databricks Managed MCP and Delta table storage")
    
    # Initialize retriever
    if "retriever" not in st.session_state:
        st.session_state.retriever = DocumentRetriever()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "last_context" not in st.session_state:
        st.session_state.last_context = []
    
    # Sidebar - Document Statistics
    with st.sidebar:
        st.header("📊 Document Statistics")
        
        try:
            # Get document stats from table
            stats_query = f"""
            SELECT 
                COUNT(*) as total_docs,
                SUM(pages) as total_pages,
                SUM(pictures) as total_pictures,
                SUM(tables) as total_tables
            FROM {DOCUMENTS_TABLE}
            """
            
            response = workspace_client.statement_execution.execute_statement(
                warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),
                statement=stats_query,
                wait_timeout="10s"
            )
            
            if response.result and response.result.data_array:
                row = response.result.data_array[0]
                st.metric("Documents", row[0])
                st.metric("Total Pages", row[1] or 0)
                st.metric("Pictures", row[2] or 0)
                st.metric("Tables", row[3] or 0)
            
        except Exception as e:
            st.error(f"Could not load statistics: {e}")
        
        st.markdown("---")
        
        # Show recent context if available
        if st.session_state.last_context:
            st.subheader("🔍 Last Retrieved")
            for doc in st.session_state.last_context[:3]:
                doc_name = doc['doc_path'].split('/')[-1]
                with st.expander(f"{doc_name} (Page {doc.get('page_number', '?')})"):
                    st.caption(f"Similarity: {doc.get('similarity_score', 0):.2f}")
                    st.text(doc['chunk_text'][:200] + "...")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Process user query
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching documents and generating response..."):
                # Retrieve relevant documents from Delta tables
                context = st.session_state.retriever.search_documents(prompt, limit=int(os.getenv("MAX_SEARCH_RESULTS", "5")))
                st.session_state.last_context = context
                
                # Get response using Databricks Managed MCP
                response = get_databricks_mcp_response(prompt, context)
                
                st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    # Check required environment variables
    required_env_vars = [
        "DATABRICKS_WAREHOUSE_ID", 
        "DATABRICKS_LLM_ENDPOINT"
    ]
    
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        st.error(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        st.info("Please configure these in your app.yaml file and redeploy.")
        st.info("Example: DATABRICKS_WAREHOUSE_ID: 'abc123def456'")
        st.stop()
    
    # Show configuration info
    with st.sidebar:
        with st.expander("🔧 Configuration"):
            st.write(f"**Warehouse ID:** {os.getenv('DATABRICKS_WAREHOUSE_ID')[:8]}...")
            st.write(f"**LLM Endpoint:** {os.getenv('DATABRICKS_LLM_ENDPOINT')}")
            st.write(f"**Catalog:** {CATALOG}")
            st.write(f"**Schema:** {SCHEMA}")
    
    main()