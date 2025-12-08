from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DocumentMetadata:
    source_path: str
    file_type: str
    num_pages: Optional[int] = None

@dataclass
class DocumentChunk:
    content: str
    chunk_id: str
    document_id: str
    metadata: Optional[DocumentMetadata] = None

@dataclass
class IngestionResult:
    document_path: str
    num_chunks: int
    status: str
