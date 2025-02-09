# core/rag_core/indexing/chunker.py

from typing import List, Optional, Any
from langchain.text_splitter import TokenTextSplitter, TextSplitter
from langchain.docstore.document import Document as LangChainDocument
from syntok.segmenter import analyze

from core.datasets.schema import Document as SchemaDocument

class Chunker:
    """Document chunking using LangChain's text splitters with support for both fixed-size and sentence-aware strategies."""
    
    def __init__(self, chunk_size: int = 100, chunk_overlap: float = 0.2):
        """Initialize the chunker.
        
        Args:
            chunk_size: Target size for chunks in tokens (minimum size)
            chunk_overlap: Overlap ratio between chunks (0.0 to 1.0)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = int(chunk_size * chunk_overlap)
    
    def run(self, documents: List[SchemaDocument], mode: str = "fixed") -> List[LangChainDocument]:
        """Create document chunks using the specified mode.
        
        Args:
            documents: List of SchemaDocument objects to split
            mode: Chunking mode ("fixed" or "sentence")
        
        Returns:
            List of LangChainDocument chunks
        """
        if mode == "fixed":
            return self._create_fixed_chunks(documents)
        elif mode == "sentence":
            return self._create_sentence_chunks(documents)
        else:
            raise ValueError(f"Unknown chunking mode: {mode}")

    def _create_fixed_chunks(self, documents: List[SchemaDocument]) -> List[LangChainDocument]:
        """Create fixed-size chunks with overlap using TokenTextSplitter."""
        splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        all_chunks = []
        for doc in documents:
            langchain_doc = self._convert_to_langchain_doc(doc)
            chunks = splitter.split_documents([langchain_doc])
            all_chunks.extend(self._add_metadata(chunks, doc))
        return all_chunks
    
    def _create_sentence_chunks(self, documents: List[SchemaDocument]) -> List[LangChainDocument]:
        """Create chunks while preserving sentence boundaries.
        Each chunk will be at least chunk_size tokens, completing the current sentence.
        """
        splitter = MinSizeTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        all_chunks = []
        for doc in documents:
            langchain_doc = self._convert_to_langchain_doc(doc)
            chunks = splitter.split_documents([langchain_doc])
            all_chunks.extend(self._add_metadata(chunks, doc))
        return all_chunks

    def _convert_to_langchain_doc(self, doc: SchemaDocument) -> LangChainDocument:
        """Convert our schema Document to LangChain Document."""
        return LangChainDocument(
            page_content=doc.content,
            metadata={}
            # TODO: Consider adding metadata to the LangChainDocument
            # metadata={
            #     "doc_id": doc.id,
            #     **doc.metadata.model_dump(exclude_none=True)
            # }
        )
    
    def _add_metadata(self, chunks: List[LangChainDocument], source_doc: SchemaDocument) -> List[LangChainDocument]:
        """Add additional metadata to chunks including start and end indices."""
        current_index = 0
        for i, chunk in enumerate(chunks):
            # Calculate start and end indices
            chunk_text = chunk.page_content
            if i > 0:  # For chunks after the first one
                # Find the start position in the source document after the previous end
                start_idx = source_doc.content.find(chunk_text, current_index)
            else:
                # For the first chunk, start from the beginning
                start_idx = source_doc.content.find(chunk_text)
            
            # If found, update indices
            if start_idx != -1:
                end_idx = start_idx + len(chunk_text)
                current_index = end_idx
            else:
                # Fallback if exact match not found (e.g., due to whitespace differences)
                start_idx = current_index
                end_idx = start_idx + len(chunk_text)
            
            chunk.metadata.update({
                'chunk_index': i,
                'source_doc_id': source_doc.id,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        return chunks

class MinSizeTextSplitter(TextSplitter):
    """Text splitter that ensures minimum chunk size while preserving sentence boundaries."""

    def __init__(
        self,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
        length_function: Optional[callable] = None,
        **kwargs: Any,
    ):
        """Initialize the text splitter.
        
        Args:
            chunk_size: The minimum size of chunks in tokens
            chunk_overlap: The number of tokens to overlap between chunks
            length_function: Function to measure text length (optional)
        """
        super().__init__(**kwargs)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._length_function = length_function or (lambda x: len(x.split()))
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using syntok's analyzer."""
        if not text.strip():
            return []
        
        sentences = []
        # analyze returns paragraphs containing sentences containing tokens
        for paragraph in analyze(text):
            # Join tokens in each sentence
            for sentence in paragraph:
                sentence_text = ''.join(token.spacing + token.value for token in sentence).strip()
                if sentence_text:
                    sentences.append(sentence_text)
        
        return sentences
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks of minimum size while preserving sentence boundaries."""
        # Split into sentences using syntok
        sentences = self._split_into_sentences(text)
        if not sentences:
            return [text]
        
        # Combine sentences until they meet minimum size
        result = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = self._length_function(sentence)
            
            # If a single sentence exceeds chunk size, keep it as is
            if sentence_size >= self._chunk_size and not current_chunk:
                result.append(sentence)
                continue
                
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size
            
            # If we have enough tokens
            if current_size >= self._chunk_size:
                result.append(" ".join(current_chunk))
                # Keep last sentence for overlap
                if len(current_chunk) > 1:  # Only keep for overlap if we have more than one sentence
                    current_chunk = [current_chunk[-1]]
                    current_size = self._length_function(current_chunk[0])
                else:
                    current_chunk = []
                    current_size = 0
        
        # Add any remaining text
        if current_chunk:
            result.append(" ".join(current_chunk))
        
        return result
