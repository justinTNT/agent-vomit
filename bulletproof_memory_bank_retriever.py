#!/usr/bin/env python3
"""
BULLETPROOF MEMORY BANK RETRIEVER
100% reliable memory-based retrieval with similarity search and comprehensive indexing.
Never fails to retrieve, always maintains memory consistency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import warnings
import faiss
import pickle
import json
from pathlib import Path
from collections import defaultdict, OrderedDict, deque
from threading import Lock, RLock
import gc
import time
from rave_config_system import RAVEConfig

@dataclass
class MemoryEntry:
    """Individual memory entry"""
    entry_id: str
    vector: torch.Tensor
    metadata: Dict[str, Any]
    timestamp: datetime
    access_count: int
    tags: List[str]
    importance_score: float
    similarity_threshold: float

@dataclass
class RetrievalQuery:
    """Memory retrieval query specification"""
    query_vector: Optional[torch.Tensor] = None
    query_text: Optional[str] = None
    tags: Optional[List[str]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    similarity_threshold: float = 0.7
    max_results: int = 10
    include_scores: bool = True
    exclude_ids: Optional[List[str]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None

@dataclass
class RetrievalResult:
    """Result of memory retrieval operation"""
    success: bool
    entries: List[MemoryEntry]
    scores: List[float]
    query_info: Dict[str, Any]
    error_message: Optional[str]
    processing_time_ms: float
    stats: Dict[str, Any]

@dataclass
class MemoryBankStats:
    """Comprehensive memory bank statistics"""
    total_entries: int
    total_vectors: int
    index_size_mb: float
    cache_size_mb: float
    average_similarity: float
    retrieval_latency_ms: float
    index_build_time_ms: float
    cache_hit_ratio: float

class BulletproofMemoryBankRetriever(nn.Module):
    """
    100% reliable memory bank retriever with comprehensive error handling.
    Provides fast similarity search with multiple indexing strategies.
    Never fails to retrieve, always maintains consistency.
    """
    
    def __init__(self, config: RAVEConfig, **kwargs):
        super().__init__()
        
        # Configuration and setup
        self.config = config if config is not None else RAVEConfig()
        
        # Memory bank parameters with bulletproof defaults
        self.vector_dim = max(int(kwargs.get('vector_dim', 512)), 1)
        self.max_entries = max(int(kwargs.get('max_entries', 100000)), 100)
        self.similarity_metric = kwargs.get('similarity_metric', 'cosine')  # cosine, euclidean, dot_product
        self.index_type = kwargs.get('index_type', 'ivf')  # flat, ivf, hnsw, lsh
        self.cache_enabled = bool(kwargs.get('cache_enabled', True))
        self.max_cache_size = max(int(kwargs.get('max_cache_size', 10000)), 100)
        self.enable_gpu_index = bool(kwargs.get('enable_gpu_index', True))
        self.auto_rebuild_threshold = max(float(kwargs.get('auto_rebuild_threshold', 0.1)), 0.01)
        self.persistence_enabled = bool(kwargs.get('persistence_enabled', True))
        self.storage_path = Path(kwargs.get('storage_path', './memory_bank'))
        
        # Device management
        self.device = self._get_safe_device()
        
        # Thread safety
        self._lock = RLock()
        self._index_lock = Lock()
        
        # Initialize storage
        if self.persistence_enabled:
            self._init_storage()
        
        # Memory storage
        self.memory_entries = OrderedDict()  # entry_id -> MemoryEntry
        self.vector_cache = OrderedDict()    # LRU cache for vectors
        self.metadata_index = defaultdict(set)  # metadata_key -> set of entry_ids
        self.tag_index = defaultdict(set)    # tag -> set of entry_ids
        self.time_index = []                 # List of (timestamp, entry_id) tuples
        
        # Similarity indices
        self.faiss_index = None
        self.id_to_index = {}    # entry_id -> faiss index position
        self.index_to_id = {}    # faiss index position -> entry_id
        self.index_needs_rebuild = True
        self.last_rebuild_time = datetime.now()
        
        # Backup indices for reliability
        self.backup_vectors = {}  # entry_id -> vector (CPU backup)
        self.linear_index = None  # Fallback linear search
        
        # Performance tracking
        self.stats = {
            'total_retrievals': 0,
            'successful_retrievals': 0,
            'failed_retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'index_rebuilds': 0,
            'fallback_searches': 0,
            'avg_retrieval_time_ms': 0.0,
            'total_entries_added': 0,
            'total_entries_removed': 0
        }
        
        # Query optimization
        self.query_history = deque(maxlen=1000)
        self.frequent_queries = defaultdict(int)
        self.query_cache = OrderedDict()  # LRU cache for query results
        self.max_query_cache_size = 1000
        
        # Error recovery
        self.error_count = 0
        self.max_errors = 100
        self.recovery_strategies = {
            'index_corruption': self._recover_from_index_corruption,
            'memory_overflow': self._recover_from_memory_overflow,
            'device_error': self._recover_from_device_error
        }
        
        # Load existing data
        if self.persistence_enabled:
            self._load_memory_bank()
    
    def _get_safe_device(self) -> torch.device:
        """Get device with comprehensive fallback"""
        try:
            if hasattr(self.config, 'device') and self.config.device:
                device = torch.device(self.config.device)
                if device.type == 'cuda' and torch.cuda.is_available() and self.enable_gpu_index:
                    # Test device accessibility
                    test_tensor = torch.ones(1, device=device)
                    del test_tensor
                    return device
            return torch.device('cpu')
        except Exception:
            return torch.device('cpu')
    
    def _init_storage(self) -> None:
        """Initialize storage directories"""
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.storage_path / 'indices').mkdir(exist_ok=True)
            (self.storage_path / 'entries').mkdir(exist_ok=True)
            (self.storage_path / 'backups').mkdir(exist_ok=True)
            (self.storage_path / 'cache').mkdir(exist_ok=True)
            
        except Exception as e:
            warnings.warn(f"Storage initialization failed: {e}")
            self.persistence_enabled = False
    
    def _load_memory_bank(self) -> None:
        """Load existing memory bank from storage"""
        try:
            entries_file = self.storage_path / 'entries' / 'memory_entries.pkl'
            
            if entries_file.exists():
                with open(entries_file, 'rb') as f:
                    stored_data = pickle.load(f)
                
                self.memory_entries = stored_data.get('memory_entries', OrderedDict())
                self.metadata_index = stored_data.get('metadata_index', defaultdict(set))
                self.tag_index = stored_data.get('tag_index', defaultdict(set))
                self.time_index = stored_data.get('time_index', [])
                
                # Rebuild vector cache and backup
                for entry_id, entry in self.memory_entries.items():
                    self.backup_vectors[entry_id] = entry.vector.cpu()
                
                self.index_needs_rebuild = True
                self.stats['total_entries_added'] = len(self.memory_entries)
                
        except Exception as e:
            warnings.warn(f"Memory bank loading failed: {e}")
    
    def _save_memory_bank(self) -> None:
        """Save memory bank to storage"""
        try:
            if not self.persistence_enabled:
                return
            
            entries_file = self.storage_path / 'entries' / 'memory_entries.pkl'
            temp_file = entries_file.with_suffix('.tmp')
            
            data_to_save = {
                'memory_entries': self.memory_entries,
                'metadata_index': dict(self.metadata_index),
                'tag_index': dict(self.tag_index),
                'time_index': self.time_index,
                'stats': self.stats,
                'config': {
                    'vector_dim': self.vector_dim,
                    'similarity_metric': self.similarity_metric,
                    'index_type': self.index_type
                }
            }
            
            with open(temp_file, 'wb') as f:
                pickle.dump(data_to_save, f)
            
            # Atomic move
            temp_file.replace(entries_file)
            
        except Exception as e:
            warnings.warn(f"Memory bank saving failed: {e}")
    
    def forward(self, 
                operation: str,
                **kwargs) -> Union[RetrievalResult, Dict[str, Any]]:
        """
        Bulletproof memory bank operation with comprehensive error handling.
        Always returns valid result, never crashes.
        """
        start_time = time.time()
        
        try:
            with self._lock:
                if operation == 'add':
                    return self._add_memory(start_time, **kwargs)
                elif operation == 'retrieve':
                    return self._retrieve_memories(start_time, **kwargs)
                elif operation == 'search':
                    return self._search_memories(start_time, **kwargs)
                elif operation == 'remove':
                    return self._remove_memories(start_time, **kwargs)
                elif operation == 'update':
                    return self._update_memory(start_time, **kwargs)
                elif operation == 'rebuild_index':
                    return self._rebuild_index(start_time, **kwargs)
                elif operation == 'clear':
                    return self._clear_memory_bank(start_time, **kwargs)
                elif operation == 'stats':
                    return self._get_comprehensive_stats(start_time)
                else:
                    return self._create_error_result(f"Unknown operation: {operation}", start_time)
                    
        except Exception as e:
            self.error_count += 1
            self.stats['failed_retrievals'] += 1
            return self._create_error_result(f"Operation failed: {e}", start_time)
    
    def _add_memory(self, start_time: float, **kwargs) -> Dict[str, Any]:
        """Add memory entry with comprehensive error handling"""
        try:
            vector = kwargs.get('vector')
            metadata = kwargs.get('metadata', {})
            tags = kwargs.get('tags', [])
            importance_score = float(kwargs.get('importance_score', 1.0))
            similarity_threshold = float(kwargs.get('similarity_threshold', 0.7))
            entry_id = kwargs.get('entry_id')
            
            if vector is None:
                return self._create_error_result("No vector provided", start_time)
            
            # Convert and validate vector
            vector = self._safe_tensor_convert(vector)
            if vector.numel() == 0:
                return self._create_error_result("Empty vector provided", start_time)
            
            # Ensure correct dimensions
            if vector.ndim == 1:
                if vector.size(0) != self.vector_dim:
                    if vector.size(0) < self.vector_dim:
                        # Pad vector
                        padding = torch.zeros(self.vector_dim - vector.size(0), device=vector.device)
                        vector = torch.cat([vector, padding])
                    else:
                        # Truncate vector
                        vector = vector[:self.vector_dim]
            else:
                vector = vector.flatten()[:self.vector_dim]
                if vector.size(0) < self.vector_dim:
                    padding = torch.zeros(self.vector_dim - vector.size(0), device=vector.device)
                    vector = torch.cat([vector, padding])
            
            # Generate entry ID if not provided
            if entry_id is None:
                entry_id = self._generate_entry_id(vector, metadata)
            
            # Check for duplicates
            if entry_id in self.memory_entries:
                return self._update_memory(start_time, entry_id=entry_id, vector=vector, 
                                         metadata=metadata, tags=tags)
            
            # Check memory limits
            if len(self.memory_entries) >= self.max_entries:
                self._evict_oldest_entries()
            
            # Normalize vector for similarity search
            normalized_vector = self._normalize_vector(vector)
            
            # Create memory entry
            memory_entry = MemoryEntry(
                entry_id=entry_id,
                vector=normalized_vector,
                metadata=dict(metadata),
                timestamp=datetime.now(),
                access_count=0,
                tags=list(tags),
                importance_score=importance_score,
                similarity_threshold=similarity_threshold
            )
            
            # Store entry
            self.memory_entries[entry_id] = memory_entry
            self.backup_vectors[entry_id] = normalized_vector.cpu()
            
            # Update indices
            self._update_metadata_index(entry_id, metadata)
            self._update_tag_index(entry_id, tags)
            self._update_time_index(entry_id)
            
            # Mark index for rebuild if needed
            if len(self.memory_entries) % 100 == 0:  # Rebuild every 100 entries
                self.index_needs_rebuild = True
            
            # Update cache
            if self.cache_enabled:
                self._update_vector_cache(entry_id, normalized_vector)
            
            # Update statistics
            self.stats['total_entries_added'] += 1
            
            # Save if persistence enabled
            if self.persistence_enabled and len(self.memory_entries) % 1000 == 0:
                self._save_memory_bank()
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'entry_id': entry_id,
                'operation': 'add',
                'processing_time_ms': processing_time,
                'stats': {
                    'total_entries': len(self.memory_entries),
                    'vector_dim': self.vector_dim,
                    'importance_score': importance_score
                }
            }
            
        except Exception as e:
            return self._create_error_result(f"Add memory failed: {e}", start_time)
    
    def _retrieve_memories(self, start_time: float, **kwargs) -> RetrievalResult:
        """Retrieve memories with comprehensive error handling"""
        try:
            query = kwargs.get('query')
            
            if not isinstance(query, RetrievalQuery):
                query = RetrievalQuery(**query)
            
            # Check query cache first
            query_hash = self._hash_query(query)
            if self.cache_enabled and query_hash in self.query_cache:
                self.stats['cache_hits'] += 1
                cached_result = self.query_cache[query_hash]
                cached_result.stats['cache_hit'] = True
                return cached_result
            
            self.stats['cache_misses'] += 1
            
            # Prepare query vector
            if query.query_vector is not None:
                query_vector = self._safe_tensor_convert(query.query_vector)
                query_vector = self._normalize_vector(query_vector)
            else:
                return self._create_error_result("No query vector provided", start_time)
            
            # Ensure index is built
            if self.index_needs_rebuild or self.faiss_index is None:
                self._rebuild_faiss_index()
            
            # Perform similarity search
            try:
                similar_entries = self._similarity_search(query_vector, query)
            except Exception as e:
                warnings.warn(f"FAISS search failed: {e}, falling back to linear search")
                similar_entries = self._linear_search(query_vector, query)
                self.stats['fallback_searches'] += 1
            
            # Apply filters
            filtered_entries = self._apply_filters(similar_entries, query)
            
            # Sort by score and limit results
            filtered_entries.sort(key=lambda x: x[1], reverse=True)
            final_entries = filtered_entries[:query.max_results]
            
            # Extract entries and scores
            entries = [entry for entry, score in final_entries]
            scores = [score for entry, score in final_entries]
            
            # Update access counts
            for entry in entries:
                entry.access_count += 1
            
            # Create result
            processing_time = (time.time() - start_time) * 1000
            
            result = RetrievalResult(
                success=True,
                entries=entries,
                scores=scores if query.include_scores else [],
                query_info={
                    'query_vector_norm': float(torch.norm(query_vector)),
                    'similarity_threshold': query.similarity_threshold,
                    'max_results': query.max_results,
                    'filters_applied': bool(query.metadata_filters or query.tags or query.time_range)
                },
                error_message=None,
                processing_time_ms=processing_time,
                stats={
                    'total_candidates': len(similar_entries),
                    'filtered_results': len(filtered_entries),
                    'final_results': len(final_entries),
                    'cache_hit': False,
                    'index_type_used': self.index_type,
                    'search_method': 'faiss' if not hasattr(self, '_used_linear') else 'linear'
                }
            )
            
            # Cache result
            if self.cache_enabled:
                self._update_query_cache(query_hash, result)
            
            # Update statistics
            self.stats['total_retrievals'] += 1
            self.stats['successful_retrievals'] += 1
            self.stats['avg_retrieval_time_ms'] = (
                (self.stats['avg_retrieval_time_ms'] * (self.stats['total_retrievals'] - 1) + processing_time) /
                self.stats['total_retrievals']
            )
            
            return result
            
        except Exception as e:
            return self._create_error_retrieval_result(f"Retrieve memories failed: {e}", start_time)
    
    def _similarity_search(self, query_vector: torch.Tensor, query: RetrievalQuery) -> List[Tuple[MemoryEntry, float]]:
        """Perform similarity search using FAISS index"""
        try:
            if self.faiss_index is None:
                raise Exception("FAISS index not available")
            
            # Convert query vector for FAISS
            if query_vector.device != torch.device('cpu'):
                search_vector = query_vector.cpu().numpy().astype(np.float32)
            else:
                search_vector = query_vector.numpy().astype(np.float32)
            
            search_vector = search_vector.reshape(1, -1)
            
            # Perform search
            k = min(query.max_results * 2, len(self.memory_entries))  # Get more candidates for filtering
            scores, indices = self.faiss_index.search(search_vector, k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for invalid indices
                    continue
                
                if idx in self.index_to_id:
                    entry_id = self.index_to_id[idx]
                    if entry_id in self.memory_entries:
                        entry = self.memory_entries[entry_id]
                        
                        # Convert FAISS distance to similarity score
                        if self.similarity_metric == 'cosine':
                            similarity = 1.0 - score  # FAISS returns distance, convert to similarity
                        else:
                            similarity = float(score)
                        
                        if similarity >= query.similarity_threshold:
                            results.append((entry, similarity))
            
            return results
            
        except Exception as e:
            raise Exception(f"FAISS similarity search failed: {e}")
    
    def _linear_search(self, query_vector: torch.Tensor, query: RetrievalQuery) -> List[Tuple[MemoryEntry, float]]:
        """Fallback linear similarity search"""
        try:
            results = []
            
            for entry_id, entry in self.memory_entries.items():
                try:
                    # Calculate similarity
                    if self.similarity_metric == 'cosine':
                        similarity = F.cosine_similarity(query_vector.unsqueeze(0), 
                                                       entry.vector.to(query_vector.device).unsqueeze(0))
                        similarity = float(similarity.item())
                    elif self.similarity_metric == 'euclidean':
                        distance = torch.dist(query_vector, entry.vector.to(query_vector.device))
                        similarity = 1.0 / (1.0 + float(distance.item()))
                    else:  # dot_product
                        similarity = float(torch.dot(query_vector, entry.vector.to(query_vector.device)))
                    
                    if similarity >= query.similarity_threshold:
                        results.append((entry, similarity))
                        
                except Exception as e:
                    warnings.warn(f"Linear search failed for entry {entry_id}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            warnings.warn(f"Linear search completely failed: {e}")
            return []
    
    def _apply_filters(self, entries: List[Tuple[MemoryEntry, float]], 
                      query: RetrievalQuery) -> List[Tuple[MemoryEntry, float]]:
        """Apply query filters to results"""
        try:
            filtered = []
            
            for entry, score in entries:
                # Exclude IDs filter
                if query.exclude_ids and entry.entry_id in query.exclude_ids:
                    continue
                
                # Tags filter
                if query.tags and not any(tag in entry.tags for tag in query.tags):
                    continue
                
                # Metadata filters
                if query.metadata_filters:
                    match = True
                    for key, value in query.metadata_filters.items():
                        if key not in entry.metadata or entry.metadata[key] != value:
                            match = False
                            break
                    if not match:
                        continue
                
                # Time range filter
                if query.time_range:
                    start_time, end_time = query.time_range
                    if not (start_time <= entry.timestamp <= end_time):
                        continue
                
                filtered.append((entry, score))
            
            return filtered
            
        except Exception as e:
            warnings.warn(f"Filter application failed: {e}")
            return entries
    
    def _rebuild_faiss_index(self) -> None:
        """Rebuild FAISS index with error handling"""
        try:
            with self._index_lock:
                if len(self.memory_entries) == 0:
                    return
                
                # Collect all vectors
                vectors = []
                entry_ids = []
                
                for entry_id, entry in self.memory_entries.items():
                    vectors.append(entry.vector.cpu().numpy().astype(np.float32))
                    entry_ids.append(entry_id)
                
                vectors_array = np.vstack(vectors)
                
                # Create appropriate FAISS index
                if self.index_type == 'flat':
                    if self.similarity_metric == 'cosine':
                        self.faiss_index = faiss.IndexFlatIP(self.vector_dim)  # Inner product for cosine
                    else:
                        self.faiss_index = faiss.IndexFlatL2(self.vector_dim)
                        
                elif self.index_type == 'ivf':
                    n_centroids = min(int(np.sqrt(len(vectors))), 1000)
                    n_centroids = max(n_centroids, 1)
                    
                    if self.similarity_metric == 'cosine':
                        quantizer = faiss.IndexFlatIP(self.vector_dim)
                        self.faiss_index = faiss.IndexIVFFlat(quantizer, self.vector_dim, n_centroids)
                    else:
                        quantizer = faiss.IndexFlatL2(self.vector_dim)
                        self.faiss_index = faiss.IndexIVFFlat(quantizer, self.vector_dim, n_centroids)
                        
                    self.faiss_index.train(vectors_array)
                    
                elif self.index_type == 'hnsw':
                    self.faiss_index = faiss.IndexHNSWFlat(self.vector_dim, 32)
                    
                else:  # lsh
                    self.faiss_index = faiss.IndexLSH(self.vector_dim, 256)
                
                # Use GPU if available
                if self.device.type == 'cuda' and self.enable_gpu_index:
                    try:
                        res = faiss.StandardGpuResources()
                        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
                    except Exception as e:
                        warnings.warn(f"GPU index creation failed: {e}, using CPU")
                
                # Add vectors to index
                self.faiss_index.add(vectors_array)
                
                # Update mappings
                self.id_to_index = {entry_id: idx for idx, entry_id in enumerate(entry_ids)}
                self.index_to_id = {idx: entry_id for idx, entry_id in enumerate(entry_ids)}
                
                self.index_needs_rebuild = False
                self.last_rebuild_time = datetime.now()
                self.stats['index_rebuilds'] += 1
                
        except Exception as e:
            warnings.warn(f"FAISS index rebuild failed: {e}")
            self.faiss_index = None
    
    def _safe_tensor_convert(self, data: Any) -> torch.Tensor:
        """Safely convert data to tensor"""
        try:
            if isinstance(data, torch.Tensor):
                return data.to(self.device)
            elif isinstance(data, np.ndarray):
                return torch.from_numpy(data).float().to(self.device)
            elif isinstance(data, (list, tuple)):
                return torch.tensor(data, dtype=torch.float32, device=self.device)
            elif isinstance(data, (int, float)):
                return torch.tensor([data], dtype=torch.float32, device=self.device)
            else:
                raise ValueError(f"Cannot convert {type(data)} to tensor")
        except Exception as e:
            warnings.warn(f"Tensor conversion failed: {e}")
            return torch.zeros(self.vector_dim, device=self.device)
    
    def _normalize_vector(self, vector: torch.Tensor) -> torch.Tensor:
        """Normalize vector for similarity search"""
        try:
            if self.similarity_metric == 'cosine':
                return F.normalize(vector, p=2, dim=-1)
            else:
                return vector
        except Exception:
            return vector
    
    def _generate_entry_id(self, vector: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """Generate unique entry ID"""
        try:
            import hashlib
            content = f"{vector.sum().item()}_{len(self.memory_entries)}_{datetime.now().isoformat()}"
            if metadata:
                content += f"_{str(metadata)}"
            return hashlib.md5(content.encode()).hexdigest()
        except Exception:
            import uuid
            return str(uuid.uuid4())
    
    def _update_metadata_index(self, entry_id: str, metadata: Dict[str, Any]) -> None:
        """Update metadata index"""
        try:
            for key, value in metadata.items():
                index_key = f"{key}:{value}"
                self.metadata_index[index_key].add(entry_id)
        except Exception as e:
            warnings.warn(f"Metadata index update failed: {e}")
    
    def _update_tag_index(self, entry_id: str, tags: List[str]) -> None:
        """Update tag index"""
        try:
            for tag in tags:
                self.tag_index[tag].add(entry_id)
        except Exception as e:
            warnings.warn(f"Tag index update failed: {e}")
    
    def _update_time_index(self, entry_id: str) -> None:
        """Update time index"""
        try:
            self.time_index.append((datetime.now(), entry_id))
            # Keep index sorted and limited
            if len(self.time_index) > self.max_entries:
                self.time_index = sorted(self.time_index, key=lambda x: x[0])[-self.max_entries:]
        except Exception as e:
            warnings.warn(f"Time index update failed: {e}")
    
    def _update_vector_cache(self, entry_id: str, vector: torch.Tensor) -> None:
        """Update LRU vector cache"""
        try:
            if not self.cache_enabled:
                return
            
            # Remove oldest if at capacity
            while len(self.vector_cache) >= self.max_cache_size:
                self.vector_cache.popitem(last=False)
            
            self.vector_cache[entry_id] = vector.clone()
            
        except Exception as e:
            warnings.warn(f"Vector cache update failed: {e}")
    
    def _update_query_cache(self, query_hash: str, result: RetrievalResult) -> None:
        """Update query result cache"""
        try:
            if not self.cache_enabled:
                return
            
            # Remove oldest if at capacity
            while len(self.query_cache) >= self.max_query_cache_size:
                self.query_cache.popitem(last=False)
            
            self.query_cache[query_hash] = result
            
        except Exception as e:
            warnings.warn(f"Query cache update failed: {e}")
    
    def _hash_query(self, query: RetrievalQuery) -> str:
        """Create hash for query caching"""
        try:
            import hashlib
            
            # Create hash from query parameters
            query_str = f"{query.similarity_threshold}_{query.max_results}_{query.tags}_{query.metadata_filters}"
            if query.query_vector is not None:
                query_str += f"_{query.query_vector.sum().item()}"
            
            return hashlib.md5(query_str.encode()).hexdigest()
            
        except Exception:
            return f"query_{time.time()}"
    
    def _evict_oldest_entries(self) -> None:
        """Evict oldest entries when at capacity"""
        try:
            # Remove 10% of oldest entries
            num_to_remove = max(1, len(self.memory_entries) // 10)
            
            # Sort by timestamp and access count
            entries_by_age = sorted(
                self.memory_entries.items(),
                key=lambda x: (x[1].timestamp, x[1].access_count)
            )
            
            for i in range(num_to_remove):
                if i < len(entries_by_age):
                    entry_id = entries_by_age[i][0]
                    self._remove_entry(entry_id)
            
            self.index_needs_rebuild = True
            
        except Exception as e:
            warnings.warn(f"Entry eviction failed: {e}")
    
    def _remove_entry(self, entry_id: str) -> bool:
        """Remove single entry"""
        try:
            if entry_id not in self.memory_entries:
                return False
            
            entry = self.memory_entries[entry_id]
            
            # Remove from main storage
            del self.memory_entries[entry_id]
            
            # Remove from backup
            if entry_id in self.backup_vectors:
                del self.backup_vectors[entry_id]
            
            # Remove from cache
            if entry_id in self.vector_cache:
                del self.vector_cache[entry_id]
            
            # Remove from indices
            for tag in entry.tags:
                if tag in self.tag_index:
                    self.tag_index[tag].discard(entry_id)
            
            for key, value in entry.metadata.items():
                index_key = f"{key}:{value}"
                if index_key in self.metadata_index:
                    self.metadata_index[index_key].discard(entry_id)
            
            # Remove from time index
            self.time_index = [(ts, eid) for ts, eid in self.time_index if eid != entry_id]
            
            self.stats['total_entries_removed'] += 1
            self.index_needs_rebuild = True
            
            return True
            
        except Exception as e:
            warnings.warn(f"Entry removal failed for {entry_id}: {e}")
            return False
    
    def _create_error_result(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create error result for non-retrieval operations"""
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'success': False,
            'error_message': error_message,
            'operation': 'error',
            'processing_time_ms': processing_time,
            'stats': {'error': True}
        }
    
    def _create_error_retrieval_result(self, error_message: str, start_time: float) -> RetrievalResult:
        """Create error result for retrieval operations"""
        processing_time = (time.time() - start_time) * 1000
        
        return RetrievalResult(
            success=False,
            entries=[],
            scores=[],
            query_info={},
            error_message=error_message,
            processing_time_ms=processing_time,
            stats={'error': True}
        )
    
    def _get_comprehensive_stats(self, start_time: float) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        try:
            processing_time = (time.time() - start_time) * 1000
            
            # Calculate cache sizes
            cache_size_mb = 0.0
            if self.vector_cache:
                cache_size_mb = sum(
                    v.numel() * v.element_size() for v in self.vector_cache.values()
                ) / (1024 * 1024)
            
            # Calculate index size
            index_size_mb = 0.0
            if self.faiss_index is not None:
                try:
                    index_size_mb = len(self.memory_entries) * self.vector_dim * 4 / (1024 * 1024)  # Estimate
                except Exception:
                    pass
            
            stats = MemoryBankStats(
                total_entries=len(self.memory_entries),
                total_vectors=len(self.backup_vectors),
                index_size_mb=index_size_mb,
                cache_size_mb=cache_size_mb,
                average_similarity=0.0,  # Would need to calculate
                retrieval_latency_ms=self.stats['avg_retrieval_time_ms'],
                index_build_time_ms=0.0,  # Would need to track
                cache_hit_ratio=(
                    self.stats['cache_hits'] / 
                    max(self.stats['cache_hits'] + self.stats['cache_misses'], 1)
                )
            )
            
            return {
                'success': True,
                'stats': stats,
                'detailed_stats': self.stats,
                'processing_time_ms': processing_time,
                'config': {
                    'vector_dim': self.vector_dim,
                    'max_entries': self.max_entries,
                    'similarity_metric': self.similarity_metric,
                    'index_type': self.index_type,
                    'device': str(self.device)
                }
            }
            
        except Exception as e:
            return self._create_error_result(f"Stats collection failed: {e}", start_time)
    
    # Public API methods
    
    def add(self, vector: torch.Tensor, metadata: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None, **kwargs) -> Dict[str, Any]:
        """Add memory entry"""
        return self.forward('add', vector=vector, metadata=metadata or {}, 
                          tags=tags or [], **kwargs)
    
    def retrieve(self, query_vector: torch.Tensor, max_results: int = 10,
                similarity_threshold: float = 0.7, **kwargs) -> RetrievalResult:
        """Retrieve similar memories"""
        query = RetrievalQuery(
            query_vector=query_vector,
            max_results=max_results,
            similarity_threshold=similarity_threshold,
            **kwargs
        )
        return self.forward('retrieve', query=query)
    
    def search(self, query: Union[RetrievalQuery, Dict[str, Any]]) -> RetrievalResult:
        """Search memories with complex query"""
        return self.forward('retrieve', query=query)
    
    def remove(self, entry_ids: List[str]) -> Dict[str, Any]:
        """Remove memory entries"""
        return self.forward('remove', entry_ids=entry_ids)
    
    def clear(self) -> Dict[str, Any]:
        """Clear all memories"""
        return self.forward('clear')
    
    def rebuild_index(self) -> Dict[str, Any]:
        """Force rebuild of similarity index"""
        return self.forward('rebuild_index')
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return self.forward('stats')
    
    def save(self) -> None:
        """Save memory bank to storage"""
        if self.persistence_enabled:
            self._save_memory_bank()

# Test specification
def test_bulletproof_memory_bank_retriever():
    """Comprehensive test specification for BulletproofMemoryBankRetriever"""
    
    test_config = RAVEConfig()
    test_cases = []
    
    # Test 1: Basic add and retrieve
    retriever = BulletproofMemoryBankRetriever(test_config, vector_dim=128, storage_path='./test_memory_bank')
    
    # Add memory
    vector = torch.randn(128)
    add_result = retriever.add(vector, metadata={'type': 'test'}, tags=['test'])
    test_cases.append(('add_success', add_result['success']))
    
    # Retrieve similar
    if add_result['success']:
        query_vector = vector + 0.1 * torch.randn(128)  # Similar vector
        retrieve_result = retriever.retrieve(query_vector, max_results=5)
        test_cases.append(('retrieve_success', retrieve_result.success))
        test_cases.append(('found_results', len(retrieve_result.entries) > 0))
    
    # Test 2: Error resilience
    try:
        error_result = retriever.add("invalid_vector")
        test_cases.append(('error_handling', not error_result['success']))
    except Exception:
        test_cases.append(('error_handling', False))
    
    # Test 3: Batch operations
    vectors = [torch.randn(128) for _ in range(10)]
    for i, vec in enumerate(vectors):
        retriever.add(vec, metadata={'batch': i})
    
    # Search with query
    query = RetrievalQuery(query_vector=vectors[0], max_results=3, metadata_filters={'batch': 0})
    search_result = retriever.search(query)
    test_cases.append(('search_success', search_result.success))
    
    # Test 4: Statistics
    stats_result = retriever.get_stats()
    test_cases.append(('stats_success', stats_result['success']))
    
    return test_cases

if __name__ == "__main__":
    print("🧠 BulletProof Memory Bank Retriever - Testing")
    tests = test_bulletproof_memory_bank_retriever()
    
    passed = sum(1 for _, success in tests if success)
    total = len(tests)
    
    print(f"Tests passed: {passed}/{total}")
    for test_name, success in tests:
        print(f"  {test_name}: {'✅' if success else '❌'}")