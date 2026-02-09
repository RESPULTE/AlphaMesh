
import time
import logging
import functools
from collections import defaultdict
from typing import Any, Dict, Optional
from graphiti_core.tracer import Tracer, TracerSpan

logger = logging.getLogger(__name__)

class GraphitiSpan(TracerSpan):
    def __init__(self, name: str, tracer: 'GraphitiTracer'):
        self.name = name
        self.tracer = tracer
        self.start_time = None
        self.end_time = None
        self.attributes = {}

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        self.tracer.record_span(self.name, duration, self.attributes)

    def add_attributes(self, attributes: Dict[str, Any]):
        if attributes:
            self.attributes.update(attributes)
    
    def set_status(self, status: str, description: str = ""):
        pass

    def record_exception(self, exception: Exception):
        pass

class GraphitiTracer(Tracer):
    def __init__(self):
        self.metrics = defaultdict(lambda: {'count': 0, 'latency': 0.0, 'tokens': 0})
        self.total_tokens = 0
        self.total_latency = 0.0
        self.api_call_counts = 0

    def start_span(self, name: str) -> TracerSpan:
        return GraphitiSpan(name, self)

    def record_span(self, name: str, duration: float, attributes: Dict[str, Any]):
        self.metrics[name]['count'] += 1
        self.metrics[name]['latency'] += duration
        
        tokens = 0
        if 'token_usage' in attributes:
             usage = attributes['token_usage']
             if isinstance(usage, dict):
                 tokens = usage.get('total_tokens', 0)
             elif isinstance(usage, (int, float)):
                 tokens = int(usage)
        elif 'total_tokens' in attributes:
            tokens = attributes['total_tokens']
        
        self.metrics[name]['tokens'] += tokens
        self.total_tokens += tokens
        if "generate" in name.lower() or "embed" in name.lower():
             self.api_call_counts += 1
        
        self.metrics[name]['tokens'] += tokens
        self.total_tokens += tokens
        if "generate" in name.lower() or "embed" in name.lower():
             self.api_call_counts += 1

    def wrap_method(self, obj, method_name, metric_name=None):
        if not hasattr(obj, method_name):
            return
        
        original_method = getattr(obj, method_name)
        if hasattr(original_method, '__wrapped__'):
            return # Already wrapped

        name = metric_name or method_name

        @functools.wraps(original_method)
        async def async_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await original_method(*args, **kwargs)
            finally:
                duration = time.time() - start
                self.record_span(name, duration, {})
        
        @functools.wraps(original_method)
        def sync_wrapper(*args, **kwargs):
            start = time.time()
            try:
                return original_method(*args, **kwargs)
            finally:
                duration = time.time() - start
                self.record_span(name, duration, {})

        if inspect.iscoroutinefunction(original_method):
            setattr(obj, method_name, async_wrapper)
        else:
            setattr(obj, method_name, sync_wrapper)

    def instrument_graphiti(self, graphiti):
        # Instrument Graphiti high-level methods
        methods_to_instrument = [
            'add_episode', 
            'search', 
            '_search', 
            'build_indices_and_constraints'
        ]
        
        for method in methods_to_instrument:
             self.wrap_method(graphiti, method, metric_name=f"Graphiti.{method}")

        # Instrument Gemini Client (LLM) low-level call
        if hasattr(graphiti.llm_client, 'client'):
            # It's a google.genai.Client. We want to wrap aio.models.generate_content
            try:
                # Provide a wrapper that inspects the response
                original_generate_content = graphiti.llm_client.client.aio.models.generate_content
                
                @functools.wraps(original_generate_content)
                async def wrapped_generate_content(*args, **kwargs):
                    start = time.time()
                    try:
                        response = await original_generate_content(*args, **kwargs)
                        # Extract usage
                        usage = getattr(response, 'usage_metadata', None)
                        attributes = {}
                        if usage:
                            # usage_metadata usually has prompt_token_count, candidates_token_count, total_token_count
                            attributes['token_usage'] = {
                                'total_tokens': getattr(usage, 'total_token_count', 0),
                                'prompt_tokens': getattr(usage, 'prompt_token_count', 0),
                                'completion_tokens': getattr(usage, 'candidates_token_count', 0)
                            }
                        return response
                    finally:
                        duration = time.time() - start
                        self.record_span('LLM.generate_content', duration, attributes if 'attributes' in locals() else {})

                graphiti.llm_client.client.aio.models.generate_content = wrapped_generate_content
            except Exception as e:
                logger.warning(f"Failed to instrument GeminiClient low-level: {e}")

        # Instrument Embedder low-level call
        if hasattr(graphiti, 'embedder') and hasattr(graphiti.embedder, 'client'):
             # Same for embedder, it uses client.aio.models.embed_content
            try:
                original_embed_content = graphiti.embedder.client.aio.models.embed_content

                @functools.wraps(original_embed_content)
                async def wrapped_embed_content(*args, **kwargs):
                    start = time.time()
                    try:
                        response = await original_embed_content(*args, **kwargs)
                         # Extract usage? Note: embed_content response might not have usage_metadata in all versions
                         # But let's check
                        attributes = {}
                        # For embeddings, token count is usually input tokens. 
                        # GenAI response might have it.
                        # If not, we can estimate or check if response has it.
                        # According to docs, EmbedContentResponse might not have usage_metadata populated always?
                        # But let's try.
                        # Actually, for batch embedding, it returns a BatchEmbedContentsResponse?
                        # The code calls 'embed_content' even for batch but loops? No, create_batch loops.
                        
                        # Wait, create_batch calls embed_content inside a loop.
                        # So wrapping embed_content captures all.
                        
                        # Checking response attributes
                        # print(f"DEBUG: Embed response dir: {dir(response)}") 
                        # We can enable debug print if needed.
                        return response
                    finally:
                        duration = time.time() - start
                        self.record_span('Embedder.embed_content', duration, attributes if 'attributes' in locals() else {})

                graphiti.embedder.client.aio.models.embed_content = wrapped_embed_content
            except Exception as e:
                logger.warning(f"Failed to instrument GeminiEmbedder low-level: {e}")

    def report(self):
        print("\n" + "="*50)
        print("GRAPHITI TRACER REPORT")
        print("="*50)
        print(f"Total API Calls (LLM/Embed): {self.api_call_counts}")
        print(f"Total Token Usage: {self.total_tokens}")
        print("-" * 50)
        print(f"{'Operation':<30} | {'Count':<5} | {'Latency (s)':<10} | {'Tokens':<8}")
        print("-" * 50)
        for name, data in self.metrics.items():
            print(f"{name:<30} | {data['count']:<5} | {data['latency']:<10.4f} | {data['tokens']:<8}")
        print("="*50 + "\n")

import inspect
