"""Hierarchical Semantic Encoder.

Encodes natural language into a hierarchical semantic tree structure
with information-theoretic importance scores.

This is the core of the research contribution:
1. Extract semantic hierarchy (intent → entities → attributes → details)
2. Calculate importance scores analytically
3. Enable principled compression by pruning low-importance branches
"""

import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .groq_client import GroqClient
from .hierarchical_signal import (
    SemanticNode, 
    SemanticLevel, 
    HierarchicalSignal,
    InformationCalculator,
    CompressionBoundCalculator
)
from .tokenization import TiktokenTokenizer


HIERARCHICAL_ENCODER_PROMPT = """You are a semantic parser that extracts hierarchical structure from messages.

Extract the following hierarchy:

LEVEL 0 - INTENT (one of: DELEGATE, ANALYZE, GENERATE, EVALUATE, TRANSFORM, QUERY, RESPOND, REPORT)

LEVEL 1 - ENTITIES (who/what is involved):
- actors: people, teams, organizations
- objects: tickets, documents, systems
- actions: what needs to be done

LEVEL 2 - ATTRIBUTES (properties):
- urgency: priority level
- quantities: numbers, amounts, percentages
- timeframes: deadlines, durations
- status: current state

LEVEL 3 - DETAILS (context):
- causes: root causes, reasons
- effects: impacts, consequences
- conditions: requirements, constraints

Output JSON in this exact format:
{
  "intent": "DELEGATE",
  "entities": {
    "actors": [{"name": "...", "role": "..."}],
    "objects": [{"id": "...", "type": "...", "description": "..."}],
    "actions": [{"verb": "...", "target": "..."}]
  },
  "attributes": {
    "urgency": "critical|high|medium|low",
    "quantities": [{"value": "...", "unit": "...", "context": "..."}],
    "timeframes": [{"duration": "...", "deadline": "..."}],
    "status": "..."
  },
  "details": {
    "causes": ["..."],
    "effects": ["..."],
    "conditions": ["..."]
  }
}

Be thorough - extract ALL entities, quantities, and details from the message.
Output ONLY valid JSON."""


@dataclass
class HierarchicalEncodingResult:
    """Result of hierarchical encoding."""
    signal: HierarchicalSignal
    raw_extraction: Dict[str, Any]
    encoding_tokens: int
    theoretical_bound: float
    efficiency: float


class HierarchicalEncoder:
    """Encodes natural language into hierarchical semantic trees.
    
    The encoding process:
    1. LLM extracts structured hierarchy from text
    2. Build semantic tree with proper levels
    3. Calculate entropy for each node
    4. Calculate importance scores analytically
    5. Compute theoretical compression bounds
    """
    
    def __init__(self, groq_client: GroqClient):
        self.client = groq_client
        self.tokenizer = TiktokenTokenizer()
        self.info_calc = InformationCalculator()
        self.bound_calc = CompressionBoundCalculator(self.info_calc)
    
    async def encode(self, text: str) -> HierarchicalEncodingResult:
        """Encode text into hierarchical semantic signal.
        
        Args:
            text: Natural language input
            
        Returns:
            HierarchicalEncodingResult with signal and metrics
        """
        # Update corpus for information calculations
        self.info_calc.update_corpus(text)
        
        # Extract hierarchy via LLM
        raw = await self._extract_hierarchy(text)
        
        # Build semantic tree
        root = self._build_tree(raw, text)
        
        # Calculate importance scores
        self._calculate_importance(root, text)
        
        # Create signal
        original_tokens = self.tokenizer.count_tokens(text)
        signal = HierarchicalSignal(
            root=root,
            original_tokens=original_tokens,
            original_text=text
        )
        
        # Calculate theoretical bound
        signal_json = signal.to_json()
        encoding_tokens = self.tokenizer.count_tokens(signal_json)
        
        # Assume 80% similarity target for bound calculation
        theoretical_bound = self.bound_calc.minimum_bits_for_similarity(signal, 0.8)
        efficiency = self.bound_calc.compression_efficiency(
            signal, 
            signal.total_entropy(),
            0.8
        )
        
        return HierarchicalEncodingResult(
            signal=signal,
            raw_extraction=raw,
            encoding_tokens=encoding_tokens,
            theoretical_bound=theoretical_bound,
            efficiency=efficiency
        )
    
    async def _extract_hierarchy(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract hierarchical structure."""
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": HIERARCHICAL_ENCODER_PROMPT},
                {"role": "user", "content": text}
            ],
            json_mode=True,
            temperature=0.0
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Fallback structure
            return {
                "intent": "QUERY",
                "entities": {"actors": [], "objects": [], "actions": []},
                "attributes": {"urgency": "medium", "quantities": [], "timeframes": [], "status": "unknown"},
                "details": {"causes": [], "effects": [], "conditions": []}
            }
    
    def _build_tree(self, raw: Dict[str, Any], original_text: str) -> SemanticNode:
        """Build semantic tree from extracted hierarchy."""
        # Level 0: Intent
        intent = raw.get("intent", "QUERY")
        root = SemanticNode(
            content=intent,
            level=SemanticLevel.INTENT,
            node_type="intent",
            entropy=self.info_calc.content_entropy(intent)
        )
        
        # Level 1: Entities
        entities = raw.get("entities", {})
        
        # Actors
        for actor in entities.get("actors", []):
            if isinstance(actor, dict):
                content = f"{actor.get('name', '')} ({actor.get('role', '')})"
            else:
                content = str(actor)
            
            node = SemanticNode(
                content=content,
                level=SemanticLevel.ENTITIES,
                node_type="actor",
                entropy=self.info_calc.content_entropy(content),
                metadata=actor if isinstance(actor, dict) else {"name": actor}
            )
            root.children.append(node)
        
        # Objects
        for obj in entities.get("objects", []):
            if isinstance(obj, dict):
                content = f"{obj.get('type', 'object')}: {obj.get('id', '')} - {obj.get('description', '')}"
            else:
                content = str(obj)
            
            node = SemanticNode(
                content=content,
                level=SemanticLevel.ENTITIES,
                node_type="object",
                entropy=self.info_calc.content_entropy(content),
                metadata=obj if isinstance(obj, dict) else {"id": obj}
            )
            root.children.append(node)
        
        # Actions
        for action in entities.get("actions", []):
            if isinstance(action, dict):
                content = f"{action.get('verb', '')} {action.get('target', '')}"
            else:
                content = str(action)
            
            node = SemanticNode(
                content=content,
                level=SemanticLevel.ENTITIES,
                node_type="action",
                entropy=self.info_calc.content_entropy(content),
                metadata=action if isinstance(action, dict) else {"verb": action}
            )
            root.children.append(node)
        
        # Level 2: Attributes
        attributes = raw.get("attributes", {})
        
        # Urgency
        urgency = attributes.get("urgency", "medium")
        if urgency:
            node = SemanticNode(
                content=f"urgency: {urgency}",
                level=SemanticLevel.ATTRIBUTES,
                node_type="urgency",
                entropy=self.info_calc.content_entropy(urgency),
                metadata={"value": urgency}
            )
            root.children.append(node)
        
        # Quantities
        for qty in attributes.get("quantities", []):
            if isinstance(qty, dict):
                content = f"{qty.get('value', '')} {qty.get('unit', '')} ({qty.get('context', '')})"
            else:
                content = str(qty)
            
            node = SemanticNode(
                content=content,
                level=SemanticLevel.ATTRIBUTES,
                node_type="quantity",
                entropy=self.info_calc.content_entropy(content),
                metadata=qty if isinstance(qty, dict) else {"value": qty}
            )
            root.children.append(node)
        
        # Timeframes
        for tf in attributes.get("timeframes", []):
            if isinstance(tf, dict):
                content = f"timeframe: {tf.get('duration', '')} {tf.get('deadline', '')}"
            else:
                content = str(tf)
            
            node = SemanticNode(
                content=content,
                level=SemanticLevel.ATTRIBUTES,
                node_type="timeframe",
                entropy=self.info_calc.content_entropy(content),
                metadata=tf if isinstance(tf, dict) else {"duration": tf}
            )
            root.children.append(node)
        
        # Level 3: Details
        details = raw.get("details", {})
        
        for cause in details.get("causes", []):
            node = SemanticNode(
                content=f"cause: {cause}",
                level=SemanticLevel.DETAILS,
                node_type="cause",
                entropy=self.info_calc.content_entropy(str(cause))
            )
            root.children.append(node)
        
        for effect in details.get("effects", []):
            node = SemanticNode(
                content=f"effect: {effect}",
                level=SemanticLevel.DETAILS,
                node_type="effect",
                entropy=self.info_calc.content_entropy(str(effect))
            )
            root.children.append(node)
        
        for condition in details.get("conditions", []):
            node = SemanticNode(
                content=f"condition: {condition}",
                level=SemanticLevel.DETAILS,
                node_type="condition",
                entropy=self.info_calc.content_entropy(str(condition))
            )
            root.children.append(node)
        
        return root
    
    def _calculate_importance(self, root: SemanticNode, original_text: str) -> None:
        """Calculate importance scores for all nodes."""
        for node in root.flatten():
            node.importance = self.info_calc.importance_score(node, original_text)


class HierarchicalCompressor:
    """Compress hierarchical signals by pruning low-importance nodes.
    
    This implements principled compression:
    1. Sort nodes by importance
    2. Prune lowest importance nodes until bit budget met
    3. Track semantic preservation
    """
    
    def __init__(self):
        self.tokenizer = TiktokenTokenizer()
    
    def compress(
        self, 
        signal: HierarchicalSignal, 
        target_bits: Optional[float] = None,
        target_ratio: Optional[float] = None,
        min_similarity: float = 0.7
    ) -> HierarchicalSignal:
        """Compress signal by pruning low-importance nodes.
        
        Args:
            signal: Original hierarchical signal
            target_bits: Target bit budget (optional)
            target_ratio: Target compression ratio (optional)
            min_similarity: Minimum semantic similarity to preserve
            
        Returns:
            Compressed signal with pruned nodes
        """
        if target_ratio:
            target_bits = signal.total_entropy() * target_ratio
        
        if not target_bits:
            target_bits = signal.total_entropy() * 0.5  # Default 50% compression
        
        # Get all nodes sorted by importance (ascending - prune least important first)
        all_nodes = signal.root.flatten()
        
        # Never prune the root (intent)
        prunable = [n for n in all_nodes if n.level != SemanticLevel.INTENT]
        prunable.sort(key=lambda n: n.importance)
        
        # Prune until we hit target
        current_bits = signal.total_entropy()
        pruned_nodes = set()
        
        for node in prunable:
            if current_bits <= target_bits:
                break
            
            # Check if pruning this would drop below min importance threshold
            remaining_importance = sum(
                n.importance for n in all_nodes 
                if n not in pruned_nodes and n != node
            )
            total_importance = sum(n.importance for n in all_nodes)
            
            if remaining_importance / total_importance < min_similarity:
                break  # Don't prune - would lose too much
            
            pruned_nodes.add(node)
            current_bits -= node.entropy
        
        # Build compressed tree
        compressed_root = self._rebuild_without_pruned(signal.root, pruned_nodes)
        
        return HierarchicalSignal(
            root=compressed_root,
            original_tokens=signal.original_tokens,
            original_text=signal.original_text
        )
    
    def _rebuild_without_pruned(
        self, 
        node: SemanticNode, 
        pruned: set
    ) -> SemanticNode:
        """Rebuild tree excluding pruned nodes."""
        new_children = []
        for child in node.children:
            if child not in pruned:
                new_child = self._rebuild_without_pruned(child, pruned)
                new_children.append(new_child)
        
        return SemanticNode(
            content=node.content,
            level=node.level,
            node_type=node.node_type,
            importance=node.importance,
            entropy=node.entropy,
            children=new_children,
            metadata=node.metadata
        )
