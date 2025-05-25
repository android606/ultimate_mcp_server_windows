"""
Working Memory Dashboard API
Provides real-time working memory management and optimization endpoints for the UMS Explorer.
"""

import asyncio
import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel


@dataclass
class WorkingMemoryItem:
    """Enhanced memory item with working memory specific metadata."""
    memory_id: str
    content: str
    memory_type: str
    memory_level: str
    importance: int
    confidence: float
    created_at: float
    last_accessed_at: Optional[float]
    access_count: int
    workflow_id: Optional[str]
    
    # Working memory specific fields
    temperature: float = 0.0  # Activity level (0-100)
    priority: str = "medium"  # critical, high, medium, low
    access_frequency: float = 0.0  # Normalized access frequency
    retention_score: float = 0.0  # How likely to remain in working memory
    added_at: float = 0.0  # When added to working memory


@dataclass
class WorkingMemoryStats:
    """Working memory statistics and metrics."""
    active_count: int
    capacity: int
    pressure: float  # 0-100%
    temperature: float  # Average activity level
    focus_score: float  # 0-100%
    efficiency: float  # 0-100%
    avg_retention_time: float
    total_accesses: int
    last_updated: float


@dataclass
class OptimizationSuggestion:
    """Memory optimization suggestion."""
    id: str
    title: str
    description: str
    priority: str  # high, medium, low
    impact: str  # High, Medium, Low
    icon: str
    action: str
    confidence: float = 0.0
    estimated_improvement: Dict[str, float] = None


class WorkingMemoryRequest(BaseModel):
    memory_id: str


class OptimizationRequest(BaseModel):
    suggestion_id: str


class FocusModeRequest(BaseModel):
    mode: str  # normal, deep, creative, analytical, maintenance
    retention_time: Optional[int] = None
    max_working_memory: Optional[int] = None


class WorkingMemoryManager:
    """Core working memory management and optimization logic."""
    
    def __init__(self, db_path: str = "storage/unified_agent_memory.db"):
        self.db_path = db_path
        self.active_memories: Dict[str, WorkingMemoryItem] = {}
        self.capacity = 7  # Miller's rule: 7±2
        self.focus_mode = "normal"
        self.retention_time = 30  # minutes
        self.connected_clients: List[WebSocket] = []
        
    def get_db_connection(self):
        """Get database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def calculate_memory_temperature(self, memory: Dict) -> float:
        """Calculate memory temperature based on access patterns."""
        now = time.time()
        last_access = memory.get('last_accessed_at', memory.get('created_at', now))
        access_count = memory.get('access_count', 0)
        
        # Recency component (decreases over time)
        time_since_access = now - last_access
        recency_score = max(0, 100 - (time_since_access / 3600) * 10)  # Decreases over hours
        
        # Frequency component
        frequency_score = min(100, access_count * 10)
        
        # Weighted combination
        temperature = recency_score * 0.7 + frequency_score * 0.3
        return round(temperature)
    
    def calculate_memory_priority(self, memory: Dict) -> str:
        """Calculate memory priority level."""
        importance = memory.get('importance', 1)
        if importance >= 9:
            return 'critical'
        elif importance >= 7:
            return 'high'
        elif importance >= 5:
            return 'medium'
        else:
            return 'low'
    
    def calculate_access_frequency(self, memory: Dict) -> float:
        """Calculate normalized access frequency."""
        access_count = memory.get('access_count', 0)
        return min(10, access_count / 5)  # Normalized to 0-10 scale
    
    def calculate_retention_score(self, memory: Dict) -> float:
        """Calculate how likely memory should remain in working memory."""
        importance = memory.get('importance', 1)
        confidence = memory.get('confidence', 0.5)
        access_count = memory.get('access_count', 0)
        
        score = (importance * 0.4 + confidence * 100 * 0.3 + min(access_count * 10, 100) * 0.3) / 10
        return round(score, 2)
    
    def enhance_memory_for_working_memory(self, memory: Dict) -> WorkingMemoryItem:
        """Convert database memory to enhanced working memory item."""
        return WorkingMemoryItem(
            memory_id=memory['memory_id'],
            content=memory['content'],
            memory_type=memory['memory_type'],
            memory_level=memory['memory_level'],
            importance=memory['importance'],
            confidence=memory.get('confidence', 0.5),
            created_at=memory['created_at'],
            last_accessed_at=memory.get('last_accessed_at'),
            access_count=memory.get('access_count', 0),
            workflow_id=memory.get('workflow_id'),
            temperature=self.calculate_memory_temperature(memory),
            priority=self.calculate_memory_priority(memory),
            access_frequency=self.calculate_access_frequency(memory),
            retention_score=self.calculate_retention_score(memory),
            added_at=time.time()
        )
    
    def calculate_focus_score(self) -> float:
        """Calculate current focus score based on working memory coherence."""
        if not self.active_memories:
            return 100.0
        
        memories = list(self.active_memories.values())
        
        # Calculate average importance
        avg_importance = sum(m.importance for m in memories) / len(memories)
        
        # Calculate diversity penalty
        type_variety = len(set(m.memory_type for m in memories))
        level_variety = len(set(m.memory_level for m in memories))
        
        # Lower variety = higher focus
        variety_penalty = (type_variety + level_variety) * 5
        importance_bonus = avg_importance * 10
        
        focus_score = max(0, min(100, importance_bonus - variety_penalty + 20))
        return round(focus_score, 1)
    
    def calculate_efficiency(self) -> float:
        """Calculate working memory efficiency."""
        if not self.active_memories:
            return 100.0
        
        memories = list(self.active_memories.values())
        
        # Average temperature (activity level)
        avg_temperature = sum(m.temperature for m in memories) / len(memories)
        
        # Utilization rate
        utilization = (len(memories) / self.capacity) * 100
        
        # Optimal utilization is around 70%
        optimal_utilization = 100 - abs(utilization - 70) if abs(utilization - 70) < 30 else 70
        
        efficiency = (avg_temperature * 0.6 + optimal_utilization * 0.4)
        return round(efficiency)
    
    def get_working_memory_stats(self) -> WorkingMemoryStats:
        """Get current working memory statistics."""
        memories = list(self.active_memories.values())
        
        return WorkingMemoryStats(
            active_count=len(memories),
            capacity=self.capacity,
            pressure=round((len(memories) / self.capacity) * 100),
            temperature=round(sum(m.temperature for m in memories) / len(memories)) if memories else 0,
            focus_score=self.calculate_focus_score(),
            efficiency=self.calculate_efficiency(),
            avg_retention_time=round(sum(m.retention_score for m in memories) / len(memories)) if memories else 0,
            total_accesses=sum(m.access_count for m in memories),
            last_updated=time.time()
        )
    
    def generate_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on current state."""
        suggestions = []
        stats = self.get_working_memory_stats()
        memories = list(self.active_memories.values())
        
        # High pressure suggestion
        if stats.pressure > 80:
            suggestions.append(OptimizationSuggestion(
                id="reduce-pressure",
                title="Reduce Memory Pressure",
                description="Working memory is near capacity. Consider removing lower priority items.",
                priority="high",
                impact="High",
                icon="alert-triangle",
                action="Auto-Remove",
                confidence=0.9,
                estimated_improvement={"pressure": -20, "efficiency": 15}
            ))
        
        # Cold memories suggestion
        cold_memories = [m for m in memories if m.temperature < 30]
        if cold_memories:
            suggestions.append(OptimizationSuggestion(
                id="remove-cold",
                title="Remove Stale Memories",
                description=f"{len(cold_memories)} memories haven't been accessed recently.",
                priority="medium",
                impact="Medium",
                icon="snowflake",
                action="Clear Stale",
                confidence=0.8,
                estimated_improvement={"temperature": 15, "efficiency": 10}
            ))
        
        # Low focus suggestion
        if stats.focus_score < 50:
            suggestions.append(OptimizationSuggestion(
                id="improve-focus",
                title="Improve Focus",
                description="Working memory contains diverse, unrelated items. Consider focusing on a single task.",
                priority="medium",
                impact="High",
                icon="target",
                action="Focus Mode",
                confidence=0.7,
                estimated_improvement={"focus_score": 30, "efficiency": 20}
            ))
        
        # Underutilization suggestion
        if stats.active_count < self.capacity / 2:
            suggestions.append(OptimizationSuggestion(
                id="add-related",
                title="Add Related Memories",
                description="Working memory has capacity for more relevant items.",
                priority="low",
                impact="Medium",
                icon="plus-circle",
                action="Add Related",
                confidence=0.6,
                estimated_improvement={"efficiency": 10, "focus_score": 5}
            ))
        
        return suggestions
    
    async def load_initial_working_memory(self) -> List[WorkingMemoryItem]:
        """Load initial working memory with high-importance memories."""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Get high-importance or working-level memories
            cursor.execute("""
                SELECT * FROM memories 
                WHERE memory_level = 'working' OR importance >= 8
                ORDER BY created_at DESC, importance DESC
                LIMIT ?
            """, (self.capacity,))
            
            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                enhanced_memory = self.enhance_memory_for_working_memory(memory_dict)
                memories.append(enhanced_memory)
                self.active_memories[enhanced_memory.memory_id] = enhanced_memory
            
            return memories
            
        finally:
            conn.close()
    
    async def add_to_working_memory(self, memory_id: str) -> bool:
        """Add a memory to working memory."""
        if len(self.active_memories) >= self.capacity:
            return False
        
        if memory_id in self.active_memories:
            return False
        
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM memories WHERE memory_id = ?", (memory_id,))
            row = cursor.fetchone()
            
            if not row:
                return False
            
            memory_dict = dict(row)
            enhanced_memory = self.enhance_memory_for_working_memory(memory_dict)
            self.active_memories[memory_id] = enhanced_memory
            
            # Broadcast update to connected clients
            await self.broadcast_update()
            
            return True
            
        finally:
            conn.close()
    
    async def remove_from_working_memory(self, memory_id: str) -> bool:
        """Remove a memory from working memory."""
        if memory_id not in self.active_memories:
            return False
        
        del self.active_memories[memory_id]
        
        # Broadcast update to connected clients
        await self.broadcast_update()
        
        return True
    
    async def clear_working_memory(self):
        """Clear all working memory."""
        self.active_memories.clear()
        await self.broadcast_update()
    
    async def apply_focus_mode(self, mode: str, retention_time: Optional[int] = None, max_memory: Optional[int] = None):
        """Apply focus mode settings."""
        mode_settings = {
            'deep': {'capacity': 5, 'retention': 60},
            'creative': {'capacity': 9, 'retention': 45},
            'analytical': {'capacity': 6, 'retention': 90},
            'maintenance': {'capacity': 3, 'retention': 20},
            'normal': {'capacity': 7, 'retention': 30}
        }
        
        settings = mode_settings.get(mode, mode_settings['normal'])
        
        self.focus_mode = mode
        self.capacity = max_memory or settings['capacity']
        self.retention_time = retention_time or settings['retention']
        
        # If we're over capacity, remove lowest priority memories
        if len(self.active_memories) > self.capacity:
            memories_by_priority = sorted(
                self.active_memories.values(),
                key=lambda m: (m.importance, m.retention_score),
                reverse=True
            )
            
            # Keep only the top memories
            to_keep = memories_by_priority[:self.capacity]
            self.active_memories = {m.memory_id: m for m in to_keep}
        
        await self.broadcast_update()
    
    async def auto_optimize(self) -> List[str]:
        """Apply automatic optimizations."""
        applied_optimizations = []
        suggestions = self.generate_optimization_suggestions()
        
        for suggestion in suggestions:
            if suggestion.priority in ['medium', 'low'] and suggestion.confidence > 0.7:
                success = await self.apply_optimization(suggestion.id)
                if success:
                    applied_optimizations.append(suggestion.title)
        
        return applied_optimizations
    
    async def apply_optimization(self, suggestion_id: str) -> bool:
        """Apply a specific optimization."""
        memories = list(self.active_memories.values())
        
        if suggestion_id == "reduce-pressure":
            # Remove lowest priority memories
            low_priority = [m for m in memories if m.priority == 'low']
            for memory in low_priority[:2]:
                await self.remove_from_working_memory(memory.memory_id)
            return True
            
        elif suggestion_id == "remove-cold":
            # Remove cold memories
            cold_memories = [m for m in memories if m.temperature < 30]
            for memory in cold_memories[:3]:
                await self.remove_from_working_memory(memory.memory_id)
            return True
            
        elif suggestion_id == "improve-focus":
            # Switch to deep focus mode
            await self.apply_focus_mode('deep')
            return True
            
        elif suggestion_id == "add-related":
            # Add related memories
            await self.add_related_memories()
            return True
        
        return False
    
    async def add_related_memories(self):
        """Add memories related to current working memory."""
        if not self.active_memories or len(self.active_memories) >= self.capacity:
            return
        
        current_types = set(m.memory_type for m in self.active_memories.values())
        current_workflows = set(m.workflow_id for m in self.active_memories.values() if m.workflow_id)
        
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Find related memories
            placeholders = ','.join('?' * len(current_types)) if current_types else "''"
            workflow_placeholders = ','.join('?' * len(current_workflows)) if current_workflows else "''"
            
            query = f"""
                SELECT * FROM memories 
                WHERE memory_id NOT IN ({','.join('?' * len(self.active_memories))})
                AND (memory_type IN ({placeholders}) OR workflow_id IN ({workflow_placeholders}))
                AND importance >= 6
                ORDER BY importance DESC
                LIMIT ?
            """
            
            params = (
                list(self.active_memories.keys()) + 
                list(current_types) + 
                list(current_workflows) + 
                [self.capacity - len(self.active_memories)]
            )
            
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                memory_dict = dict(row)
                enhanced_memory = self.enhance_memory_for_working_memory(memory_dict)
                self.active_memories[enhanced_memory.memory_id] = enhanced_memory
                
                if len(self.active_memories) >= self.capacity:
                    break
            
        finally:
            conn.close()
        
        await self.broadcast_update()
    
    def get_memory_pool(self, search: str = "", filter_type: str = "", limit: int = 50) -> List[Dict]:
        """Get available memory pool for working memory."""
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Build query
            where_conditions = ["memory_id NOT IN ({})".format(','.join('?' * len(self.active_memories)))]
            params = list(self.active_memories.keys())
            
            if search:
                where_conditions.append("(content LIKE ? OR memory_type LIKE ?)")
                params.extend([f"%{search}%", f"%{search}%"])
            
            if filter_type == "high":
                where_conditions.append("importance >= 8")
            elif filter_type == "recent":
                day_ago = time.time() - 86400
                where_conditions.append("created_at > ?")
                params.append(day_ago)
            elif filter_type == "related" and self.active_memories:
                current_types = set(m.memory_type for m in self.active_memories.values())
                current_workflows = set(m.workflow_id for m in self.active_memories.values() if m.workflow_id)
                
                if current_types or current_workflows:
                    type_placeholders = ','.join('?' * len(current_types)) if current_types else "''"
                    workflow_placeholders = ','.join('?' * len(current_workflows)) if current_workflows else "''"
                    where_conditions.append(f"(memory_type IN ({type_placeholders}) OR workflow_id IN ({workflow_placeholders}))")
                    params.extend(list(current_types) + list(current_workflows))
            
            query = f"""
                SELECT * FROM memories 
                WHERE {' AND '.join(where_conditions)}
                ORDER BY importance DESC
                LIMIT ?
            """
            params.append(limit)
            
            cursor.execute(query, params)
            
            memories = []
            for row in cursor.fetchall():
                memory_dict = dict(row)
                memory_dict['access_frequency'] = self.calculate_access_frequency(memory_dict)
                memories.append(memory_dict)
            
            return memories
            
        finally:
            conn.close()
    
    def generate_heatmap_data(self, timeframe: str = "24h") -> List[Dict]:
        """Generate memory activity heatmap data."""
        now = time.time()
        intervals = []
        
        # Configure timeframe
        timeframe_config = {
            '1h': {'seconds': 300, 'count': 12},      # 5 minute intervals
            '6h': {'seconds': 1800, 'count': 12},     # 30 minute intervals
            '24h': {'seconds': 3600, 'count': 24},    # 1 hour intervals
            '7d': {'seconds': 86400, 'count': 7}      # 1 day intervals
        }
        
        config = timeframe_config.get(timeframe, timeframe_config['24h'])
        interval_seconds = config['seconds']
        interval_count = config['count']
        
        conn = self.get_db_connection()
        try:
            cursor = conn.cursor()
            
            for i in range(interval_count):
                interval_start = now - (interval_count - i) * interval_seconds
                interval_end = interval_start + interval_seconds
                
                # Count activities in this interval
                cursor.execute("""
                    SELECT COUNT(*) as activity_count 
                    FROM memories 
                    WHERE created_at >= ? AND created_at <= ?
                """, (interval_start, interval_end))
                
                activity_count = cursor.fetchone()[0]
                
                intervals.append({
                    'time': interval_start,
                    'activity': activity_count,
                    'intensity': min(1.0, activity_count / 10)  # Normalize to 0-1
                })
            
            return intervals
            
        finally:
            conn.close()
    
    async def register_client(self, websocket: WebSocket):
        """Register a WebSocket client for real-time updates."""
        self.connected_clients.append(websocket)
    
    async def unregister_client(self, websocket: WebSocket):
        """Unregister a WebSocket client."""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)
    
    async def broadcast_update(self):
        """Broadcast working memory update to all connected clients."""
        if not self.connected_clients:
            return
        
        update_data = {
            'type': 'working_memory_update',
            'stats': asdict(self.get_working_memory_stats()),
            'active_memories': [asdict(m) for m in self.active_memories.values()],
            'suggestions': [asdict(s) for s in self.generate_optimization_suggestions()],
            'timestamp': time.time()
        }
        
        # Send to all connected clients
        disconnected_clients = []
        for client in self.connected_clients:
            try:
                await client.send_text(json.dumps(update_data))
            except Exception:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            await self.unregister_client(client)


# Global working memory manager instance
working_memory_manager = WorkingMemoryManager()


def setup_working_memory_routes(app: FastAPI):
    """Setup working memory API routes."""
    
    @app.get("/api/working-memory/status")
    async def get_working_memory_status():
        """Get current working memory status and statistics."""
        try:
            stats = working_memory_manager.get_working_memory_stats()
            active_memories = [asdict(m) for m in working_memory_manager.active_memories.values()]
            suggestions = [asdict(s) for s in working_memory_manager.generate_optimization_suggestions()]
            
            return {
                'status': 'connected',
                'stats': asdict(stats),
                'active_memories': active_memories,
                'suggestions': suggestions,
                'focus_mode': working_memory_manager.focus_mode,
                'capacity': working_memory_manager.capacity,
                'retention_time': working_memory_manager.retention_time
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.post("/api/working-memory/initialize")
    async def initialize_working_memory():
        """Initialize working memory with default high-importance memories."""
        try:
            memories = await working_memory_manager.load_initial_working_memory()
            stats = working_memory_manager.get_working_memory_stats()
            
            return {
                'success': True,
                'message': f'Initialized with {len(memories)} memories',
                'stats': asdict(stats),
                'active_memories': [asdict(m) for m in memories]
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.post("/api/working-memory/add")
    async def add_memory_to_working_memory(request: WorkingMemoryRequest):
        """Add a memory to working memory."""
        try:
            success = await working_memory_manager.add_to_working_memory(request.memory_id)
            
            if success:
                stats = working_memory_manager.get_working_memory_stats()
                return {
                    'success': True,
                    'message': 'Memory added to working memory',
                    'stats': asdict(stats)
                }
            else:
                return {
                    'success': False,
                    'message': 'Could not add memory (capacity reached or already exists)'
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.post("/api/working-memory/remove")
    async def remove_memory_from_working_memory(request: WorkingMemoryRequest):
        """Remove a memory from working memory."""
        try:
            success = await working_memory_manager.remove_from_working_memory(request.memory_id)
            
            if success:
                stats = working_memory_manager.get_working_memory_stats()
                return {
                    'success': True,
                    'message': 'Memory removed from working memory',
                    'stats': asdict(stats)
                }
            else:
                return {
                    'success': False,
                    'message': 'Memory not found in working memory'
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.post("/api/working-memory/clear")
    async def clear_working_memory():
        """Clear all working memory."""
        try:
            await working_memory_manager.clear_working_memory()
            stats = working_memory_manager.get_working_memory_stats()
            
            return {
                'success': True,
                'message': 'Working memory cleared',
                'stats': asdict(stats)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.post("/api/working-memory/focus-mode")
    async def set_focus_mode(request: FocusModeRequest):
        """Set focus mode and apply related optimizations."""
        try:
            await working_memory_manager.apply_focus_mode(
                request.mode,
                request.retention_time,
                request.max_working_memory
            )
            
            stats = working_memory_manager.get_working_memory_stats()
            
            return {
                'success': True,
                'message': f'Applied {request.mode} focus mode',
                'focus_mode': working_memory_manager.focus_mode,
                'capacity': working_memory_manager.capacity,
                'retention_time': working_memory_manager.retention_time,
                'stats': asdict(stats)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.post("/api/working-memory/optimize")
    async def optimize_working_memory():
        """Apply automatic working memory optimizations."""
        try:
            applied = await working_memory_manager.auto_optimize()
            stats = working_memory_manager.get_working_memory_stats()
            
            return {
                'success': True,
                'message': f'Applied {len(applied)} optimizations',
                'optimizations_applied': applied,
                'stats': asdict(stats)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.post("/api/working-memory/apply-suggestion")
    async def apply_optimization_suggestion(request: OptimizationRequest):
        """Apply a specific optimization suggestion."""
        try:
            success = await working_memory_manager.apply_optimization(request.suggestion_id)
            
            if success:
                stats = working_memory_manager.get_working_memory_stats()
                suggestions = [asdict(s) for s in working_memory_manager.generate_optimization_suggestions()]
                
                return {
                    'success': True,
                    'message': 'Optimization applied successfully',
                    'stats': asdict(stats),
                    'suggestions': suggestions
                }
            else:
                return {
                    'success': False,
                    'message': 'Could not apply optimization'
                }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.get("/api/working-memory/pool")
    async def get_memory_pool(
        search: str = "",
        filter_type: str = "",  # "", "high", "recent", "related"
        limit: int = 50
    ):
        """Get available memory pool for working memory."""
        try:
            memories = working_memory_manager.get_memory_pool(search, filter_type, limit)
            
            return {
                'success': True,
                'memories': memories,
                'count': len(memories)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.get("/api/working-memory/heatmap")
    async def get_memory_heatmap(timeframe: str = "24h"):
        """Get memory activity heatmap data."""
        try:
            heatmap_data = working_memory_manager.generate_heatmap_data(timeframe)
            
            return {
                'success': True,
                'timeframe': timeframe,
                'data': heatmap_data
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e
    
    @app.websocket("/ws/working-memory")
    async def working_memory_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time working memory updates."""
        await websocket.accept()
        await working_memory_manager.register_client(websocket)
        
        try:
            # Send initial data
            initial_data = {
                'type': 'initial_data',
                'stats': asdict(working_memory_manager.get_working_memory_stats()),
                'active_memories': [asdict(m) for m in working_memory_manager.active_memories.values()],
                'suggestions': [asdict(s) for s in working_memory_manager.generate_optimization_suggestions()],
                'focus_mode': working_memory_manager.focus_mode,
                'capacity': working_memory_manager.capacity
            }
            await websocket.send_text(json.dumps(initial_data))
            
            # Keep connection alive and handle messages
            while True:
                try:
                    # Wait for messages from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle different message types
                    if message.get('type') == 'ping':
                        await websocket.send_text(json.dumps({'type': 'pong'}))
                    
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    break
                    
        finally:
            await working_memory_manager.unregister_client(websocket)


# Background task to periodically update working memory
async def working_memory_background_task():
    """Background task for periodic working memory updates."""
    while True:
        try:
            # Update temperatures and stats periodically
            for memory in working_memory_manager.active_memories.values():
                # Recalculate temperature based on current time
                memory.temperature = working_memory_manager.calculate_memory_temperature(asdict(memory))
            
            # Broadcast updates if there are connected clients
            if working_memory_manager.connected_clients:
                await working_memory_manager.broadcast_update()
            
            # Wait 30 seconds before next update
            await asyncio.sleep(30)
            
        except Exception as e:
            print(f"Background task error: {e}")
            await asyncio.sleep(60)  # Wait longer if there's an error


def start_background_tasks(app: FastAPI):
    """Start background tasks for working memory management."""
    
    @app.on_event("startup")
    async def startup_event():
        # Start background task
        asyncio.create_task(working_memory_background_task())
        
        # Initialize working memory with default data
        try:
            await working_memory_manager.load_initial_working_memory()
            print("✅ Working memory initialized successfully")
        except Exception as e:
            print(f"⚠️ Could not initialize working memory: {e}") 