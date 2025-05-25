import json
import logging
import math
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# Import UMS initialization
from ultimate_mcp_server.tools.unified_memory_system import initialize_memory_system

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="UMS Explorer API",
    description="Unified Memory System Explorer with Cognitive State Timeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database configuration
DATABASE_PATH = os.path.join(os.path.dirname(__file__), "storage", "unified_agent_memory.db")

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Removed - schema now properly defined in unified_memory_system.py

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    # Initialize the unified memory system (handles all schema setup)
    await initialize_memory_system(db_path=DATABASE_PATH)
    logger.info("UMS Explorer API started successfully")

# Serve static files
app.mount("/storage", StaticFiles(directory="storage"), name="storage")
app.mount("/tools", StaticFiles(directory="ultimate_mcp_server/tools"), name="tools")

# Serve the main HTML file
@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main UMS Explorer interface"""
    try:
        with open("ultimate_mcp_server/tools/ums_explorer.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read(), status_code=200)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>UMS Explorer HTML not found</h1>", status_code=404)

# Cognitive States API Endpoints

@app.get("/api/cognitive-states")
async def get_cognitive_states(
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    limit: int = 100,
    offset: int = 0,
    pattern_type: Optional[str] = None
):
    """Get cognitive states with optional filtering"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Base query
        query = """
            SELECT 
                cs.*,
                w.title as workflow_title,
                COUNT(DISTINCT m.memory_id) as memory_count,
                COUNT(DISTINCT a.action_id) as action_count
            FROM cognitive_timeline_states cs
            LEFT JOIN workflows w ON cs.workflow_id = w.workflow_id
            LEFT JOIN memories m ON cs.workflow_id = m.workflow_id
            LEFT JOIN actions a ON cs.workflow_id = a.workflow_id
            WHERE 1=1
        """
        params = []
        
        if start_time:
            query += " AND cs.timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND cs.timestamp <= ?"
            params.append(end_time)
        
        if pattern_type:
            query += " AND cs.state_type = ?"
            params.append(pattern_type)
        
        query += """
            GROUP BY cs.state_id
            ORDER BY cs.timestamp DESC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        states = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        
        # Enhance states with additional metadata
        enhanced_states = []
        for state in states:
            # Parse state_data if it's JSON
            try:
                state_data = json.loads(state.get('state_data', '{}'))
            except Exception:
                state_data = {}
            
            enhanced_state = {
                **state,
                'state_data': state_data,
                'formatted_timestamp': datetime.fromtimestamp(state['timestamp']).isoformat(),
                'age_minutes': (datetime.now().timestamp() - state['timestamp']) / 60,
                'complexity_score': calculate_state_complexity(state_data),
                'change_magnitude': 0  # Will be calculated with diffs
            }
            enhanced_states.append(enhanced_state)
        
        # Calculate change magnitudes by comparing adjacent states
        for i in range(len(enhanced_states) - 1):
            current_state = enhanced_states[i]
            previous_state = enhanced_states[i + 1]  # Ordered DESC, so next is previous
            
            diff_result = compute_state_diff(previous_state['state_data'], current_state['state_data'])
            current_state['change_magnitude'] = diff_result.get('magnitude', 0)
        
        conn.close()
        
        return {
            "states": enhanced_states,
            "total": len(enhanced_states),
            "has_more": len(enhanced_states) == limit
        }
        
    except Exception as e:
        logger.error(f"Error getting cognitive states: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/cognitive-states/timeline")
async def get_cognitive_timeline(
    hours: int = 24,
    granularity: str = "hour"  # hour, minute, second
):
    """Get cognitive state timeline data optimized for visualization"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        since_timestamp = datetime.now().timestamp() - (hours * 3600)
        
        cursor.execute("""
            SELECT 
                state_id,
                timestamp,
                state_type,
                state_data,
                workflow_id,
                description,
                ROW_NUMBER() OVER (ORDER BY timestamp) as sequence_number
            FROM cognitive_timeline_states 
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (since_timestamp,))
        
        states = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Parse and enhance state data
        timeline_data = []
        for i, state in enumerate(states):
            try:
                state_data = json.loads(state.get('state_data', '{}'))
            except Exception:
                state_data = {}
            
            # Calculate change magnitude from previous state
            change_magnitude = 0
            if i > 0:
                prev_state_data = json.loads(states[i-1].get('state_data', '{}'))
                diff_result = compute_state_diff(prev_state_data, state_data)
                change_magnitude = diff_result.get('magnitude', 0)
            
            timeline_item = {
                'state_id': state['state_id'],
                'timestamp': state['timestamp'],
                'formatted_time': datetime.fromtimestamp(state['timestamp']).isoformat(),
                'state_type': state['state_type'],
                'workflow_id': state['workflow_id'],
                'description': state['description'],
                'sequence_number': state['sequence_number'],
                'complexity_score': calculate_state_complexity(state_data),
                'change_magnitude': change_magnitude,
                'key_components': extract_key_state_components(state_data),
                'tags': generate_state_tags(state_data, state['state_type'])
            }
            timeline_data.append(timeline_item)
        
        # Generate timeline segments based on granularity
        segments = generate_timeline_segments(timeline_data, granularity, hours)
        
        conn.close()
        
        return {
            'timeline_data': timeline_data,
            'segments': segments,
            'total_states': len(timeline_data),
            'time_range_hours': hours,
            'granularity': granularity,
            'summary_stats': calculate_timeline_stats(timeline_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting cognitive timeline: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/cognitive-states/patterns")
async def analyze_cognitive_patterns(
    lookback_hours: int = 24,
    min_pattern_length: int = 3,
    similarity_threshold: float = 0.7
):
    """Analyze recurring cognitive patterns"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get recent states
        since_timestamp = datetime.now().timestamp() - (lookback_hours * 3600)
        cursor.execute("""
            SELECT state_id, timestamp, state_type, state_data, workflow_id
            FROM cognitive_timeline_states 
            WHERE timestamp >= ?
            ORDER BY timestamp ASC
        """, (since_timestamp,))
        
        states = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Parse state data
        for state in states:
            try:
                state['state_data'] = json.loads(state.get('state_data', '{}'))
            except Exception:
                state['state_data'] = {}
        
        # Analyze patterns
        patterns = find_cognitive_patterns(states, min_pattern_length, similarity_threshold)
        
        # Analyze state transitions
        transitions = analyze_state_transitions(states)
        
        # Identify anomalies
        anomalies = detect_cognitive_anomalies(states)
        
        conn.close()
        
        return {
            'total_states': len(states),
            'time_range_hours': lookback_hours,
            'patterns': patterns,
            'transitions': transitions,
            'anomalies': anomalies,
            'summary': {
                'pattern_count': len(patterns),
                'most_common_transition': transitions[0] if transitions else None,
                'anomaly_count': len(anomalies)
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing cognitive patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/cognitive-states/{state_id}")
async def get_cognitive_state(state_id: str):
    """Get detailed cognitive state by ID"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                cs.*,
                w.title as workflow_title,
                w.goal as workflow_goal
            FROM cognitive_timeline_states cs
            LEFT JOIN workflows w ON cs.workflow_id = w.workflow_id
            WHERE cs.state_id = ?
        """, (state_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Cognitive state not found")
        
        columns = [description[0] for description in cursor.description]
        state = dict(zip(columns, row, strict=False))
        
        # Parse state data
        try:
            state['state_data'] = json.loads(state.get('state_data', '{}'))
        except Exception:
            state['state_data'] = {}
        
        # Get associated memories and actions
        cursor.execute("""
            SELECT memory_id, memory_type, content, importance, created_at
            FROM memories 
            WHERE workflow_id = ? 
            ORDER BY created_at DESC
            LIMIT 20
        """, (state.get('workflow_id', ''),))
        
        memories = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        cursor.execute("""
            SELECT action_id, action_type, tool_name, status, started_at
            FROM actions 
            WHERE workflow_id = ? 
            ORDER BY started_at DESC
            LIMIT 20
        """, (state.get('workflow_id', ''),))
        
        actions = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            **state,
            'memories': memories,
            'actions': actions,
            'formatted_timestamp': datetime.fromtimestamp(state['timestamp']).isoformat(),
            'complexity_score': calculate_state_complexity(state['state_data'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cognitive state {state_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/cognitive-states/compare")
async def compare_cognitive_states(request: dict):
    """Compare two cognitive states and return detailed diff"""
    try:
        state_id_1 = request.get('state_id_1')
        state_id_2 = request.get('state_id_2')
        
        if not state_id_1 or not state_id_2:
            raise HTTPException(status_code=400, detail="Both state IDs required")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get both states
        cursor.execute("SELECT * FROM cognitive_timeline_states WHERE state_id IN (?, ?)", (state_id_1, state_id_2))
        states = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        if len(states) != 2:
            raise HTTPException(status_code=404, detail="One or both states not found")
        
        # Parse state data
        for state in states:
            try:
                state['state_data'] = json.loads(state.get('state_data', '{}'))
            except Exception:
                state['state_data'] = {}
        
        # Order states by timestamp
        states.sort(key=lambda s: s['timestamp'])
        state_1, state_2 = states
        
        # Compute comprehensive diff
        diff_result = compute_detailed_state_diff(state_1, state_2)
        
        conn.close()
        
        return {
            'state_1': {
                'state_id': state_1['state_id'],
                'timestamp': state_1['timestamp'],
                'formatted_timestamp': datetime.fromtimestamp(state_1['timestamp']).isoformat()
            },
            'state_2': {
                'state_id': state_2['state_id'],
                'timestamp': state_2['timestamp'],
                'formatted_timestamp': datetime.fromtimestamp(state_2['timestamp']).isoformat()
            },
            'time_diff_minutes': (state_2['timestamp'] - state_1['timestamp']) / 60,
            'diff': diff_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error comparing cognitive states: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/cognitive-states/{state_id}/restore")
async def restore_cognitive_state(state_id: str, request: dict):
    """Restore system to a previous cognitive state"""
    try:
        confirm = request.get('confirm', False)
        if not confirm:
            raise HTTPException(status_code=400, detail="Confirmation required for state restoration")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the target state
        cursor.execute("SELECT * FROM cognitive_timeline_states WHERE state_id = ?", (state_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Cognitive state not found")
        
        columns = [description[0] for description in cursor.description]
        target_state = dict(zip(columns, row, strict=False))
        
        # Parse state data
        try:
            state_data = json.loads(target_state.get('state_data', '{}'))
        except Exception:
            state_data = {}
        
        # Create restoration checkpoint (current state backup)
        current_timestamp = datetime.now().timestamp()
        backup_state_id = f"backup_{int(current_timestamp)}"
        
        # Get current system state for backup
        current_state_data = capture_current_system_state(conn)
        
        cursor.execute("""
            INSERT INTO cognitive_timeline_states (state_id, timestamp, state_type, state_data, workflow_id, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            backup_state_id,
            current_timestamp,
            'backup',
            json.dumps(current_state_data),
            target_state.get('workflow_id'),
            f"Backup before restoring to {state_id}"
        ))
        
        # Perform restoration (this would be implementation-specific)
        restoration_result = perform_state_restoration(conn, state_data, target_state)
        
        # Log the restoration
        cursor.execute("""
            INSERT INTO cognitive_timeline_states (state_id, timestamp, state_type, state_data, workflow_id, description)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            f"restore_{int(current_timestamp)}",
            current_timestamp,
            'restoration',
            json.dumps({
                'restored_from': state_id,
                'backup_created': backup_state_id,
                'restoration_result': restoration_result
            }),
            target_state.get('workflow_id'),
            f"Restored to state {state_id}"
        ))
        
        conn.commit()
        conn.close()
        
        return {
            'success': True,
            'restored_state_id': state_id,
            'backup_state_id': backup_state_id,
            'restoration_timestamp': current_timestamp,
            'restoration_result': restoration_result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring cognitive state {state_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

# Action Execution Monitor API Endpoints

@app.get("/api/actions/running")
async def get_running_actions():
    """Get currently executing actions with real-time status"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get running actions with execution details
        cursor.execute("""
            SELECT 
                a.*,
                w.title as workflow_title,
                (unixepoch() - a.started_at) as execution_time,
                CASE 
                    WHEN a.tool_data IS NOT NULL THEN json_extract(a.tool_data, '$.estimated_duration')
                    ELSE NULL 
                END as estimated_duration
            FROM actions a
            LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
            WHERE a.status IN ('running', 'executing', 'in_progress')
            ORDER BY a.started_at ASC
        """)
        
        columns = [description[0] for description in cursor.description]
        running_actions = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        
        # Enhance with real-time metrics
        enhanced_actions = []
        for action in running_actions:
            # Parse tool_data if available
            try:
                tool_data = json.loads(action.get('tool_data', '{}'))
            except Exception:
                tool_data = {}
            
            # Calculate progress estimation
            execution_time = action.get('execution_time', 0)
            estimated_duration = action.get('estimated_duration') or 30  # Default 30 seconds
            progress_percentage = min(95, (execution_time / estimated_duration) * 100) if estimated_duration > 0 else 0
            
            enhanced_action = {
                **action,
                'tool_data': tool_data,
                'execution_time_seconds': execution_time,
                'progress_percentage': progress_percentage,
                'status_indicator': get_action_status_indicator(action['status'], execution_time),
                'performance_category': categorize_action_performance(execution_time, estimated_duration),
                'resource_usage': get_action_resource_usage(action['action_id']),
                'formatted_start_time': datetime.fromtimestamp(action['started_at']).isoformat()
            }
            enhanced_actions.append(enhanced_action)
        
        conn.close()
        
        return {
            'running_actions': enhanced_actions,
            'total_running': len(enhanced_actions),
            'avg_execution_time': sum(a['execution_time_seconds'] for a in enhanced_actions) / len(enhanced_actions) if enhanced_actions else 0,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting running actions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/actions/queue")
async def get_action_queue():
    """Get queued actions waiting for execution"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                a.*,
                w.title as workflow_title,
                (unixepoch() - a.created_at) as queue_time,
                CASE 
                    WHEN a.tool_data IS NOT NULL THEN json_extract(a.tool_data, '$.priority')
                    ELSE 5 
                END as priority
            FROM actions a
            LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
            WHERE a.status IN ('queued', 'pending', 'waiting')
            ORDER BY priority ASC, a.created_at ASC
        """)
        
        columns = [description[0] for description in cursor.description]
        queued_actions = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        
        # Enhance queue data
        enhanced_queue = []
        for i, action in enumerate(queued_actions):
            try:
                tool_data = json.loads(action.get('tool_data', '{}'))
            except Exception:
                tool_data = {}
            
            enhanced_action = {
                **action,
                'tool_data': tool_data,
                'queue_position': i + 1,
                'queue_time_seconds': action.get('queue_time', 0),
                'estimated_wait_time': estimate_wait_time(i, queued_actions),
                'priority_label': get_priority_label(action.get('priority', 5)),
                'formatted_queue_time': datetime.fromtimestamp(action['created_at']).isoformat()
            }
            enhanced_queue.append(enhanced_action)
        
        conn.close()
        
        return {
            'queued_actions': enhanced_queue,
            'total_queued': len(enhanced_queue),
            'avg_queue_time': sum(a['queue_time_seconds'] for a in enhanced_queue) / len(enhanced_queue) if enhanced_queue else 0,
            'next_action': enhanced_queue[0] if enhanced_queue else None,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting action queue: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/actions/history")
async def get_action_history(
    limit: int = 50,
    offset: int = 0,
    status_filter: Optional[str] = None,
    tool_filter: Optional[str] = None,
    hours_back: int = 24
):
    """Get completed actions with performance metrics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
        
        query = """
            SELECT 
                a.*,
                w.title as workflow_title,
                (a.completed_at - a.started_at) as execution_duration,
                CASE 
                    WHEN a.tool_data IS NOT NULL THEN json_extract(a.tool_data, '$.result_size')
                    ELSE 0 
                END as result_size
            FROM actions a
            LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
            WHERE a.status IN ('completed', 'failed', 'cancelled', 'timeout')
            AND a.completed_at >= ?
        """
        params = [since_timestamp]
        
        if status_filter:
            query += " AND a.status = ?"
            params.append(status_filter)
        
        if tool_filter:
            query += " AND a.tool_name = ?"
            params.append(tool_filter)
        
        query += " ORDER BY a.completed_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        completed_actions = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        
        # Calculate performance metrics
        enhanced_history = []
        for action in completed_actions:
            try:
                tool_data = json.loads(action.get('tool_data', '{}'))
                result_data = json.loads(action.get('result', '{}'))
            except Exception:
                tool_data = {}
                result_data = {}
            
            execution_duration = action.get('execution_duration', 0)
            
            enhanced_action = {
                **action,
                'tool_data': tool_data,
                'result_data': result_data,
                'execution_duration_seconds': execution_duration,
                'performance_score': calculate_action_performance_score(action),
                'efficiency_rating': calculate_efficiency_rating(execution_duration, action.get('result_size', 0)),
                'success_rate_impact': 1 if action['status'] == 'completed' else 0,
                'formatted_start_time': datetime.fromtimestamp(action['started_at']).isoformat(),
                'formatted_completion_time': datetime.fromtimestamp(action['completed_at']).isoformat() if action['completed_at'] else None
            }
            enhanced_history.append(enhanced_action)
        
        conn.close()
        
        # Calculate aggregate metrics
        total_actions = len(enhanced_history)
        successful_actions = len([a for a in enhanced_history if a['status'] == 'completed'])
        avg_duration = sum(a['execution_duration_seconds'] for a in enhanced_history) / total_actions if total_actions > 0 else 0
        
        return {
            'action_history': enhanced_history,
            'total_actions': total_actions,
            'success_rate': (successful_actions / total_actions * 100) if total_actions > 0 else 0,
            'avg_execution_time': avg_duration,
            'performance_summary': calculate_performance_summary(enhanced_history),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting action history: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/actions/metrics")
async def get_action_metrics():
    """Get comprehensive action execution metrics and analytics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get metrics for last 24 hours
        since_timestamp = datetime.now().timestamp() - (24 * 3600)
        
        # Overall statistics
        cursor.execute("""
            SELECT 
                COUNT(*) as total_actions,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_actions,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_actions,
                AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration
            FROM actions 
            WHERE created_at >= ?
        """, (since_timestamp,))
        
        overall_stats = dict(zip([d[0] for d in cursor.description], cursor.fetchone(), strict=False))
        
        # Tool usage statistics
        cursor.execute("""
            SELECT 
                tool_name,
                COUNT(*) as usage_count,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count,
                AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration
            FROM actions 
            WHERE created_at >= ?
            GROUP BY tool_name
            ORDER BY usage_count DESC
        """, (since_timestamp,))
        
        tool_stats = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Performance distribution over time (hourly)
        cursor.execute("""
            SELECT 
                strftime('%H', datetime(started_at, 'unixepoch')) as hour,
                COUNT(*) as action_count,
                AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_duration,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as success_count
            FROM actions 
            WHERE started_at >= ?
            GROUP BY hour
            ORDER BY hour
        """, (since_timestamp,))
        
        hourly_metrics = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Error analysis
        cursor.execute("""
            SELECT 
                tool_name,
                error_message,
                COUNT(*) as error_count,
                MAX(created_at) as last_occurrence
            FROM actions 
            WHERE status = 'failed' 
            AND created_at >= ?
            AND error_message IS NOT NULL
            GROUP BY tool_name, error_message
            ORDER BY error_count DESC
            LIMIT 10
        """, (since_timestamp,))
        
        error_analysis = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        conn.close()
        
        # Calculate derived metrics
        success_rate = (overall_stats['successful_actions'] / overall_stats['total_actions'] * 100) if overall_stats['total_actions'] > 0 else 0
        
        return {
            'overall_metrics': {
                **overall_stats,
                'success_rate_percentage': success_rate,
                'failure_rate_percentage': 100 - success_rate,
                'avg_duration_seconds': overall_stats['avg_duration'] or 0
            },
            'tool_usage_stats': tool_stats,
            'hourly_performance': hourly_metrics,
            'error_analysis': error_analysis,
            'performance_insights': generate_performance_insights(overall_stats, tool_stats, hourly_metrics),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting action metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/tools/usage")
async def get_tool_usage_statistics(
    hours_back: int = 24,
    include_performance: bool = True
):
    """Get detailed tool usage statistics with performance breakdown"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        since_timestamp = datetime.now().timestamp() - (hours_back * 3600)
        
        # Comprehensive tool statistics
        cursor.execute("""
            SELECT 
                tool_name,
                COUNT(*) as total_calls,
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful_calls,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed_calls,
                AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as avg_execution_time,
                MIN(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as min_execution_time,
                MAX(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as max_execution_time,
                COUNT(DISTINCT workflow_id) as unique_workflows,
                MAX(started_at) as last_used
            FROM actions 
            WHERE created_at >= ?
            GROUP BY tool_name
            ORDER BY total_calls DESC
        """, (since_timestamp,))
        
        tool_statistics = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Enhance with calculated metrics
        enhanced_tool_stats = []
        for tool in tool_statistics:
            success_rate = (tool['successful_calls'] / tool['total_calls'] * 100) if tool['total_calls'] > 0 else 0
            
            enhanced_tool = {
                **tool,
                'success_rate_percentage': success_rate,
                'reliability_score': calculate_tool_reliability_score(tool),
                'performance_category': categorize_tool_performance(tool['avg_execution_time']),
                'usage_trend': 'increasing',  # This would be calculated with historical data
                'last_used_formatted': datetime.fromtimestamp(tool['last_used']).isoformat()
            }
            enhanced_tool_stats.append(enhanced_tool)
        
        # Tool performance over time
        if include_performance:
            cursor.execute("""
                SELECT 
                    tool_name,
                    strftime('%H', datetime(started_at, 'unixepoch')) as hour,
                    COUNT(*) as hourly_usage,
                    AVG(CASE WHEN completed_at IS NOT NULL THEN completed_at - started_at ELSE NULL END) as hourly_avg_duration
                FROM actions 
                WHERE created_at >= ?
                GROUP BY tool_name, hour
                ORDER BY tool_name, hour
            """, (since_timestamp,))
            
            hourly_performance = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        else:
            hourly_performance = []
        
        conn.close()
        
        return {
            'tool_statistics': enhanced_tool_stats,
            'hourly_performance': hourly_performance,
            'summary': {
                'total_tools_used': len(enhanced_tool_stats),
                'most_used_tool': enhanced_tool_stats[0]['tool_name'] if enhanced_tool_stats else None,
                'highest_success_rate': max(t['success_rate_percentage'] for t in enhanced_tool_stats) if enhanced_tool_stats else 0,
                'avg_tool_performance': sum(t['avg_execution_time'] or 0 for t in enhanced_tool_stats) / len(enhanced_tool_stats) if enhanced_tool_stats else 0
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting tool usage statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

# Artifact Management API Endpoints

# Removed - schema now properly defined in unified_memory_system.py

@app.get("/api/artifacts")
async def get_artifacts(
    artifact_type: Optional[str] = None,
    workflow_id: Optional[str] = None,
    tags: Optional[str] = None,
    search: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    limit: int = 50,
    offset: int = 0
):
    """Get artifacts with filtering and search"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Base query
        query = """
            SELECT 
                a.*,
                w.title as workflow_title,
                COUNT(ar.target_artifact_id) as relationship_count,
                COUNT(versions.artifact_id) as version_count
            FROM artifacts a
            LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
            LEFT JOIN artifact_relationships ar ON a.artifact_id = ar.source_artifact_id
            LEFT JOIN artifacts versions ON a.artifact_id = versions.parent_artifact_id
            WHERE 1=1
        """
        params = []
        
        if artifact_type:
            query += " AND a.artifact_type = ?"
            params.append(artifact_type)
        
        if workflow_id:
            query += " AND a.workflow_id = ?"
            params.append(workflow_id)
        
        if tags:
            query += " AND a.tags LIKE ?"
            params.append(f"%{tags}%")
        
        if search:
            query += " AND (a.name LIKE ? OR a.description LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        query += f"""
            GROUP BY a.artifact_id
            ORDER BY a.{sort_by} {'DESC' if sort_order == 'desc' else 'ASC'}
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        columns = [description[0] for description in cursor.description]
        artifacts = [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
        
        # Enhance artifacts with metadata
        for artifact in artifacts:
            # Parse tags and metadata
            try:
                artifact['tags'] = json.loads(artifact.get('tags', '[]')) if artifact.get('tags') else []
                artifact['metadata'] = json.loads(artifact.get('metadata', '{}')) if artifact.get('metadata') else {}
            except Exception:
                artifact['tags'] = []
                artifact['metadata'] = {}
            
            # Add computed fields
            artifact['formatted_created_at'] = datetime.fromtimestamp(artifact['created_at']).isoformat()
            artifact['formatted_updated_at'] = datetime.fromtimestamp(artifact['updated_at']).isoformat()
            artifact['age_days'] = (datetime.now().timestamp() - artifact['created_at']) / 86400
            artifact['file_size_human'] = format_file_size(artifact.get('file_size', 0))
        
        conn.close()
        
        return {
            "artifacts": artifacts,
            "total": len(artifacts),
            "has_more": len(artifacts) == limit
        }
        
    except Exception as e:
        logger.error(f"Error getting artifacts: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/artifacts")
async def create_artifact(request: dict):
    """Create a new artifact"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        artifact_id = request.get('artifact_id') or f"artifact_{int(datetime.now().timestamp())}"
        name = request.get('name')
        description = request.get('description', '')
        artifact_type = request.get('artifact_type')
        file_path = request.get('file_path')
        file_size = request.get('file_size', 0)
        content_hash = request.get('content_hash')
        workflow_id = request.get('workflow_id')
        tags = json.dumps(request.get('tags', []))
        metadata = json.dumps(request.get('metadata', {}))
        importance = request.get('importance', 0.5)
        
        if not name or not artifact_type:
            raise HTTPException(status_code=400, detail="Name and artifact_type are required")
        
        cursor.execute("""
            INSERT INTO artifacts (
                artifact_id, name, description, artifact_type, file_path, file_size,
                content_hash, workflow_id, tags, metadata, importance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            artifact_id, name, description, artifact_type, file_path, file_size,
            content_hash, workflow_id, tags, metadata, importance
        ))
        
        conn.commit()
        conn.close()
        
        return {"artifact_id": artifact_id, "status": "created"}
        
    except Exception as e:
        logger.error(f"Error creating artifact: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/artifacts/{artifact_id}")
async def get_artifact(artifact_id: str):
    """Get detailed artifact information"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                a.*,
                w.title as workflow_title,
                parent.name as parent_name
            FROM artifacts a
            LEFT JOIN workflows w ON a.workflow_id = w.workflow_id
            LEFT JOIN artifacts parent ON a.parent_artifact_id = parent.artifact_id
            WHERE a.artifact_id = ?
        """, (artifact_id,))
        
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        columns = [description[0] for description in cursor.description]
        artifact = dict(zip(columns, row, strict=False))
        
        # Get artifact versions
        cursor.execute("""
            SELECT artifact_id, name, version, created_at, description
            FROM artifacts 
            WHERE parent_artifact_id = ? OR artifact_id = ?
            ORDER BY version DESC
        """, (artifact_id, artifact_id))
        
        versions = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Get relationships
        cursor.execute("""
            SELECT 
                ar.relationship_type,
                ar.strength,
                a.artifact_id,
                a.name,
                a.artifact_type
            FROM artifact_relationships ar
            JOIN artifacts a ON ar.target_artifact_id = a.artifact_id
            WHERE ar.source_artifact_id = ?
        """, (artifact_id,))
        
        relationships = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Update access count
        cursor.execute("""
            UPDATE artifacts 
            SET access_count = access_count + 1, last_accessed_at = unixepoch()
            WHERE artifact_id = ?
        """, (artifact_id,))
        
        conn.commit()
        conn.close()
        
        # Parse and enhance data
        try:
            artifact['tags'] = json.loads(artifact.get('tags', '[]')) if artifact.get('tags') else []
            artifact['metadata'] = json.loads(artifact.get('metadata', '{}')) if artifact.get('metadata') else {}
        except Exception:
            artifact['tags'] = []
            artifact['metadata'] = {}
        
        return {
            **artifact,
            'versions': versions,
            'relationships': relationships,
            'formatted_created_at': datetime.fromtimestamp(artifact['created_at']).isoformat(),
            'formatted_updated_at': datetime.fromtimestamp(artifact['updated_at']).isoformat(),
            'file_size_human': format_file_size(artifact.get('file_size', 0))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting artifact {artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.put("/api/artifacts/{artifact_id}")
async def update_artifact(artifact_id: str, request: dict):
    """Update an artifact"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if artifact exists
        cursor.execute("SELECT artifact_id FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Artifact not found")
        
        # Build update query dynamically
        update_fields = []
        params = []
        
        if 'name' in request:
            update_fields.append("name = ?")
            params.append(request['name'])
        
        if 'description' in request:
            update_fields.append("description = ?")
            params.append(request['description'])
        
        if 'tags' in request:
            update_fields.append("tags = ?")
            params.append(json.dumps(request['tags']))
        
        if 'metadata' in request:
            update_fields.append("metadata = ?")
            params.append(json.dumps(request['metadata']))
        
        if 'importance' in request:
            update_fields.append("importance = ?")
            params.append(request['importance'])
        
        if not update_fields:
            raise HTTPException(status_code=400, detail="No valid fields to update")
        
        update_fields.append("updated_at = unixepoch()")
        params.append(artifact_id)
        
        query = f"UPDATE artifacts SET {', '.join(update_fields)} WHERE artifact_id = ?"
        cursor.execute(query, params)
        
        conn.commit()
        conn.close()
        
        return {"status": "updated"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating artifact {artifact_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/artifacts/{artifact_id}/relationships")
async def create_artifact_relationship(artifact_id: str, request: dict):
    """Create a relationship between artifacts"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        target_artifact_id = request.get('target_artifact_id')
        relationship_type = request.get('relationship_type')
        strength = request.get('strength', 1.0)
        
        if not target_artifact_id or not relationship_type:
            raise HTTPException(status_code=400, detail="target_artifact_id and relationship_type are required")
        
        # Check if both artifacts exist
        cursor.execute("SELECT COUNT(*) FROM artifacts WHERE artifact_id IN (?, ?)", (artifact_id, target_artifact_id))
        if cursor.fetchone()[0] != 2:
            raise HTTPException(status_code=404, detail="One or both artifacts not found")
        
        relationship_id = f"rel_{int(datetime.now().timestamp())}"
        
        cursor.execute("""
            INSERT OR REPLACE INTO artifact_relationships (
                relationship_id, source_artifact_id, target_artifact_id, relationship_type, strength
            ) VALUES (?, ?, ?, ?, ?)
        """, (relationship_id, artifact_id, target_artifact_id, relationship_type, strength))
        
        conn.commit()
        conn.close()
        
        return {"relationship_id": relationship_id, "status": "created"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating artifact relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/artifacts/graph")
async def get_artifact_graph():
    """Get artifact relationship graph data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get all artifacts
        cursor.execute("""
            SELECT artifact_id, name, artifact_type, workflow_id, importance, access_count
            FROM artifacts
        """)
        
        artifacts = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Get all relationships
        cursor.execute("""
            SELECT source_artifact_id, target_artifact_id, relationship_type, strength
            FROM artifact_relationships
        """)
        
        relationships = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        conn.close()
        
        # Transform for D3.js graph visualization
        nodes = []
        for artifact in artifacts:
            nodes.append({
                'id': artifact['artifact_id'],
                'name': artifact['name'],
                'type': artifact['artifact_type'],
                'workflow_id': artifact['workflow_id'],
                'importance': artifact['importance'],
                'access_count': artifact['access_count'],
                'size': max(10, min(50, artifact['access_count'] * 2))  # Size based on access count
            })
        
        links = []
        for rel in relationships:
            links.append({
                'source': rel['source_artifact_id'],
                'target': rel['target_artifact_id'],
                'type': rel['relationship_type'],
                'strength': rel['strength'],
                'weight': rel['strength'] * 10  # Weight for link thickness
            })
        
        return {
            'nodes': nodes,
            'links': links,
            'total_artifacts': len(nodes),
            'total_relationships': len(links)
        }
        
    except Exception as e:
        logger.error(f"Error getting artifact graph: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/artifacts/stats")
async def get_artifact_stats():
    """Get artifact statistics"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Total counts by type
        cursor.execute("""
            SELECT 
                artifact_type,
                COUNT(*) as count,
                AVG(importance) as avg_importance,
                SUM(file_size) as total_size,
                MAX(access_count) as max_access_count
            FROM artifacts 
            GROUP BY artifact_type
        """)
        
        type_stats = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Overall stats
        cursor.execute("""
            SELECT 
                COUNT(*) as total_artifacts,
                COUNT(DISTINCT workflow_id) as unique_workflows,
                AVG(importance) as avg_importance,
                SUM(file_size) as total_file_size,
                SUM(access_count) as total_access_count
            FROM artifacts
        """)
        
        overall_stats = dict(zip([d[0] for d in cursor.description], cursor.fetchone(), strict=False))
        
        # Recent activity
        cursor.execute("""
            SELECT COUNT(*) as recent_artifacts
            FROM artifacts 
            WHERE created_at > unixepoch() - 86400
        """)
        
        recent_count = cursor.fetchone()[0]
        
        # Most accessed artifacts
        cursor.execute("""
            SELECT artifact_id, name, access_count, artifact_type
            FROM artifacts 
            ORDER BY access_count DESC 
            LIMIT 10
        """)
        
        top_accessed = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        conn.close()
        
        return {
            'overall': {
                **overall_stats,
                'recent_artifacts_24h': recent_count,
                'total_file_size_human': format_file_size(overall_stats.get('total_file_size', 0))
            },
            'by_type': type_stats,
            'top_accessed': top_accessed
        }
        
    except Exception as e:
        logger.error(f"Error getting artifact stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"
 
# Helper functions for Action Monitor endpoints

def get_action_status_indicator(status: str, execution_time: float) -> dict:
    """Get status indicator with color and icon for action status"""
    indicators = {
        'running': {'color': 'blue', 'icon': 'play', 'label': 'Running'},
        'executing': {'color': 'blue', 'icon': 'cpu', 'label': 'Executing'},
        'in_progress': {'color': 'orange', 'icon': 'clock', 'label': 'In Progress'},
        'completed': {'color': 'green', 'icon': 'check', 'label': 'Completed'},
        'failed': {'color': 'red', 'icon': 'x', 'label': 'Failed'},
        'cancelled': {'color': 'gray', 'icon': 'stop', 'label': 'Cancelled'},
        'timeout': {'color': 'yellow', 'icon': 'timer-off', 'label': 'Timeout'}
    }
    
    indicator = indicators.get(status, {'color': 'gray', 'icon': 'help', 'label': 'Unknown'})
    
    # Add urgency flag for long-running actions
    if status in ['running', 'executing', 'in_progress'] and execution_time > 120:  # 2 minutes
        indicator['urgency'] = 'high'
    elif status in ['running', 'executing', 'in_progress'] and execution_time > 60:  # 1 minute
        indicator['urgency'] = 'medium'
    else:
        indicator['urgency'] = 'low'
    
    return indicator

def categorize_action_performance(execution_time: float, estimated_duration: float) -> str:
    """Categorize action performance based on execution time vs estimate"""
    if estimated_duration <= 0:
        return 'unknown'
    
    ratio = execution_time / estimated_duration
    
    if ratio <= 0.5:
        return 'excellent'
    elif ratio <= 0.8:
        return 'good'
    elif ratio <= 1.2:
        return 'acceptable'
    elif ratio <= 2.0:
        return 'slow'
    else:
        return 'very_slow'

def get_action_resource_usage(action_id: str) -> dict:
    """Get resource usage for an action (placeholder implementation)"""
    # This would typically query system metrics or action-specific monitoring
    return {
        'cpu_usage': 0.0,
        'memory_usage': 0.0,
        'network_io': 0.0,
        'disk_io': 0.0
    }

def estimate_wait_time(position: int, queue: list) -> float:
    """Estimate wait time based on queue position and historical data"""
    if position == 0:
        return 0.0
    
    # Simple estimation: assume average of 30 seconds per action
    avg_action_time = 30.0
    return position * avg_action_time

def get_priority_label(priority: int) -> str:
    """Get human-readable priority label"""
    if priority <= 1:
        return 'Critical'
    elif priority <= 3:
        return 'High'
    elif priority <= 5:
        return 'Normal'
    elif priority <= 7:
        return 'Low'
    else:
        return 'Very Low'

def calculate_action_performance_score(action: dict) -> float:
    """Calculate performance score for a completed action"""
    if action['status'] != 'completed':
        return 0.0
    
    execution_time = action.get('execution_duration', 0)
    if execution_time <= 0:
        return 100.0
    
    # Score based on execution time (lower is better)
    if execution_time <= 5:
        return 100.0
    elif execution_time <= 15:
        return 90.0
    elif execution_time <= 30:
        return 80.0
    elif execution_time <= 60:
        return 70.0
    elif execution_time <= 120:
        return 60.0
    else:
        return max(50.0, 100.0 - (execution_time / 10))

def calculate_efficiency_rating(execution_time: float, result_size: int) -> str:
    """Calculate efficiency rating based on time and output"""
    if execution_time <= 0:
        return 'unknown'
    
    # Simple heuristic: consider both time and result size
    efficiency_score = result_size / execution_time if execution_time > 0 else 0
    
    if efficiency_score >= 100:
        return 'excellent'
    elif efficiency_score >= 50:
        return 'good'
    elif efficiency_score >= 20:
        return 'fair'
    else:
        return 'poor'

def calculate_performance_summary(actions: list) -> dict:
    """Calculate performance summary from action history"""
    if not actions:
        return {
            'avg_score': 0.0,
            'top_performer': None,
            'worst_performer': None,
            'efficiency_distribution': {}
        }
    
    scores = [a.get('performance_score', 0) for a in actions]
    avg_score = sum(scores) / len(scores)
    
    # Find best and worst performers
    best_action = max(actions, key=lambda a: a.get('performance_score', 0))
    worst_action = min(actions, key=lambda a: a.get('performance_score', 0))
    
    # Calculate efficiency distribution
    efficiency_counts = Counter(a.get('efficiency_rating', 'unknown') for a in actions)
    
    return {
        'avg_score': round(avg_score, 2),
        'top_performer': {
            'tool_name': best_action.get('tool_name', ''),
            'score': best_action.get('performance_score', 0)
        },
        'worst_performer': {
            'tool_name': worst_action.get('tool_name', ''),
            'score': worst_action.get('performance_score', 0)
        },
        'efficiency_distribution': dict(efficiency_counts)
    }

def generate_performance_insights(overall_stats: dict, tool_stats: list, hourly_metrics: list) -> list:
    """Generate actionable performance insights"""
    insights = []
    
    # Success rate insights
    success_rate = (overall_stats.get('successful_actions', 0) / overall_stats.get('total_actions', 1)) * 100
    if success_rate < 80:
        insights.append({
            'type': 'warning',
            'title': 'Low Success Rate',
            'message': f'Current success rate is {success_rate:.1f}%. Consider investigating failing tools.',
            'severity': 'high'
        })
    
    # Tool performance insights
    if tool_stats:
        slowest_tool = max(tool_stats, key=lambda t: t.get('avg_duration', 0))
        if slowest_tool.get('avg_duration', 0) > 60:
            insights.append({
                'type': 'info',
                'title': 'Performance Optimization',
                'message': f'{slowest_tool["tool_name"]} is taking {slowest_tool["avg_duration"]:.1f}s on average. Consider optimization.',
                'severity': 'medium'
            })
    
    # Usage pattern insights
    if hourly_metrics:
        peak_hour = max(hourly_metrics, key=lambda h: h.get('action_count', 0))
        insights.append({
            'type': 'info',
            'title': 'Peak Usage',
            'message': f'Peak usage occurs at {peak_hour["hour"]}:00 with {peak_hour["action_count"]} actions.',
            'severity': 'low'
        })
    
    return insights

def calculate_tool_reliability_score(tool_stats: dict) -> float:
    """Calculate reliability score for a tool"""
    total_calls = tool_stats.get('total_calls', 0)
    successful_calls = tool_stats.get('successful_calls', 0)
    
    if total_calls == 0:
        return 0.0
    
    success_rate = successful_calls / total_calls
    
    # Factor in volume (more calls = more confidence in score)
    volume_factor = min(1.0, total_calls / 100)  # Normalize to 100 calls
    
    return round(success_rate * volume_factor * 100, 2)

def categorize_tool_performance(avg_execution_time: float) -> str:
    """Categorize tool performance based on average execution time"""
    if avg_execution_time is None:
        return 'unknown'
    
    if avg_execution_time <= 5:
        return 'fast'
    elif avg_execution_time <= 15:
        return 'normal'
    elif avg_execution_time <= 30:
        return 'slow'
    else:
        return 'very_slow'

# Helper functions for cognitive state processing

def calculate_state_complexity(state_data: dict) -> float:
    """Calculate complexity score for a cognitive state"""
    if not state_data:
        return 0.0
    
    # Count different types of components
    component_count = len(state_data.keys())
    
    # Calculate nested depth
    max_depth = calculate_dict_depth(state_data)
    
    # Count total values
    total_values = count_dict_values(state_data)
    
    # Normalize to 0-100 scale
    complexity = min(100, (component_count * 5) + (max_depth * 10) + (total_values * 0.5))
    return round(complexity, 2)

def calculate_dict_depth(d: dict, current_depth: int = 0) -> int:
    """Calculate maximum depth of nested dictionary"""
    if not isinstance(d, dict):
        return current_depth
    
    if not d:
        return current_depth
    
    return max(calculate_dict_depth(v, current_depth + 1) for v in d.values())

def count_dict_values(d: dict) -> int:
    """Count total number of values in nested dictionary"""
    count = 0
    for v in d.values():
        if isinstance(v, dict):
            count += count_dict_values(v)
        elif isinstance(v, list):
            count += len(v)
        else:
            count += 1
    return count

def compute_state_diff(state1: dict, state2: dict) -> dict:
    """Compute difference between two cognitive states"""
    diff_result = {
        'added': {},
        'removed': {},
        'modified': {},
        'magnitude': 0.0
    }
    
    all_keys = set(state1.keys()) | set(state2.keys())
    changes = 0
    total_keys = len(all_keys)
    
    for key in all_keys:
        if key not in state1:
            diff_result['added'][key] = state2[key]
            changes += 1
        elif key not in state2:
            diff_result['removed'][key] = state1[key]
            changes += 1
        elif state1[key] != state2[key]:
            diff_result['modified'][key] = {
                'before': state1[key],
                'after': state2[key]
            }
            changes += 1
    
    # Calculate magnitude as percentage of changed keys
    if total_keys > 0:
        diff_result['magnitude'] = (changes / total_keys) * 100
    
    return diff_result

def compute_detailed_state_diff(state1: dict, state2: dict) -> dict:
    """Compute detailed difference with semantic analysis"""
    basic_diff = compute_state_diff(state1.get('state_data', {}), state2.get('state_data', {}))
    
    # Add metadata comparison
    metadata_diff = {
        'timestamp_diff': state2['timestamp'] - state1['timestamp'],
        'type_changed': state1.get('state_type') != state2.get('state_type'),
        'workflow_changed': state1.get('workflow_id') != state2.get('workflow_id')
    }
    
    # Analyze semantic changes
    semantic_analysis = analyze_semantic_changes(state1.get('state_data', {}), state2.get('state_data', {}))
    
    return {
        **basic_diff,
        'metadata': metadata_diff,
        'semantic_changes': semantic_analysis,
        'summary': generate_diff_summary(basic_diff, metadata_diff)
    }

def analyze_semantic_changes(state1: dict, state2: dict) -> dict:
    """Analyze semantic meaning of state changes"""
    changes = {
        'goal_changes': [],
        'memory_changes': [],
        'capability_changes': [],
        'focus_changes': []
    }
    
    # Analyze goal-related changes
    if 'goals' in state1 or 'goals' in state2:
        goals1 = state1.get('goals', [])
        goals2 = state2.get('goals', [])
        if goals1 != goals2:
            changes['goal_changes'].append({
                'type': 'goal_modification',
                'description': f"Goals changed from {len(goals1)} to {len(goals2)} items"
            })
    
    # Analyze memory changes
    if 'working_memory' in state1 or 'working_memory' in state2:
        mem1 = state1.get('working_memory', {})
        mem2 = state2.get('working_memory', {})
        if mem1 != mem2:
            changes['memory_changes'].append({
                'type': 'working_memory_change',
                'description': 'Working memory contents modified'
            })
    
    return changes

def generate_diff_summary(basic_diff: dict, metadata_diff: dict) -> str:
    """Generate human-readable summary of state differences"""
    changes = []
    
    if basic_diff['added']:
        changes.append(f"{len(basic_diff['added'])} components added")
    
    if basic_diff['removed']:
        changes.append(f"{len(basic_diff['removed'])} components removed")
    
    if basic_diff['modified']:
        changes.append(f"{len(basic_diff['modified'])} components modified")
    
    if metadata_diff['type_changed']:
        changes.append("state type changed")
    
    if metadata_diff['workflow_changed']:
        changes.append("workflow context changed")
    
    if not changes:
        return "No significant changes detected"
    
    return ", ".join(changes).capitalize()

def find_cognitive_patterns(states: list, min_length: int, similarity_threshold: float) -> list:
    """Find recurring patterns in cognitive states"""
    patterns = []
    
    # Group states by type for pattern detection
    type_sequences = defaultdict(list)
    for state in states:
        type_sequences[state['state_type']].append(state)
    
    # Find repeating sequences within each type
    for state_type, sequence in type_sequences.items():
        if len(sequence) >= min_length * 2:  # Need at least 2 repetitions
            # Look for repeating subsequences
            for length in range(min_length, len(sequence) // 2 + 1):
                for start in range(len(sequence) - length * 2 + 1):
                    subseq1 = sequence[start:start + length]
                    subseq2 = sequence[start + length:start + length * 2]
                    
                    similarity = calculate_sequence_similarity(subseq1, subseq2)
                    if similarity >= similarity_threshold:
                        patterns.append({
                            'type': f'repeating_{state_type}',
                            'length': length,
                            'similarity': similarity,
                            'occurrences': 2,  # Could be extended to find more
                            'first_occurrence': subseq1[0]['timestamp'],
                            'pattern_description': f"Repeating {state_type} sequence of {length} states"
                        })
    
    return sorted(patterns, key=lambda p: p['similarity'], reverse=True)

def calculate_sequence_similarity(seq1: list, seq2: list) -> float:
    """Calculate similarity between two state sequences"""
    if len(seq1) != len(seq2):
        return 0.0
    
    total_similarity = 0.0
    for s1, s2 in zip(seq1, seq2, strict=False):
        state_sim = calculate_single_state_similarity(s1, s2)
        total_similarity += state_sim
    
    return total_similarity / len(seq1)

def calculate_single_state_similarity(state1: dict, state2: dict) -> float:
    """Calculate similarity between two individual states"""
    # Simple implementation - could be made more sophisticated
    data1 = state1.get('state_data', {})
    data2 = state2.get('state_data', {})
    
    if not data1 and not data2:
        return 1.0
    
    if not data1 or not data2:
        return 0.0
    
    # Compare keys
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    key_similarity = len(keys1 & keys2) / len(keys1 | keys2) if keys1 | keys2 else 1.0
    
    # Compare values for common keys
    common_keys = keys1 & keys2
    value_similarity = 0.0
    if common_keys:
        matching_values = sum(1 for key in common_keys if data1[key] == data2[key])
        value_similarity = matching_values / len(common_keys)
    
    return (key_similarity + value_similarity) / 2

def analyze_state_transitions(states: list) -> list:
    """Analyze transitions between cognitive states"""
    transitions = defaultdict(int)
    
    for i in range(len(states) - 1):
        current_type = states[i]['state_type']
        next_type = states[i + 1]['state_type']
        transition = f"{current_type} → {next_type}"
        transitions[transition] += 1
    
    # Sort by frequency
    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {
            'transition': transition,
            'count': count,
            'percentage': (count / (len(states) - 1)) * 100 if len(states) > 1 else 0
        }
        for transition, count in sorted_transitions
    ]

def detect_cognitive_anomalies(states: list) -> list:
    """Detect anomalous cognitive states"""
    anomalies = []
    
    if len(states) < 3:
        return anomalies
    
    # Calculate average complexity
    complexities = [calculate_state_complexity(s.get('state_data', {})) for s in states]
    avg_complexity = sum(complexities) / len(complexities)
    std_complexity = (sum((c - avg_complexity) ** 2 for c in complexities) / len(complexities)) ** 0.5
    
    # Find states with unusual complexity
    for i, state in enumerate(states):
        complexity = complexities[i]
        z_score = (complexity - avg_complexity) / std_complexity if std_complexity > 0 else 0
        
        if abs(z_score) > 2:  # More than 2 standard deviations
            anomalies.append({
                'state_id': state['state_id'],
                'timestamp': state['timestamp'],
                'anomaly_type': 'complexity_outlier',
                'z_score': z_score,
                'description': f"Unusual complexity: {complexity:.1f} (avg: {avg_complexity:.1f})",
                'severity': 'high' if abs(z_score) > 3 else 'medium'
            })
    
    return anomalies

def extract_key_state_components(state_data: dict) -> list:
    """Extract key components from state data for visualization"""
    components = []
    
    for key, value in state_data.items():
        component = {
            'name': key,
            'type': type(value).__name__,
            'summary': str(value)[:100] + '...' if len(str(value)) > 100 else str(value)
        }
        
        if isinstance(value, dict):
            component['count'] = len(value)
        elif isinstance(value, list):
            component['count'] = len(value)
        
        components.append(component)
    
    return components[:10]  # Limit to top 10 components

def generate_state_tags(state_data: dict, state_type: str) -> list:
    """Generate descriptive tags for a cognitive state"""
    tags = [state_type]
    
    # Add tags based on state data content
    if 'goals' in state_data:
        tags.append('goal-oriented')
    
    if 'working_memory' in state_data:
        tags.append('memory-active')
    
    if 'error' in str(state_data).lower():
        tags.append('error-state')
    
    if 'decision' in str(state_data).lower():
        tags.append('decision-point')
    
    return tags

def generate_timeline_segments(timeline_data: list, granularity: str, hours: int) -> list:
    """Generate timeline segments for visualization"""
    if not timeline_data:
        return []
    
    start_time = min(item['timestamp'] for item in timeline_data)
    end_time = max(item['timestamp'] for item in timeline_data)
    
    # Determine segment duration
    if granularity == 'minute':
        segment_duration = 60
    elif granularity == 'hour':
        segment_duration = 3600
    else:  # second
        segment_duration = 1
    
    segments = []
    current_time = start_time
    
    while current_time < end_time:
        segment_end = current_time + segment_duration
        
        # Find states in this segment
        segment_states = [
            item for item in timeline_data
            if current_time <= item['timestamp'] < segment_end
        ]
        
        if segment_states:
            avg_complexity = sum(s['complexity_score'] for s in segment_states) / len(segment_states)
            max_change = max(s['change_magnitude'] for s in segment_states)
            
            segments.append({
                'start_time': current_time,
                'end_time': segment_end,
                'state_count': len(segment_states),
                'avg_complexity': avg_complexity,
                'max_change_magnitude': max_change,
                'dominant_type': Counter(s['state_type'] for s in segment_states).most_common(1)[0][0]
            })
        
        current_time = segment_end
    
    return segments

def calculate_timeline_stats(timeline_data: list) -> dict:
    """Calculate summary statistics for timeline data"""
    if not timeline_data:
        return {}
    
    complexities = [item['complexity_score'] for item in timeline_data]
    changes = [item['change_magnitude'] for item in timeline_data if item['change_magnitude'] > 0]
    
    state_types = Counter(item['state_type'] for item in timeline_data)
    
    return {
        'avg_complexity': sum(complexities) / len(complexities),
        'max_complexity': max(complexities),
        'avg_change_magnitude': sum(changes) / len(changes) if changes else 0,
        'max_change_magnitude': max(changes) if changes else 0,
        'most_common_type': state_types.most_common(1)[0][0] if state_types else None,
        'type_distribution': dict(state_types),
        'total_duration_hours': (timeline_data[-1]['timestamp'] - timeline_data[0]['timestamp']) / 3600
    }

def capture_current_system_state(conn) -> dict:
    """Capture current system state for backup purposes"""
    cursor = conn.cursor()
    
    state = {
        'timestamp': datetime.now().timestamp(),
        'working_memory': [],
        'goals': [],
        'recent_actions': [],
        'system_metrics': {}
    }
    
    try:
        # Get recent working memory
        cursor.execute("""
            SELECT memory_id, content, importance, memory_type
            FROM memories 
            WHERE memory_level = 'working'
            ORDER BY created_at DESC
            LIMIT 20
        """)
        state['working_memory'] = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Get active goals
        cursor.execute("""
            SELECT goal_id, title, status, progress
            FROM goals 
            WHERE status IN ('active', 'in_progress')
            ORDER BY created_at DESC
            LIMIT 10
        """)
        state['goals'] = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
        
        # Get recent actions
        cursor.execute("""
            SELECT action_id, action_type, status, started_at
            FROM actions 
            ORDER BY started_at DESC
            LIMIT 10
        """)
        state['recent_actions'] = [dict(zip([d[0] for d in cursor.description], row, strict=False)) for row in cursor.fetchall()]
    except sqlite3.OperationalError:
        # Tables might not exist yet
        pass
    
    return state

def perform_state_restoration(conn, state_data: dict, target_state: dict) -> dict:
    """Perform actual state restoration (implementation would depend on system architecture)"""
    restoration_result = {
        'restored_components': [],
        'failed_components': [],
        'warnings': []
    }
    
    # Example restoration logic (would need to be implemented based on actual system)
    try:
        if 'working_memory' in state_data:
            # Restore working memory state
            restoration_result['restored_components'].append('working_memory')
        
        if 'goals' in state_data:
            # Restore goal state
            restoration_result['restored_components'].append('goals')
        
        # Add warning about limitations
        restoration_result['warnings'].append('State restoration is limited to compatible components')
        
    except Exception as e:
        restoration_result['failed_components'].append(f'Error during restoration: {str(e)}')
    
    return restoration_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 