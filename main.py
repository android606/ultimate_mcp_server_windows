import json
import logging
import os
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

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

def ensure_cognitive_states_table():
    """Ensure cognitive_timeline_states table exists for timeline feature"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create cognitive_timeline_states table if it doesn't exist (separate from existing cognitive_states)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cognitive_timeline_states (
            state_id TEXT PRIMARY KEY,
            timestamp REAL NOT NULL,
            state_type TEXT NOT NULL,
            state_data TEXT NOT NULL,
            workflow_id TEXT,
            description TEXT,
            created_at REAL DEFAULT (unixepoch()),
            FOREIGN KEY (workflow_id) REFERENCES workflows (workflow_id)
        )
    """)
    
    # Create index for performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_timeline_states_timestamp ON cognitive_timeline_states (timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cognitive_timeline_states_type ON cognitive_timeline_states (state_type)")
    
    conn.commit()
    conn.close()

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    ensure_cognitive_states_table()
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