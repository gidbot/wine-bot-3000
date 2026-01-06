#!/usr/bin/env python3
"""
Wine Menu Analyzer Web App
Flask application that analyzes wine menu photos, extracts wines using Claude Vision,
looks up retail prices via Wine Labs API, and calculates markups.
"""

import os
import json
import hashlib
import re
import sqlite3
import uuid
import base64
import shutil
from contextlib import contextmanager
import requests
import urllib3
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from flask import Flask, request, jsonify, send_file, render_template_string

import anthropic

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Wine Labs API Configuration
WINE_LABS_BASE_URL = "https://external-api.wine-labs.com"
WINE_LABS_USER_ID = os.environ.get("WINE_LABS_USER_ID", "")

# Storage configuration
# Use persistent disk in production (/var/data on Render), local directory in development
STORAGE_DIR = os.environ.get("CACHE_DIR", os.path.dirname(__file__))
CACHE_DB = os.path.join(STORAGE_DIR, "wine_cache.db")
CACHE_TTL_HOURS = 24  # Cache expires after 24 hours (set to 0 to never expire)

# Shareable analyses configuration
ANALYSES_DB = os.path.join(STORAGE_DIR, "analyses.db")
IMAGES_DIR = os.path.join(STORAGE_DIR, "images")
ANALYSIS_EXPIRY_DAYS = 7  # Shared analyses expire after 7 days

# Glass pour estimation - standard 5oz pour = 5 glasses per 750ml bottle
GLASSES_PER_BOTTLE = 5

# API Timeout Configuration (in seconds)
CLAUDE_TIMEOUT = 60  # Claude Vision can be slow with large images
WINE_LABS_TIMEOUT = 30  # Wine Labs API timeout per request

# Anthropic API key from environment
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

# =============================================================================
# SQLite Cache Class
# =============================================================================

class APICache:
    """
    SQLite-based cache for API responses.
    
    Benefits over JSON file:
    - Handles concurrent access safely (ACID compliant)
    - Only reads/writes what's needed (not entire cache)
    - Automatic cleanup of expired entries
    - More efficient for large datasets
    """
    
    def __init__(self, db_path: str, ttl_hours: int = 24):
        self.db_path = db_path
        self.ttl_hours = ttl_hours
        self._init_db()
    
    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize database schema."""
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    endpoint TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    response TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_cached_at ON cache(cached_at)')
    
    def _make_key(self, endpoint: str, payload: dict) -> str:
        """Create a unique cache key from endpoint and payload."""
        payload_for_key = {k: v for k, v in payload.items() if k != "user_id"}
        key_str = f"{endpoint}:{json.dumps(payload_for_key, sort_keys=True)}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cleanup_expired(self, conn):
        """Remove expired entries from cache."""
        if self.ttl_hours > 0:
            expiry_time = datetime.now() - timedelta(hours=self.ttl_hours)
            conn.execute(
                "DELETE FROM cache WHERE cached_at < ?",
                (expiry_time.isoformat(),)
            )
    
    def get(self, endpoint: str, payload: dict) -> Optional[dict]:
        """Get cached response if available and not expired."""
        key = self._make_key(endpoint, payload)
        
        with self._get_conn() as conn:
            # Periodically clean up expired entries
            self._cleanup_expired(conn)
            
            row = conn.execute(
                "SELECT response FROM cache WHERE key = ?",
                (key,)
            ).fetchone()
            
            if row:
                return json.loads(row["response"])
            return None
    
    def set(self, endpoint: str, payload: dict, response: dict):
        """Cache a response."""
        key = self._make_key(endpoint, payload)
        
        with self._get_conn() as conn:
            conn.execute('''
                INSERT OR REPLACE INTO cache (key, endpoint, payload, response, cached_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                key,
                endpoint,
                json.dumps(payload),
                json.dumps(response),
                datetime.now().isoformat()
            ))
    
    def clear(self):
        """Clear all cached data."""
        with self._get_conn() as conn:
            conn.execute("DELETE FROM cache")
    
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
            
            if self.ttl_hours > 0:
                expiry_time = datetime.now() - timedelta(hours=self.ttl_hours)
                expired = conn.execute(
                    "SELECT COUNT(*) FROM cache WHERE cached_at < ?",
                    (expiry_time.isoformat(),)
                ).fetchone()[0]
            else:
                expired = 0
            
            return {
                "total_entries": total,
                "expired_entries": expired,
                "valid_entries": total - expired
            }


# Global cache instance
cache = APICache(CACHE_DB, CACHE_TTL_HOURS)


# =============================================================================
# Analyses Storage (for shareable links)
# =============================================================================

class AnalysesStore:
    """
    SQLite-based storage for shareable wine analyses.
    Stores analysis results and manages image files with automatic expiration.
    """
    
    def __init__(self, db_path: str, images_dir: str, expiry_days: int = 7):
        self.db_path = db_path
        self.images_dir = images_dir
        self.expiry_days = expiry_days
        self._init_storage()
    
    def _init_storage(self):
        """Initialize database and images directory."""
        # Create images directory if it doesn't exist
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize database
        with self._get_conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analyses (
                    id TEXT PRIMARY KEY,
                    image_filename TEXT NOT NULL,
                    results_json TEXT NOT NULL,
                    wine_count INTEGER DEFAULT 0,
                    matched_count INTEGER DEFAULT 0,
                    restaurant_name TEXT DEFAULT '',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP NOT NULL
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON analyses(expires_at)')
            
            # Add restaurant_name column if it doesn't exist (migration for existing DBs)
            try:
                conn.execute("ALTER TABLE analyses ADD COLUMN restaurant_name TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # Column already exists
    
    @contextmanager
    def _get_conn(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()
    
    def cleanup_expired(self):
        """Remove expired analyses and their image files."""
        with self._get_conn() as conn:
            # Find expired records
            expired = conn.execute(
                "SELECT id, image_filename FROM analyses WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            ).fetchall()
            
            # Delete image files
            for record in expired:
                image_path = os.path.join(self.images_dir, record["image_filename"])
                if os.path.exists(image_path):
                    try:
                        os.remove(image_path)
                        print(f"Deleted expired image: {record['image_filename']}")
                    except OSError as e:
                        print(f"Error deleting image {record['image_filename']}: {e}")
            
            # Delete database records
            if expired:
                conn.execute("DELETE FROM analyses WHERE expires_at < ?", 
                           (datetime.now().isoformat(),))
                print(f"Cleaned up {len(expired)} expired analyses")
            
            return len(expired)
    
    def save_analysis(self, image_base64: str, media_type: str, results: dict, restaurant_name: str = "") -> str:
        """
        Save an analysis with its image.
        
        Returns the unique ID for the shareable link.
        """
        # Run cleanup first
        self.cleanup_expired()
        
        # Generate unique ID
        analysis_id = str(uuid.uuid4())[:12]  # Short ID for nicer URLs
        
        # Determine file extension from media type
        ext_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/webp': '.webp',
            'image/heic': '.heic',
        }
        ext = ext_map.get(media_type, '.jpg')
        image_filename = f"{analysis_id}{ext}"
        
        # Save image file
        image_path = os.path.join(self.images_dir, image_filename)
        image_data = base64.b64decode(image_base64)
        with open(image_path, 'wb') as f:
            f.write(image_data)
        
        # Calculate expiration
        expires_at = datetime.now() + timedelta(days=self.expiry_days)
        
        # Save to database
        with self._get_conn() as conn:
            conn.execute('''
                INSERT INTO analyses (id, image_filename, results_json, wine_count, matched_count, restaurant_name, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                analysis_id,
                image_filename,
                json.dumps(results),
                results.get("total_wines", 0),
                results.get("matched_wines", 0),
                restaurant_name,
                expires_at.isoformat()
            ))
        
        return analysis_id
    
    def get_analysis(self, analysis_id: str) -> Optional[dict]:
        """
        Retrieve an analysis by ID.
        
        Returns None if not found or expired.
        """
        # Run cleanup first
        self.cleanup_expired()
        
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE id = ? AND expires_at > ?",
                (analysis_id, datetime.now().isoformat())
            ).fetchone()
            
            if not row:
                return None
            
            # Check if image file exists
            image_path = os.path.join(self.images_dir, row["image_filename"])
            if not os.path.exists(image_path):
                return None
            
            return {
                "id": row["id"],
                "image_filename": row["image_filename"],
                "results": json.loads(row["results_json"]),
                "wine_count": row["wine_count"],
                "matched_count": row["matched_count"],
                "restaurant_name": row["restaurant_name"] or "",
                "created_at": row["created_at"],
                "expires_at": row["expires_at"]
            }
    
    def get_image_path(self, analysis_id: str) -> Optional[str]:
        """Get the file path for an analysis's image."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT image_filename FROM analyses WHERE id = ? AND expires_at > ?",
                (analysis_id, datetime.now().isoformat())
            ).fetchone()
            
            if not row:
                return None
            
            image_path = os.path.join(self.images_dir, row["image_filename"])
            return image_path if os.path.exists(image_path) else None
    
    def stats(self) -> dict:
        """Get storage statistics."""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
            
            # Calculate total image size
            total_size = 0
            for filename in os.listdir(self.images_dir):
                filepath = os.path.join(self.images_dir, filename)
                if os.path.isfile(filepath):
                    total_size += os.path.getsize(filepath)
            
            return {
                "total_analyses": total,
                "total_image_size_mb": round(total_size / (1024 * 1024), 2),
                "expiry_days": self.expiry_days
            }


# Global analyses store instance
analyses_store = AnalysesStore(ANALYSES_DB, IMAGES_DIR, ANALYSIS_EXPIRY_DAYS)


# =============================================================================
# Claude Vision Integration
# =============================================================================

def analyze_menu_with_claude(image_base64: str, media_type: str = "image/jpeg", user_context: str = "") -> List[Dict]:
    """
    Use Claude Vision to analyze a wine menu image and extract wine information.
    
    Args:
        image_base64: Base64-encoded image data
        media_type: MIME type of the image
        user_context: Optional user-provided context about the menu
    
    Returns a list of dicts with keys: name, vintage, price
    Raises: TimeoutError, ConnectionError, or ValueError on failure
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        timeout=CLAUDE_TIMEOUT
    )
    
    # Build the prompt with optional user context
    context_section = ""
    if user_context:
        context_section = f"""
ADDITIONAL CONTEXT FROM USER:
{user_context}

Use this context to help identify wines more accurately. For example, if the user says it's a Sicilian wine list, prioritize Sicilian producers and appellations.

"""
    
    prompt = context_section + """Analyze this wine menu image and extract all wines listed with their prices.

CRITICAL RULES:
1. PRODUCER NAME ALWAYS COMES FIRST - this is essential for wine database matching
2. NEVER include grape variety names (Chardonnay, Pinot Noir, Cabernet Sauvignon, Merlot, Fiano, Grillo, etc.) in the wine name - these are NOT part of how wines are named in databases
3. Wine menus often organize wines under category headers - use these to infer region/appellation ONLY

FORMAT BY REGION:

CHAMPAGNE: Use format "Producer CuvéeName" - nothing else
- "Bollinger Special Cuvée" (NOT "Bollinger Champagne Chardonnay...")
- "Krug Grande Cuvée"
- "Dom Pérignon" 
- "Ayala Brut Majeur"
- "Bollinger La Grande Année"
- "Bérêche Brut Réserve"
- "Marc Hébrart Special Club"

BURGUNDY: Use format "Producer Appellation [Color if needed]"
- "Domaine Fourrier Bourgogne Blanc"
- "Dureuil-Janthial Bourgogne Blanc" 
- "Ballot-Millot Meursault"
- "Domaine Raveneau Chablis"
- For Burgundy, add "Blanc" or "Rouge" only when the appellation is generic (Bourgogne)

BORDEAUX: Use format "Château Name Appellation"
- "Château Margaux"
- "Château Latour Pauillac"

ITALIAN WINES: Use format "Producer Appellation 'CuvéeName'" - PRODUCER FIRST!
- "Biondi Etna Bianco 'Outis'" (NOT "Etna Bianco DOC 'Outis' Biondi")
- "Maugeri Etna Bianco Superiore 'Contrada Volpare'"
- "Ayunta Etna Bianco 'Piante Sparse'"
- "Ciro Picariello Fiano di Avellino"
- "Centopassi Terre Siciliane 'Rocce di Pietra Longa'"
- "Gaja Barbaresco"
- "Produttori del Barbaresco Barbaresco"
- "Vietti Barolo"
- For Italian wines: Producer FIRST, then DOC/DOCG appellation, then vineyard/cuvée name in quotes

OTHER REGIONS: Producer + Appellation/Region
- Keep it simple and searchable
- Producer name ALWAYS comes first

GLASS vs BOTTLE:
- Look for indicators like "glass", "gl", "/gl", "by the glass", or a "Glass" column/section
- If a wine is priced by the glass, set is_glass to true
- If by the bottle (default), set is_glass to false
- Glass pours are typically $10-35, bottles are typically $40+

Return the data as a JSON array with objects containing these fields:
- "name": the wine name formatted as above (producer + cuvée/appellation, NO grape varieties)
- "vintage": the year as a string (or null if not shown)
- "price": the numeric price (just the number, no currency symbol)
- "is_glass": true if this is a by-the-glass price, false if by the bottle

Only include wines where you can clearly read the name. If a price is not visible or unclear, use null.

Return ONLY the JSON array, no other text. Example format:
[
  {"name": "Domaine Fourrier Bourgogne Blanc", "vintage": "2023", "price": 150, "is_glass": false},
  {"name": "Bollinger La Grande Année", "vintage": "2014", "price": 28, "is_glass": true},
  {"name": "Krug Grande Cuvée", "vintage": null, "price": 450, "is_glass": false}
]"""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ],
                }
            ],
        )
    except anthropic.APITimeoutError:
        raise TimeoutError("Image analysis timed out. Please try again with a smaller or clearer image.")
    except anthropic.APIConnectionError:
        raise ConnectionError("Could not connect to the AI service. Please check your internet connection.")
    except anthropic.APIStatusError as e:
        raise ValueError(f"AI service error: {e.message}")
    
    # Extract JSON from response
    response_text = message.content[0].text.strip()
    
    print(f"Claude raw response: {response_text[:500]}...")
    
    # Try to parse JSON (handle potential markdown code blocks)
    if response_text.startswith("```"):
        # Remove markdown code block formatting
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
    
    # Handle empty array response
    if response_text.strip() == "[]":
        return []
    
    try:
        wines = json.loads(response_text)
        
        # Validate that we got a list
        if not isinstance(wines, list):
            print(f"Claude returned non-list type: {type(wines)}")
            return []
        
        # Filter out any invalid entries
        valid_wines = []
        for wine in wines:
            if isinstance(wine, dict) and wine.get("name"):
                valid_wines.append(wine)
        
        return valid_wines
        
    except json.JSONDecodeError as e:
        print(f"Failed to parse Claude response: {e}")
        print(f"Response was: {response_text}")
        # Return empty list instead of raising - let frontend handle gracefully
        return []


# =============================================================================
# Wine Labs API Functions
# =============================================================================

def cached_post(endpoint: str, payload: dict, cache_empty_results: bool = True) -> Optional[dict]:
    """
    Make a cached POST request to Wine Labs API.
    
    Args:
        endpoint: API endpoint path
        payload: Request payload
        cache_empty_results: If False, don't cache responses with empty "results" arrays.
                            This allows retrying later when data might be available.
    
    Returns None on failure (timeout, connection error, or API error).
    Failures are logged but don't raise exceptions - allows partial results.
    """
    cached_response = cache.get(endpoint, payload)
    if cached_response is not None:
        # If we cached an empty result and cache_empty_results is False,
        # ignore the cache and try again (data might be available now)
        if not cache_empty_results:
            results = cached_response.get("results", [])
            if not results:
                print(f"[CACHE SKIP - empty results] {endpoint}")
                # Fall through to make a fresh API call
            else:
                print(f"[CACHED] {endpoint}")
                return cached_response
        else:
            print(f"[CACHED] {endpoint}")
            return cached_response
    
    print(f"[API CALL] {endpoint}")
    url = f"{WINE_LABS_BASE_URL}{endpoint}"
    
    try:
        response = requests.post(
            url, 
            json=payload, 
            verify=False,
            timeout=WINE_LABS_TIMEOUT
        )
        if response.status_code != 200:
            print(f"  Wine Labs API error: {response.status_code}")
            return None
        
        data = response.json()
        
        # Only cache if we have results OR if caching empty results is allowed
        results = data.get("results", [])
        if results or cache_empty_results:
            cache.set(endpoint, payload, data)
        else:
            print(f"  [NOT CACHING - no results found]")
        
        return data
    except requests.exceptions.Timeout:
        print(f"  Wine Labs API timeout for {endpoint}")
        return None
    except requests.exceptions.ConnectionError:
        print(f"  Wine Labs API connection error for {endpoint}")
        return None
    except Exception as e:
        print(f"  Wine Labs API error: {e}")
        return None


def get_retail_price(query: str) -> dict:
    """
    Get retail price by searching /price_stats with wine name query.
    Uses winelabs_price from the response.
    
    Returns dict with: price, count, source
    """
    result = {
        "price": None,
        "count": 0,
        "source": None
    }
    
    if not query:
        return result
    
    payload = {
        "user_id": WINE_LABS_USER_ID,
        "query": query,
        "currency": "USD"
    }
    
    print(f"  [PRICE] Searching for: {query}")
    
    # Don't cache empty results - price data may become available later
    data = cached_post("/price_stats", payload, cache_empty_results=False)
    
    if data:
        results_list = data.get("results", [])
        print(f"  [PRICE] Results count: {len(results_list)}")
        
        if results_list:
            stats = results_list[0]
            winelabs_price = stats.get("winelabs_price")
            print(f"  [PRICE] winelabs_price: {winelabs_price}")
            
            if winelabs_price is not None:
                result["price"] = float(winelabs_price)
                result["count"] = stats.get("count", 0)
                result["source"] = "price_stats"
                return result
    else:
        print(f"  [PRICE] API returned None")
    
    return result


def process_wines(wines: List[Dict]) -> List[Dict]:
    """
    Process extracted wines: lookup retail prices and calculate markups.
    Handles by-the-glass wines with bottle price estimation.
    """
    results = []
    
    for wine in wines:
        # Build query from wine name
        name = wine.get("name", "")
        vintage = wine.get("vintage")
        if vintage:
            query = f"{name} {vintage}"
        else:
            query = name
        
        # Use Claude's extracted name as display name
        display_name = name
        
        # Get menu price (original price from menu)
        original_menu_price = wine.get("price")
        if original_menu_price is not None:
            try:
                original_menu_price = float(original_menu_price)
            except (ValueError, TypeError):
                original_menu_price = None
        
        # Get retail price by searching with wine name
        price_data = get_retail_price(query)
        retail_price = price_data["price"]
        listings_count = price_data["count"]
        price_source = price_data["source"]
        
        # Detect if wine is by the glass
        is_glass = wine.get("is_glass", False)
        
        # Auto-detection heuristic: if Claude didn't mark it as glass,
        # check if price seems too low compared to retail
        auto_detected_glass = False
        if not is_glass and original_menu_price is not None:
            # Heuristic 1: Menu price under $40 AND we have retail data showing it's worth more
            if original_menu_price < 40 and retail_price is not None:
                # If menu price is less than 60% of retail, likely a glass price
                if original_menu_price < retail_price * 0.6:
                    is_glass = True
                    auto_detected_glass = True
                    print(f"  Auto-detected glass pour: {display_name} @ ${original_menu_price} (retail: ${retail_price:.2f})")
            # Heuristic 2: Price is suspiciously low for any wine (under $20)
            elif original_menu_price < 20:
                is_glass = True
                auto_detected_glass = True
                print(f"  Auto-detected glass pour (low price): {display_name} @ ${original_menu_price}")
        
        # Calculate estimated bottle price and menu price for markup calculation
        if is_glass and original_menu_price is not None:
            estimated_bottle_price = original_menu_price * GLASSES_PER_BOTTLE
            menu_price = estimated_bottle_price  # Use bottle price for markup calculation
            glass_price = original_menu_price
        else:
            estimated_bottle_price = None
            menu_price = original_menu_price
            glass_price = None
        
        # Calculate markup (using estimated bottle price for glass pours)
        if menu_price is not None and retail_price is not None:
            markup_dollars = menu_price - retail_price
            markup_percent = (markup_dollars / retail_price) * 100
        else:
            markup_dollars = None
            markup_percent = None
        
        results.append({
            "original_name": wine.get("name", "Unknown"),
            "display_name": display_name,
            "vintage": wine.get("vintage"),
            "menu_price": menu_price,  # Estimated bottle price if glass pour
            "glass_price": glass_price,  # Original glass price (None if bottle)
            "is_glass": is_glass,
            "auto_detected_glass": auto_detected_glass,
            "estimated_bottle_price": estimated_bottle_price,
            "retail_price": retail_price,
            "listings_count": listings_count,
            "price_source": price_source,
            "markup_dollars": markup_dollars,
            "markup_percent": markup_percent,
            "matched": retail_price is not None
        })
    
    return results


# =============================================================================
# Flask Routes
# =============================================================================

@app.route("/")
def index():
    """Serve the main HTML interface."""
    html_path = os.path.join(os.path.dirname(__file__), "wine_report.html")
    return send_file(html_path)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Analyze a wine menu image.
    
    Expects JSON with:
    - image: base64-encoded image data
    - media_type: MIME type (e.g., "image/jpeg", "image/png")
    - context: (optional) user-provided context about the menu
    
    Returns JSON with analyzed wine data and markups.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        image_base64 = data.get("image")
        media_type = data.get("media_type", "image/jpeg")
        user_context = data.get("context", "")
        
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400
        
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        # Step 1: Analyze menu with Claude Vision
        if user_context:
            print(f"Analyzing menu with Claude Vision (with context: '{user_context[:50]}...')...")
        else:
            print("Analyzing menu with Claude Vision...")
        wines = analyze_menu_with_claude(image_base64, media_type, user_context)
        
        if not wines:
            return jsonify({
                "error": "no_wines_found",
                "error_message": "No wines could be identified in this image",
                "suggestions": [
                    "Make sure the image clearly shows a wine menu or wine list",
                    "Try taking a clearer photo with better lighting",
                    "Ensure wine names and prices are visible and readable",
                    "Crop the image to focus on the wine section"
                ],
                "wines": []
            }), 200
        
        print(f"\n{'='*60}")
        print(f"CLAUDE EXTRACTED {len(wines)} WINES:")
        print(f"{'='*60}")
        for i, wine in enumerate(wines, 1):
            name = wine.get("name", "Unknown")
            vintage = wine.get("vintage", "NV")
            price = wine.get("price", "N/A")
            query = f"{name} {vintage}" if vintage else name
            print(f"  {i}. {name}")
            print(f"     Vintage: {vintage} | Menu Price: ${price}")
            print(f"     Query to Wine Labs: \"{query}\"")
        print(f"{'='*60}\n")
        
        # Step 2: Process wines (lookup prices, calculate markups)
        print("Looking up retail prices...")
        results = process_wines(wines)
        
        # Step 3: Return results
        return jsonify({
            "success": True,
            "wines": results,
            "total_wines": len(results),
            "matched_wines": sum(1 for r in results if r["matched"]),
            "analyzed_at": datetime.now().isoformat()
        })
        
    except TimeoutError as e:
        print(f"Timeout error: {e}")
        return jsonify({
            "error": "timeout",
            "error_message": str(e),
            "suggestions": [
                "Try uploading a smaller image",
                "Crop the image to focus on the wine list",
                "Ensure good lighting and image clarity",
                "Try again in a few moments"
            ]
        }), 504
    except ConnectionError as e:
        print(f"Connection error: {e}")
        return jsonify({
            "error": "connection_error",
            "error_message": str(e),
            "suggestions": [
                "Check your internet connection",
                "Try again in a few moments"
            ]
        }), 503
    except ValueError as e:
        return jsonify({
            "error": "validation_error",
            "error_message": str(e)
        }), 400
    except Exception as e:
        print(f"Error analyzing menu: {e}")
        return jsonify({
            "error": "server_error",
            "error_message": f"An unexpected error occurred: {str(e)}",
            "suggestions": [
                "Try uploading a different image",
                "Refresh the page and try again"
            ]
        }), 500


@app.route("/cache/stats", methods=["GET"])
def cache_stats():
    """Get cache statistics."""
    return jsonify(cache.stats())


@app.route("/cache/clear", methods=["POST"])
def clear_cache():
    """Clear the cache."""
    cache.clear()
    return jsonify({"success": True, "message": "Cache cleared"})


# =============================================================================
# Shareable Analyses Routes
# =============================================================================

@app.route("/save", methods=["POST"])
def save_analysis():
    """
    Save an analysis for sharing.
    
    Expects JSON with:
    - image: base64-encoded image data
    - media_type: MIME type
    - results: the analysis results object
    - restaurant_name: name of the restaurant
    
    Returns JSON with shareable ID.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        image_base64 = data.get("image")
        media_type = data.get("media_type", "image/jpeg")
        results = data.get("results")
        restaurant_name = data.get("restaurant_name", "")
        
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400
        if not results:
            return jsonify({"error": "No results provided"}), 400
        
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        # Save the analysis
        analysis_id = analyses_store.save_analysis(image_base64, media_type, results, restaurant_name)
        
        return jsonify({
            "success": True,
            "id": analysis_id,
            "expires_in_days": ANALYSIS_EXPIRY_DAYS
        })
        
    except Exception as e:
        print(f"Error saving analysis: {e}")
        return jsonify({"error": f"Failed to save: {str(e)}"}), 500


@app.route("/share/<analysis_id>")
def view_shared_analysis(analysis_id):
    """
    View a shared analysis.
    Returns the share page HTML.
    """
    analysis = analyses_store.get_analysis(analysis_id)
    
    if not analysis:
        # Return expired/not found page
        return render_template_string(SHARE_EXPIRED_HTML), 404
    
    # Return the share page with embedded data
    return render_template_string(
        SHARE_PAGE_HTML,
        analysis_id=analysis_id,
        analysis_data=json.dumps(analysis["results"]),
        wine_count=analysis["wine_count"],
        matched_count=analysis["matched_count"],
        restaurant_name=analysis["restaurant_name"],
        created_at=analysis["created_at"],
        expires_at=analysis["expires_at"]
    )


@app.route("/share/<analysis_id>/image")
def get_shared_image(analysis_id):
    """Serve the image for a shared analysis."""
    image_path = analyses_store.get_image_path(analysis_id)
    
    if not image_path:
        return jsonify({"error": "Image not found"}), 404
    
    return send_file(image_path)


@app.route("/share/<analysis_id>/data")
def get_shared_data(analysis_id):
    """Get the JSON data for a shared analysis."""
    analysis = analyses_store.get_analysis(analysis_id)
    
    if not analysis:
        return jsonify({"error": "Analysis not found or expired"}), 404
    
    return jsonify(analysis["results"])


@app.route("/storage/stats", methods=["GET"])
def storage_stats():
    """Get storage statistics for shared analyses."""
    return jsonify(analyses_store.stats())


# =============================================================================
# Share Page Templates
# =============================================================================

SHARE_PAGE_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wine Bot 3000 - Shared Analysis</title>
    <meta property="og:title" content="Wine Menu Analysis - Wine Bot 3000">
    <meta property="og:description" content="Check out this wine menu markup analysis!">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500;600&family=Source+Sans+3:wght@300;400;500;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f0f14;
            --bg-secondary: #1a1a24;
            --bg-card: rgba(255, 255, 255, 0.03);
            --border-subtle: rgba(255, 255, 255, 0.08);
            --text-primary: #f5f5f7;
            --text-secondary: #8e8e93;
            --text-muted: #636366;
            --accent-gold: #c9a227;
            --accent-green: #34c759;
            --accent-red: #ff453a;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Source Sans 3', sans-serif;
            background: var(--bg-primary);
            min-height: 100vh;
            color: var(--text-primary);
            background-image: 
                radial-gradient(ellipse at 20% 0%, rgba(114, 47, 55, 0.15) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 100%, rgba(201, 162, 39, 0.08) 0%, transparent 50%);
        }
        .container { max-width: 1100px; margin: 0 auto; padding: 40px 24px; }
        header { text-align: center; margin-bottom: 32px; }
        h1 { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 500; margin-bottom: 8px; }
        h1 span { color: var(--accent-gold); }
        .subtitle { color: var(--text-secondary); font-size: 0.95rem; }
        .restaurant-name { 
            font-family: 'Playfair Display', serif;
            font-size: 1.6rem; 
            font-weight: 500;
            color: var(--accent-gold);
            margin: 16px 0 8px;
        }
        .shared-badge { 
            display: inline-block; 
            background: rgba(201, 162, 39, 0.2); 
            color: var(--accent-gold);
            padding: 4px 12px; 
            border-radius: 20px; 
            font-size: 0.8rem;
            margin-top: 12px;
        }
        .meta-info {
            text-align: center;
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-bottom: 24px;
        }
        .results-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 16px;
            overflow: hidden;
        }
        .results-table { width: 100%; border-collapse: collapse; }
        .results-table th {
            text-align: left;
            padding: 14px 12px;
            background: rgba(0, 0, 0, 0.2);
            color: var(--text-muted);
            font-weight: 500;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .results-table td {
            padding: 16px 12px;
            border-bottom: 1px solid var(--border-subtle);
        }
        .results-table tbody tr:last-child td { border-bottom: none; }
        .wine-name { font-weight: 500; }
        .price { font-family: monospace; }
        .price-menu { color: #e4c767; }
        .price-retail { color: var(--accent-green); }
        .markup-high { color: var(--accent-red); font-weight: 600; }
        .markup-medium { color: #ff9f0a; font-weight: 600; }
        .markup-low { color: var(--accent-green); font-weight: 600; }
        .na { color: var(--text-muted); font-style: italic; }
        .cta-section { text-align: center; margin-top: 40px; padding: 32px; background: var(--bg-card); border-radius: 16px; }
        .cta-section h2 { font-family: 'Playfair Display', serif; font-size: 1.4rem; margin-bottom: 12px; }
        .cta-section p { color: var(--text-secondary); margin-bottom: 20px; }
        .cta-btn {
            display: inline-block;
            background: linear-gradient(135deg, var(--accent-gold), #a88620);
            color: #0f0f14;
            padding: 14px 32px;
            border-radius: 12px;
            text-decoration: none;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .cta-btn:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(201, 162, 39, 0.3); }
        footer { text-align: center; margin-top: 40px; color: var(--text-muted); font-size: 0.85rem; }
        
        /* Menu Image Section */
        .menu-image-section {
            margin-bottom: 24px;
        }
        .image-toggle {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 12px 20px;
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            color: var(--text-secondary);
            cursor: pointer;
            transition: all 0.2s;
            font-size: 0.9rem;
        }
        .image-toggle:hover {
            background: rgba(255, 255, 255, 0.05);
            color: var(--text-primary);
        }
        .image-toggle svg {
            width: 18px;
            height: 18px;
            transition: transform 0.3s;
        }
        .image-toggle.expanded svg {
            transform: rotate(180deg);
        }
        .image-container {
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.4s ease-out, padding 0.4s ease-out;
            background: var(--bg-card);
            border-radius: 0 0 12px 12px;
            margin-top: -12px;
            border: 1px solid var(--border-subtle);
            border-top: none;
        }
        .image-container.expanded {
            max-height: 600px;
            padding: 16px;
        }
        .menu-image {
            width: 100%;
            max-height: 500px;
            object-fit: contain;
            border-radius: 8px;
            cursor: pointer;
            transition: opacity 0.2s;
        }
        .menu-image:hover {
            opacity: 0.9;
        }
        
        /* Lightbox */
        .lightbox {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 1000;
            cursor: zoom-out;
            padding: 20px;
        }
        .lightbox.active {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .lightbox img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 8px;
        }
        .lightbox-close {
            position: absolute;
            top: 20px;
            right: 20px;
            color: white;
            font-size: 2rem;
            cursor: pointer;
            opacity: 0.7;
            transition: opacity 0.2s;
        }
        .lightbox-close:hover {
            opacity: 1;
        }
        
        @media (max-width: 768px) {
            .results-table { font-size: 0.85rem; }
            .results-table th, .results-table td { padding: 10px 8px; }
            .hide-mobile { display: none; }
            .image-container.expanded { max-height: 400px; }
            .menu-image { max-height: 350px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Wine Bot <span>3000</span></h1>
            <p class="subtitle">Wine Menu Markup Analysis</p>
            {% if restaurant_name %}
            <h2 class="restaurant-name">{{ restaurant_name }}</h2>
            {% endif %}
            <span class="shared-badge">📤 Shared Analysis</span>
        </header>
        
        <div class="meta-info">
            {{ wine_count }} wines analyzed · {{ matched_count }} matched with retail prices
        </div>
        
        <!-- Menu Image Section - starts expanded -->
        <div class="menu-image-section">
            <div class="image-toggle expanded" onclick="toggleImage()">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                    <circle cx="8.5" cy="8.5" r="1.5"/>
                    <polyline points="21 15 16 10 5 21"/>
                </svg>
                <span id="toggle-text">Hide Original Menu</span>
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="width: 14px; height: 14px;">
                    <polyline points="6 9 12 15 18 9"/>
                </svg>
            </div>
            <div class="image-container expanded" id="image-container">
                <img src="/share/{{ analysis_id }}/image" alt="Original wine menu" class="menu-image" onclick="openLightbox()">
            </div>
        </div>
        
        <div class="results-card">
            <table class="results-table">
                <thead>
                    <tr>
                        <th>Wine</th>
                        <th class="hide-mobile">Vintage</th>
                        <th>Menu</th>
                        <th>Retail</th>
                        <th>Markup</th>
                    </tr>
                </thead>
                <tbody id="results-body"></tbody>
            </table>
        </div>
        
        <div class="cta-section">
            <h2>Try it yourself!</h2>
            <p>Snap a photo of any wine menu to reveal the markups</p>
            <a href="/" class="cta-btn">🍷 Analyze a Menu</a>
        </div>
        
        <footer>Wine Bot 3000</footer>
    </div>
    
    <!-- Lightbox for full-size image -->
    <div class="lightbox" id="lightbox" onclick="closeLightbox()">
        <span class="lightbox-close">&times;</span>
        <img src="/share/{{ analysis_id }}/image" alt="Original wine menu">
    </div>
    
    <script>
        // Image toggle and lightbox functions
        function toggleImage() {
            const container = document.getElementById('image-container');
            const toggle = document.querySelector('.image-toggle');
            const toggleText = document.getElementById('toggle-text');
            
            container.classList.toggle('expanded');
            toggle.classList.toggle('expanded');
            toggleText.textContent = container.classList.contains('expanded') 
                ? 'Hide Original Menu' 
                : 'View Original Menu';
        }
        
        function openLightbox() {
            document.getElementById('lightbox').classList.add('active');
            document.body.style.overflow = 'hidden';
        }
        
        function closeLightbox() {
            document.getElementById('lightbox').classList.remove('active');
            document.body.style.overflow = '';
        }
        
        // Close lightbox on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') closeLightbox();
        });
        const analysisData = {{ analysis_data | safe }};
        const wines = analysisData.wines || [];
        
        // Sort wines by markup percent ascending, N/A values at the bottom
        const sortedWines = [...wines].sort((a, b) => {
            const aMarkup = a.markup_percent;
            const bMarkup = b.markup_percent;
            
            const aIsNull = aMarkup === null || aMarkup === undefined;
            const bIsNull = bMarkup === null || bMarkup === undefined;
            
            // N/A values go to the end
            if (aIsNull && bIsNull) {
                // Both N/A - sort by menu price ascending
                const aPrice = a.menu_price || 0;
                const bPrice = b.menu_price || 0;
                return aPrice - bPrice;
            }
            if (aIsNull) return 1;  // a goes to end
            if (bIsNull) return -1; // b goes to end
            
            // Both have values - sort ascending
            return aMarkup - bMarkup;
        });
        
        function formatPrice(price) {
            if (price === null || price === undefined) return '<span class="na">N/A</span>';
            return '$' + price.toFixed(2);
        }
        
        function formatMarkup(percent) {
            if (percent === null || percent === undefined) return '<span class="na">N/A</span>';
            let colorClass = 'markup-high';
            if (percent < 50) colorClass = 'markup-low';
            else if (percent < 100) colorClass = 'markup-medium';
            return `<span class="${colorClass}">${percent.toFixed(0)}%</span>`;
        }
        
        const tbody = document.getElementById('results-body');
        sortedWines.forEach(wine => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="wine-name">${wine.display_name || wine.original_name}</td>
                <td class="hide-mobile">${wine.vintage || '—'}</td>
                <td class="price price-menu">${formatPrice(wine.menu_price)}</td>
                <td class="price price-retail">${formatPrice(wine.retail_price)}</td>
                <td>${formatMarkup(wine.markup_percent)}</td>
            `;
            tbody.appendChild(row);
        });
    </script>
</body>
</html>
'''

SHARE_EXPIRED_HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Wine Bot 3000 - Link Expired</title>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;500&family=Source+Sans+3:wght@300;400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0f0f14;
            --text-primary: #f5f5f7;
            --text-secondary: #8e8e93;
            --accent-gold: #c9a227;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Source Sans 3', sans-serif;
            background: var(--bg-primary);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-primary);
        }
        .container { text-align: center; padding: 40px; max-width: 500px; }
        .icon { font-size: 4rem; margin-bottom: 24px; opacity: 0.6; }
        h1 { font-family: 'Playfair Display', serif; font-size: 2rem; margin-bottom: 16px; }
        p { color: var(--text-secondary); margin-bottom: 32px; line-height: 1.6; }
        .cta-btn {
            display: inline-block;
            background: linear-gradient(135deg, var(--accent-gold), #a88620);
            color: #0f0f14;
            padding: 14px 32px;
            border-radius: 12px;
            text-decoration: none;
            font-weight: 600;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">🕐</div>
        <h1>Link Expired</h1>
        <p>This shared analysis is no longer available. Shared links expire after 7 days to keep things tidy.</p>
        <a href="/" class="cta-btn">🍷 Analyze a New Menu</a>
    </div>
</body>
</html>
'''


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Wine Menu Analyzer")
    print("=" * 60)
    print(f"Cache database: {CACHE_DB}")
    print(f"Cache TTL: {CACHE_TTL_HOURS} hours")
    print(f"Anthropic API Key: {'Set' if ANTHROPIC_API_KEY else 'NOT SET'}")
    print(f"Wine Labs User ID: {'Set' if WINE_LABS_USER_ID else 'NOT SET'}")
    print()
    
    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY environment variable not set!")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print()
    
    if not WINE_LABS_USER_ID:
        print("WARNING: WINE_LABS_USER_ID environment variable not set!")
        print("Set it with: export WINE_LABS_USER_ID='your-user-id-here'")
        print()
    
    app.run(debug=True, host="0.0.0.0", port=5000)

