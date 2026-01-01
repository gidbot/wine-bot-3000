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
from contextlib import contextmanager
import requests
import urllib3
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from flask import Flask, request, jsonify, send_file

import anthropic

# Disable SSL warnings for development
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Wine Labs API Configuration
WINE_LABS_BASE_URL = "https://external-api.wine-labs.com"
WINE_LABS_USER_ID = "58a3031a-7518-4b30-a53f-3828d0a7edac"

# Cache configuration
# Use persistent disk in production (/var/data on Render), local file in development
CACHE_DIR = os.environ.get("CACHE_DIR", os.path.dirname(__file__))
CACHE_DB = os.path.join(CACHE_DIR, "wine_cache.db")
CACHE_TTL_HOURS = 24  # Cache expires after 24 hours (set to 0 to never expire)

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
# Claude Vision Integration
# =============================================================================

def analyze_menu_with_claude(image_base64: str, media_type: str = "image/jpeg") -> List[Dict]:
    """
    Use Claude Vision to analyze a wine menu image and extract wine information.
    
    Returns a list of dicts with keys: name, vintage, price
    Raises: TimeoutError, ConnectionError, or ValueError on failure
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    
    client = anthropic.Anthropic(
        api_key=ANTHROPIC_API_KEY,
        timeout=CLAUDE_TIMEOUT
    )
    
    prompt = """Analyze this wine menu image and extract all wines listed with their prices.

CRITICAL RULES:
1. NEVER include grape variety names (Chardonnay, Pinot Noir, Cabernet Sauvignon, Merlot, etc.) in the wine name - these are NOT part of how wines are named in databases
2. Wine menus often organize wines under category headers - use these to infer region/appellation ONLY

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

OTHER REGIONS: Producer + Appellation/Region
- Keep it simple and searchable

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

def cached_post(endpoint: str, payload: dict) -> Optional[dict]:
    """
    Make a cached POST request to Wine Labs API.
    
    Returns None on failure (timeout, connection error, or API error).
    Failures are logged but don't raise exceptions - allows partial results.
    """
    cached_response = cache.get(endpoint, payload)
    if cached_response is not None:
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
        cache.set(endpoint, payload, data)
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


def match_wines_to_lwin(queries: List[str]) -> Dict[str, Dict]:
    """
    Match wine queries to LWIN codes using batch endpoint.
    Returns dict mapping query -> {display_name, lwin}
    """
    if not queries:
        return {}
    
    payload = {
        "user_id": WINE_LABS_USER_ID,
        "queries": queries
    }
    
    data = cached_post("/match_to_lwin_batch", payload)
    
    if data is None:
        return {q: {"display_name": q, "lwin": None} for q in queries}
    
    api_results = data.get("results", [])
    
    results = {}
    for i, query in enumerate(queries):
        if i < len(api_results):
            match_data = api_results[i]
            
            if match_data and isinstance(match_data, dict):
                lwin = match_data.get("lwin")
                display_name = match_data.get("display_name", query)
                if lwin:
                    results[query] = {
                        "display_name": display_name,
                        "lwin": str(lwin)
                    }
                else:
                    results[query] = {"display_name": query, "lwin": None}
            else:
                results[query] = {"display_name": query, "lwin": None}
        else:
            results[query] = {"display_name": query, "lwin": None}
    
    return results


def get_retail_price(lwin: str) -> dict:
    """
    Get retail price with dual-endpoint strategy:
    1. Try /listings (US) first - most specific
    2. Fallback to /price_stats (North America) if no US listings
    
    Returns dict with: price, count, source, listings
    """
    result = {
        "price": None,
        "count": 0,
        "source": None,  # "us_listings" or "na_stats"
        "listings": []   # Individual listing details for expandable UI
    }
    
    if not lwin:
        return result
    
    # Step 1: Try US listings first
    listings_payload = {
        "user_id": WINE_LABS_USER_ID,
        "lwin": lwin,
        "currency": "USD",
        "countries": ["US"],
        "limit": 100,
        "offset": 0
    }
    
    listings_data = cached_post("/listings", listings_payload)
    
    if listings_data:
        listings_results = listings_data.get("results", [])
        
        if listings_results:
            # Extract prices and listing details
            prices = []
            listings_info = []
            
            for listing in listings_results:
                price = listing.get("offer_price")
                if price is None:
                    price = listing.get("offer_adjusted_price")
                
                if price is not None:
                    try:
                        price_float = float(price)
                        prices.append(price_float)
                        
                        # Store listing details for UI
                        listings_info.append({
                            "retailer": listing.get("shop_name", "Unknown"),
                            "price": price_float,
                            "state": listing.get("shop_state", ""),
                            "url": listing.get("offer_url", ""),
                            "last_check": listing.get("last_check_at", "")
                        })
                    except (ValueError, TypeError):
                        pass
            
            if prices:
                result["price"] = sum(prices) / len(prices)
                result["count"] = len(prices)
                result["source"] = "us_listings"
                result["listings"] = listings_info
                return result
    
    # Step 2: Fallback to North America price stats
    stats_payload = {
        "user_id": WINE_LABS_USER_ID,
        "lwin": lwin,
        "region": "north america",
        "currency": "USD"
    }
    
    stats_data = cached_post("/price_stats", stats_payload)
    
    if stats_data:
        stats_results = stats_data.get("results", [])
        
        if stats_results:
            stats = stats_results[0]
            winelabs_price = stats.get("winelabs_price")
            
            if winelabs_price is not None:
                result["price"] = float(winelabs_price)
                result["count"] = stats.get("count", 0)
                result["source"] = "na_stats"
                # No individual listings for stats endpoint
                return result
    
    return result


def get_critic_scores(lwin: str) -> dict:
    """
    Get critic scores for a wine using /critic_scores endpoint.
    
    Returns dict with: avg_score, score_count, scores (individual critic scores)
    """
    result = {
        "avg_score": None,
        "score_count": 0,
        "scores": []  # Individual critic scores for potential expandable UI
    }
    
    if not lwin:
        return result
    
    payload = {
        "user_id": WINE_LABS_USER_ID,
        "query": lwin  # Use LWIN as the query
    }
    
    data = cached_post("/critic_scores", payload)
    
    if data is None:
        return result
    
    scores_results = data.get("results", [])
    
    if not scores_results:
        return result
    
    # Extract scores and calculate average
    valid_scores = []
    scores_info = []
    
    for score_entry in scores_results:
        score = score_entry.get("review_score")
        out_of = score_entry.get("out_of", 100)
        
        if score is not None:
            try:
                score_float = float(score)
                out_of_float = float(out_of) if out_of else 100
                
                # Normalize to 100-point scale if needed
                if out_of_float != 100 and out_of_float > 0:
                    normalized_score = (score_float / out_of_float) * 100
                else:
                    normalized_score = score_float
                
                valid_scores.append(normalized_score)
                
                # Store score details
                scores_info.append({
                    "critic": score_entry.get("critic_title", "Unknown"),
                    "score": score_float,
                    "out_of": out_of_float,
                    "normalized_score": normalized_score,
                    "review_date": score_entry.get("tasting_date", ""),
                    "url": score_entry.get("url", "")
                })
            except (ValueError, TypeError):
                pass
    
    if valid_scores:
        result["avg_score"] = sum(valid_scores) / len(valid_scores)
        result["score_count"] = len(valid_scores)
        result["scores"] = scores_info
    
    return result


def process_wines(wines: List[Dict]) -> List[Dict]:
    """
    Process extracted wines: lookup retail prices, scores, and calculate markups.
    Handles by-the-glass wines with bottle price estimation.
    """
    # Build queries for LWIN matching
    queries = []
    for wine in wines:
        name = wine.get("name", "")
        vintage = wine.get("vintage")
        if vintage:
            query = f"{name} {vintage}"
        else:
            query = name
        queries.append(query)
    
    # Match all wines to LWIN codes
    wine_matches = match_wines_to_lwin(queries)
    
    # Process each wine
    results = []
    for i, wine in enumerate(wines):
        query = queries[i]
        match_info = wine_matches.get(query, {"display_name": query, "lwin": None})
        
        display_name = match_info["display_name"]
        lwin = match_info["lwin"]
        
        # Get menu price (original price from menu)
        original_menu_price = wine.get("price")
        if original_menu_price is not None:
            try:
                original_menu_price = float(original_menu_price)
            except (ValueError, TypeError):
                original_menu_price = None
        
        # Get retail price (US listings first, then NA stats fallback)
        price_data = get_retail_price(lwin)
        retail_price = price_data["price"]
        listings_count = price_data["count"]
        price_source = price_data["source"]
        listings_details = price_data["listings"]
        
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
        
        # Get critic scores
        scores_data = get_critic_scores(lwin)
        avg_score = scores_data["avg_score"]
        score_count = scores_data["score_count"]
        critic_scores = scores_data["scores"]
        
        # Calculate markup (using estimated bottle price for glass pours)
        if menu_price is not None and retail_price is not None:
            markup_dollars = menu_price - retail_price
            markup_percent = (markup_dollars / retail_price) * 100
        else:
            markup_dollars = None
            markup_percent = None
        
        # Calculate score per dollar (score / menu price - using bottle price for glass pours)
        if avg_score is not None and menu_price is not None and menu_price > 0:
            score_per_dollar = avg_score / menu_price
        else:
            score_per_dollar = None
        
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
            "listings": listings_details,
            "avg_score": avg_score,
            "score_count": score_count,
            "critic_scores": critic_scores,
            "score_per_dollar": score_per_dollar,
            "markup_dollars": markup_dollars,
            "markup_percent": markup_percent,
            "matched": lwin is not None
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
    
    Returns JSON with analyzed wine data and markups.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        image_base64 = data.get("image")
        media_type = data.get("media_type", "image/jpeg")
        
        if not image_base64:
            return jsonify({"error": "No image provided"}), 400
        
        # Remove data URL prefix if present
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        # Step 1: Analyze menu with Claude Vision
        print("Analyzing menu with Claude Vision...")
        wines = analyze_menu_with_claude(image_base64, media_type)
        
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
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Wine Menu Analyzer")
    print("=" * 60)
    print(f"Cache database: {CACHE_DB}")
    print(f"Cache TTL: {CACHE_TTL_HOURS} hours")
    print(f"Anthropic API Key: {'Set' if ANTHROPIC_API_KEY else 'NOT SET'}")
    print()
    
    if not ANTHROPIC_API_KEY:
        print("WARNING: ANTHROPIC_API_KEY environment variable not set!")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        print()
    
    app.run(debug=True, host="0.0.0.0", port=5000)

