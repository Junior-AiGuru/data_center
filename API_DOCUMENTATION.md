# MindTrip API Documentation

Base URL: `https://web-production-0e223.up.railway.app/`

All endpoints return JSON. All POST endpoints expect a JSON body with `Content-Type: application/json`.

---

## Place Object

Every place returned by the API has this shape:

```json
{
  "place_id": "ChIJ5XGt0hlBWBQREPek7GvvOgs",
  "name": "Lobby Lounge",
  "city": "Cairo",
  "city_en": "Cairo",
  "interests": ["Cafe", "Restaurants", "Waterfront", "Outdoor"],
  "category": "food_cafes",
  "price": 0,
  "rating": 4.9,
  "reviews_count": 359,
  "address": "1113 corniche El Nil, Ismailia, Qasr El Nil, Cairo Governorate 11221",
  "description": "",
  "photo_url": "https://res.cloudinary.com/...",
  "image_urls": [
    "https://res.cloudinary.com/...",
    "https://res.cloudinary.com/..."
  ],
  "opening_hours": "12am-12am",
  "lat": 30.0459449,
  "lng": 31.2321621,
  "is_hidden_gem": false,
  "day": "",
  "is_opened": "",
  "type": "",
  "cost": 0,
  "maps_url": "https://www.google.com/maps/place/?q=place_id:ChIJ5XGt0hlBWBQREPek7GvvOgs"
}
```

---

## Paginated Response

Endpoints that return lists use this wrapper:

```json
{
  "total": 787,
  "page": 1,
  "limit": 10,
  "total_pages": 79,
  "results": [ ... ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `total` | int | Total number of matching places (before pagination) |
| `page` | int | Current page number |
| `limit` | int | Number of results per page |
| `total_pages` | int | Total number of pages available |
| `results` | array | Array of place objects for this page |

---

## Null & Empty Values

The API is designed to be safe with null and empty values. You can send fields as `null`, `""`, `[]`, or simply omit them — they will all be **silently ignored** and no filtering will happen on that field.

This means the front-end can safely send the full request body every time without worrying about cleaning up empty fields first.

### What counts as "empty" (ignored)

| Value sent | Treated as | Result |
|------------|------------|--------|
| Field **omitted** from body | empty | Ignored — no filter applied |
| `null` | empty | Ignored — no filter applied |
| `""` (empty string) | empty | Ignored — no filter applied |
| `[]` (empty array) | empty | Ignored — no filter applied |
| `"Cairo"` or `["Cairo"]` | has value | Filter applied |

### Per-endpoint behavior

**`/places/getplaces`** — you can send all fields every time:
```json
{
  "city": null,
  "category": [],
  "interests": null,
  "min_rating": 4.0,
  "max_rating": null,
  "min_price": null,
  "max_price": 200,
  "hidden_gem": null,
  "sort_by": "rating",
  "order": "desc"
}
```
Only `min_rating` and `max_price` are applied here. Everything else is ignored.

**`/places/search`** — query and filters:
- `"query": null` or `"query": ""` → no text search, returns everything (with filters if provided)
- `"filters": null` or `"filters": {}` → no additional filtering

**`/places/home`** — city field:
- `"city": null` or `"city": ""` → returns data from all cities

**`/places/recommend`** — filters:
- `"filters": null` → no filtering, just recommendations
- `"selected_categories": []` → falls back to top-rated places

**`/places/top-rated`** — filters:
- `"filters": null` or `"filters": {}` → returns all places sorted by rating

### Inside the filters object (search, top-rated, recommend)

When using the `filters` object, individual keys inside it follow the same rule:
```json
{
  "filters": {
    "city_en": null,
    "category": "",
    "rating": { "gte": 4.0 }
  }
}
```
Only `rating >= 4.0` is applied. `city_en` and `category` are ignored.

---

## Reference Values

These are the possible values you can use in filters:

**Cities** (`city_en`):
`Alexandria`, `Aswan`, `Cairo`, `Fayoum`, `Giza`, `Hurghada`, `Ismailia`, `Luxor`, `Marsa Matrouh`, `Port Said`, `Sharm El Sheikh`

**Categories** (`category`):
`arts_culture`, `beaches`, `entertainment`, `food_cafes`, `historical_sites`, `nature`, `religious_sites`, `shopping`

**Interests** (fine-grained tags inside each place):
`Restaurants`, `Cafe`, `Bakery`, `Seafood`, `Street Food`, `Shopping`, `Arts & Crafts`, `Entertainment`, `Nightlife`, `Music`, `Nature`, `Park`, `Beaches & Water`, `Waterfront`, `History & Antiquities`, `Tourism`, `Mosques & Churches`, `Outdoor`


---

## Endpoints

---

### 1. Health Check

```
GET /
```

Returns a simple status message to verify the API is running.

**Response:**
```json
{
  "message": "MindTrip Recommendation API — Active!"
}
```

---

### 2. Home Screen

```
POST /places/home
```

Returns three sections for the home screen in a single call. Call this once when the home screen loads.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `city` | string | No | `null` | Filter all sections to a specific city (e.g. `"Cairo"`) |
| `seed` | int | No | random | Controls the randomization. Same seed = same results. Omit or change for a fresh mix |

**Example Request:**
```json
{
  "city": "Cairo",
  "seed": 42
}
```

**Example with no filters (all cities):**
```json
{}
```

**Response:**
```json
{
  "featured": [ ... ],
  "hidden_gems": [ ... ],
  "trending": [ ... ]
}
```

| Field | Type | Count | Description |
|-------|------|-------|-------------|
| `featured` | array | up to 5 | Top-rated places, one from each category for diversity |
| `hidden_gems` | array | up to 6 | Lesser-known but highly-rated places |
| `trending` | array | up to 8 | Most-reviewed popular places |

Each array contains place objects.

---

### 3. Recommendations

```
POST /places/recommend
```

Returns personalized place recommendations based on the user's selected interest categories. Results are shuffled but consistent within a session when using the same `seed`.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `selected_categories` | string[] | **Yes** | — | List of interest tags the user picked (e.g. `["Cafe", "Restaurants", "Nature"]`) |
| `filters` | object | No | `null` | Additional filters (see Filters section below) |
| `page` | int | No | `1` | Page number |
| `limit` | int | No | `10` | Results per page |
| `seed` | int | No | random | Session seed — keep the same seed while the user scrolls pages, change it on new session for fresh results |
| `pool_size` | int | No | `150` | How many top candidates to pick from before shuffling. Higher = more variety but lower average relevance |

**Example Request:**
```json
{
  "selected_categories": ["Cafe", "Restaurants", "Waterfront"],
  "filters": {
    "city_en": "Cairo"
  },
  "page": 1,
  "limit": 10,
  "seed": 12345
}
```

**Pagination pattern:**
- First load: `{ "page": 1, "seed": 12345 }`
- User scrolls: `{ "page": 2, "seed": 12345 }` (same seed!)
- New session/refresh: `{ "page": 1, "seed": 99999 }` (new seed)

**Response:** Paginated response (see format above).

---

### 4. Text Search

```
POST /places/search
```

Searches across place name, city, address, category, and interests. Supports multi-word queries — all words must match.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | No | `null` | Search text. Searches in: name, city, city_en, address, category, interests |
| `filters` | object | No | `null` | Additional filters applied after text matching (see Filters section) |
| `page` | int | No | `1` | Page number |
| `limit` | int | No | `10` | Results per page |

**Example — search by place name:**
```json
{
  "query": "Khan el-Khalili"
}
```

**Example — search by city:**
```json
{
  "query": "Luxor"
}
```

**Example — search with filters:**
```json
{
  "query": "cafe",
  "filters": {
    "city_en": "Cairo"
  }
}
```

**Example — no text, just filters:**
```json
{
  "query": "",
  "filters": {
    "category": "food_cafes"
  }
}
```

**Response:** Paginated response.

---

### 5. Top Rated

```
POST /places/top-rated
```

Returns places sorted by rating (adjusted for review count so places with very few reviews don't appear at the top).

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `filters` | object | No | `null` | Filter results (see Filters section) |
| `page` | int | No | `1` | Page number |
| `limit` | int | No | `10` | Results per page |

**Example Request:**
```json
{
  "filters": {
    "city_en": "Cairo",
    "category": "food_cafes"
  },
  "page": 1,
  "limit": 10
}
```

**Response:** Paginated response.

---

### 6. Get Places (General Filter)

```
POST /places/getplaces
```

General-purpose endpoint for fetching places with structured filters. All fields are optional — sending `{}` returns all places sorted by rating.

This is the most flexible endpoint. Use it when the user applies filters from a filter UI (city dropdown, category chips, price slider, etc.).

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `city` | string[] | No | `null` | Filter by city. Example: `["Cairo"]` or `["Cairo", "Giza"]` |
| `category` | string[] | No | `null` | Filter by category. Example: `["food_cafes"]` or `["food_cafes", "entertainment"]` |
| `interests` | string[] | No | `null` | Filter by interest tags (matches any). Example: `["Cafe", "Seafood"]` |
| `min_rating` | float | No | `null` | Minimum rating (inclusive). Example: `4.0` |
| `max_rating` | float | No | `null` | Maximum rating (inclusive). Example: `5.0` |
| `min_price` | int | No | `null` | Minimum price (inclusive). Example: `0` |
| `max_price` | int | No | `null` | Maximum price (inclusive). Example: `200` |
| `hidden_gem` | bool | No | `null` | Filter by hidden gem status. `true` = only hidden gems, `false` = only non-hidden gems |
| `sort_by` | string | No | `"rating"` | Sort field. Options: `"rating"`, `"reviews"`, `"price"`, `"name"` |
| `order` | string | No | `"desc"` | Sort direction. `"asc"` = ascending, `"desc"` = descending |
| `page` | int | No | `1` | Page number |
| `limit` | int | No | `10` | Results per page |

**Null/empty handling:** Any field set to `null`, `[]`, or omitted is ignored. Only fields with actual values are used as filters.

**Example — filter by city:**
```json
{
  "city": ["Cairo"]
}
```

**Example — multiple filters:**
```json
{
  "city": ["Cairo"],
  "category": ["food_cafes"],
  "min_rating": 4.0,
  "max_price": 200,
  "sort_by": "reviews",
  "order": "desc",
  "limit": 20
}
```

**Example — mixed null and real filters:**
```json
{
  "city": null,
  "category": [],
  "interests": null,
  "min_rating": 4.5,
  "max_price": 150,
  "limit": 5
}
```
This applies only `min_rating` and `max_price` — the null/empty fields are skipped.

**Example — all places, sorted by price ascending:**
```json
{
  "sort_by": "price",
  "order": "asc"
}
```

**Response:** Paginated response.

---

### 7. Single Place

```
GET /places/{place_id}
```

Returns full details for a single place by its ID.

**URL Parameter:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `place_id` | string | **Yes** | The unique place ID (e.g. `ChIJ5XGt0hlBWBQREPek7GvvOgs`) |

**Example Request:**
```
GET /places/ChIJ5XGt0hlBWBQREPek7GvvOgs
```

**Success Response (200):** A single place object (not wrapped in pagination).

```json
{
  "place_id": "ChIJ5XGt0hlBWBQREPek7GvvOgs",
  "name": "Lobby Lounge",
  "city": "Cairo",
  ...
}
```

**Error Response (404):**
```json
{
  "detail": "Place not found"
}
```

---

## Filters Object (for search, top-rated, recommend)

The `search`, `top-rated`, and `recommend` endpoints accept a `filters` object. This is a flexible key-value dictionary where the key is any field name from the place object.

**Important:** The `getplaces` endpoint does NOT use this `filters` object — it has its own dedicated fields instead.

### Filter types

**Exact match (single value):**
```json
{
  "filters": {
    "city_en": "Cairo"
  }
}
```

**Match any from list:**
```json
{
  "filters": {
    "category": ["food_cafes", "entertainment"]
  }
}
```

**Numeric range:**
```json
{
  "filters": {
    "rating": { "gte": 4.0, "lte": 5.0 },
    "price": { "gte": 0, "lte": 200 }
  }
}
```

Range operators: `gte` (>=), `lte` (<=), `gt` (>), `lt` (<). All are optional within the range object.

**Match any interest from list:**
```json
{
  "filters": {
    "interests": { "contains_any": ["Cafe", "Seafood"] }
  }
}
```

**Boolean:**
```json
{
  "filters": {
    "is_hidden_gem": true
  }
}
```

**Combined example:**
```json
{
  "filters": {
    "city_en": "Cairo",
    "category": ["food_cafes", "entertainment"],
    "rating": { "gte": 4.0 },
    "price": { "lte": 200 }
  }
}
```

---

## Quick Reference

| Endpoint | Method | Use Case |
|----------|--------|----------|
| `/` | GET | Health check |
| `/places/home` | POST | Home screen data (featured + hidden gems + trending) |
| `/places/recommend` | POST | Personalized recommendations based on user interests |
| `/places/search` | POST | Text search with optional filters |
| `/places/top-rated` | POST | Browse top-rated places with optional filters |
| `/places/getplaces` | POST | General filtering with dedicated filter fields |
| `/places/{place_id}` | GET | Get single place details |

---

## Error Handling

| Status Code | Meaning |
|-------------|---------|
| `200` | Success |
| `404` | Place not found (only from `/places/{place_id}`) |
| `422` | Validation error — the request body has wrong types or missing required fields. The response body will explain what's wrong |

Example 422 response:
```json
{
  "detail": [
    {
      "loc": ["body", "selected_categories"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```
