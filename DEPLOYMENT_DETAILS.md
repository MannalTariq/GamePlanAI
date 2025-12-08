# GamePlanAI - Deployment Details & Answers

## 1. What framework is your project using?

**✅ Flask (Python)**

Your project uses **Flask** for all backend APIs:

- **Authentication API**: `database/auth_api.py` - Flask 2.3.3
- **Corner Kick API**: `backend/hassaa/data/api_server.py` - Flask
- **Free Kick API**: `backend/hassaa/data/freekick_api.py` - Flask

**Entry Points:**
```python
# database/auth_api.py
app = Flask(__name__)

# backend/hassaa/data/api_server.py
app = Flask(__name__)

# backend/hassaa/data/freekick_api.py
app = Flask(__name__)
```

---

## 2. Does your code import any environment variables?

**✅ YES** - Your code uses `os.environ.get()` to read environment variables.

### Environment Variables Used:

#### Authentication API (`database/auth_api.py`):
```python
# Line 28
cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')

# Line 32
app.config['SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production')

# Line 465
port = int(os.environ.get('PORT', 5002))

# Line 466
debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
```

#### Corner Kick API (`backend/hassaa/data/api_server.py`):
```python
# Line 27
cors_origins = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')

# Line 752
port = int(os.environ.get('PORT', 5000))

# Line 753
debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
```

### Environment Variables Summary:

| Variable | Used In | Default Value | Purpose |
|----------|---------|---------------|---------|
| `PORT` | All APIs | 5002, 5000, 5001 | Server port (Render sets this automatically) |
| `CORS_ORIGINS` | All APIs | `http://localhost:3000` | Allowed CORS origins (comma-separated) |
| `JWT_SECRET_KEY` | Auth API | `'your-secret-key-change-in-production'` | JWT token signing key |
| `FLASK_DEBUG` | All APIs | `'False'` | Enable/disable debug mode |

---

## 3. Are you using a .env file locally?

**❌ NO** - No `.env` file exists in the repository.

**Reason**: `.env` files are gitignored (see `.gitignore` line 34).

**For Local Development:**
- Environment variables are not required locally
- Code uses default values (localhost URLs, default ports)
- APIs work out of the box for local testing

**For Production (Render):**
- Set environment variables in Render Dashboard → Service → Environment
- No `.env` file needed - Render handles environment variables directly

**If you want to create a `.env` file locally** (optional):
```env
# .env (for local development only - DO NOT COMMIT)
PORT=5002
CORS_ORIGINS=http://localhost:3000
JWT_SECRET_KEY=your-local-secret-key
FLASK_DEBUG=True
```

**Note**: This file should NOT be committed to git (already in `.gitignore`).

---

## 4. Does your app connect to a database?

**✅ YES - SQLite**

### Database Details:

- **Type**: SQLite
- **Location**: `database/database/users.db`
- **Schema File**: `database/user_schema.py`
- **Auto-created**: Yes (database is created automatically on first run)

### Database Connection Code:

**File**: `database/user_schema.py`

```python
import sqlite3

class UserSchema:
    def __init__(self, db_path: str = "database/users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
```

### Database Usage:

- **Authentication API** uses SQLite for user storage
- Database file is created automatically in `database/database/` directory
- No external database connection required
- Works out of the box on Render (SQLite files persist in Render's filesystem)

### For Production Deployment:

**Current Setup (SQLite):**
- ✅ Works on Render (free tier)
- ✅ No additional configuration needed
- ✅ Database persists automatically

**Optional Upgrade (PostgreSQL):**
- Render offers free PostgreSQL database
- Would require code changes to use `psycopg2` instead of `sqlite3`
- Better for production with multiple instances

---

## 5. What exactly happens when you click "Deploy"?

**Status**: Not yet deployed (based on current state)

### Expected Deployment Flow:

#### On Render:

1. **Connect GitHub Repository**
   - Render detects `render.yaml` (if using blueprint)
   - Or manually configure each service

2. **Build Process**:
   ```
   - Install Python dependencies from requirements.txt
   - Set environment variables
   - Start Flask app
   ```

3. **Potential Issues**:

   **If deployment fails:**
   - Check Render logs for errors
   - Common issues:
     - Missing dependencies in `requirements.txt`
     - Port not set correctly (should use `PORT` env var)
     - Database path issues (SQLite file permissions)
     - Missing ML model files (`.pt` files)

   **If deployment succeeds but API doesn't work:**
   - Check CORS configuration
   - Verify environment variables are set
   - Check service logs for runtime errors

#### On Vercel:

1. **Connect GitHub Repository**
   - Vercel detects React app
   - Uses `frontend/` as root directory

2. **Build Process**:
   ```
   - npm install
   - npm run build
   - Deploy build/ directory
   ```

3. **Potential Issues**:

   **If build fails:**
   - Check Vercel build logs
   - Common issues:
     - Missing environment variables
     - Build errors in React code
     - Missing dependencies

   **If deployment succeeds but frontend doesn't work:**
   - Check browser console for API connection errors
   - Verify environment variables are set in Vercel
   - Check CORS errors (backend CORS_ORIGINS must include Vercel URL)

---

## Deployment Checklist

### Backend (Render):

- [ ] All 3 services deployed
- [ ] Environment variables set:
  - [ ] `PORT` (auto-set by Render)
  - [ ] `CORS_ORIGINS` (your Vercel URL)
  - [ ] `JWT_SECRET_KEY` (Auth API only)
- [ ] Services show "Live" status
- [ ] Health check endpoints respond

### Frontend (Vercel):

- [ ] Project deployed
- [ ] Environment variables set:
  - [ ] `REACT_APP_AUTH_API`
  - [ ] `REACT_APP_CORNER_API`
  - [ ] `REACT_APP_FK_API`
- [ ] Build succeeds
- [ ] Frontend loads correctly

### Testing:

- [ ] Login/Signup works
- [ ] API calls succeed (check browser Network tab)
- [ ] No CORS errors
- [ ] Simulations work

---

## Quick Reference

### Environment Variables Needed:

**Render (Backend):**
```
PORT=5002 (or 5000, 5001) - Auto-set by Render
CORS_ORIGINS=https://your-frontend.vercel.app
JWT_SECRET_KEY=<generate-strong-key> - Auth API only
FLASK_DEBUG=False - Optional
```

**Vercel (Frontend):**
```
REACT_APP_AUTH_API=https://gameplanai-auth.onrender.com/api/auth
REACT_APP_CORNER_API=https://gameplanai-corner.onrender.com/api
REACT_APP_FK_API=https://gameplanai-freekick.onrender.com/api/freekick
```

---

## Summary

1. **Framework**: ✅ Flask (Python)
2. **Environment Variables**: ✅ Yes - Uses `os.environ.get()` for PORT, CORS_ORIGINS, JWT_SECRET_KEY, FLASK_DEBUG
3. **.env File**: ❌ No - Not using .env file (gitignored)
4. **Database**: ✅ SQLite (`database/database/users.db`)
5. **Deploy Status**: Ready for deployment - Follow `DEPLOYMENT_GUIDE.md`

