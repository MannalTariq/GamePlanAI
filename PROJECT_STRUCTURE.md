# GamePlanAI - Project Structure & Tech Stack

## 1. Backend Tech Stack

### Primary Framework
**Flask (Python 3.x)**

### Backend APIs

#### Authentication API
- **Framework**: Flask 2.3.3
- **Port**: 5002
- **Entry File**: `database/auth_api.py`
- **Database**: SQLite (`database/database/users.db`)

#### Corner Kick API
- **Framework**: Flask
- **Port**: 5000
- **Entry File**: `backend/hassaa/data/api_server.py`
- **ML Framework**: PyTorch + PyTorch Geometric

#### Free Kick API
- **Framework**: Flask
- **Port**: 5001
- **Entry File**: `backend/hassaa/data/freekick_api.py`

### Backend Dependencies

#### Authentication API (`database/requirements.txt`)
```
Flask==2.3.3
Flask-CORS==4.0.0
bcrypt==4.0.1
PyJWT==2.8.0
```

#### Corner Kick & Free Kick APIs (`backend/hassaa/data/requirements.txt`)
```
pandas>=1.3.0
numpy<2
scikit-learn>=0.24.2
torch>=1.9.0
torch-geometric>=2.0.0
networkx>=2.6.3
matplotlib>=3.4.3
seaborn>=0.11.2
imbalanced-learn>=0.11.0
xgboost>=1.7.0
lightgbm>=3.3.0
Flask
Flask-CORS
```

### Database
- **Type**: SQLite
- **Location**: `database/database/users.db`
- **Schema**: User authentication (id, name, email, password_hash, created_at)

---

## 2. Backend Folder Structure

```
/backend
  /hassaa
    /data
      api_server.py              # Corner Kick API (Flask entry point)
      freekick_api.py             # Free Kick API (Flask entry point)
      strategy_maker.py            # ML strategy generation
      gnn_dataset.py              # Graph Neural Network dataset
      gnn_train.py                # GNN training scripts
      interactive_tactical_setup.py
      requirements.txt             # Backend ML dependencies
      /models
        gatv2_models.py           # GNN model definitions
      /processed_csv
        player_positions.csv
    /processed
    README.md
    cleanup_script.bat

/database
  auth_api.py                     # Authentication API (Flask entry point)
  user_schema.py                  # Database schema and operations
  requirements.txt                # Auth API dependencies
  README.md
  /database
    users.db                      # SQLite database file
```

### Flask Entry Points

1. **Authentication API**: `database/auth_api.py`
   ```python
   app = Flask(__name__)
   # Runs on port 5002
   ```

2. **Corner Kick API**: `backend/hassaa/data/api_server.py`
   ```python
   app = Flask(__name__)
   # Runs on port 5000
   ```

3. **Free Kick API**: `backend/hassaa/data/freekick_api.py`
   ```python
   app = Flask(__name__)
   # Runs on port 5001
   ```

---

## 3. Frontend Tech Stack

### Framework
**React 19.1.0** (Create React App)

### Key Dependencies (`frontend/package.json`)
```json
{
  "react": "^19.1.0",
  "react-dom": "^19.1.0",
  "react-router-dom": "^7.5.3",
  "react-scripts": "5.0.1",
  "chart.js": "^4.4.9",
  "react-chartjs-2": "^5.3.0",
  "jspdf": "^3.0.1",
  "html2canvas": "^1.4.1",
  "interactjs": "^1.10.27",
  "react-icons": "^5.5.0"
}
```

### Frontend Features
- React Router for client-side routing
- Chart.js for data visualization
- jsPDF + html2canvas for PDF export
- Interact.js for drag-and-drop functionality
- JWT-based authentication

---

## 4. Frontend Folder Structure

```
/frontend
  package.json
  package-lock.json
  vercel.json                    # Vercel deployment config
  VERCEL_ENV_SETUP.md           # Environment variables guide
  /public
    index.html
    favicon.ico
    logo.png
    manifest.json
    robots.txt
  /src
    App.js                       # Main app component with routing
    App.css
    App.test.js
    index.js                     # React entry point
    index.css
    /components                  # Main page components
      Homepage.js
      Login.js
      Signup.js
      Servicepage.js
      ASP-Service.js             # Analyze Set Piece
      OPP-Service.js             # Optimize Player Positioning
      SIM-Service.js             # Simulate Strategies
      AboutUs.js
      ContactUs.js
    /js files                    # Detailed component logic
      /Homepage
        Header.js
        Footer.js
        Services.js
        FAQ.js
        WhatDrivesUs.js
      /Login
        login-header.js
        login-body.js
      /Signup
        signup-header.js
        signup-body.js
      /OPP-Service
        Oheader.js
        Obody.js
      /SIM-Service
        Simheader.js
        Simbody.js
      /ASP Service
        Aheader.js
        Abody.js
      /Servicespage
        Sheader.js
        Sbody.js
    /css files                   # Component-specific styles
      /Homepage
        Header.css
        Footer.css
        Hero.css
        Services.css
        FAQ.css
        general.css
        WhatDrivesUs.css
      /Login
        login-header.css
        login-body.css
      /Signup
        signup-header.css
        signup-body.css
      /OPP-Service
        Oheader.css
        Obody.css
      /SIM-Service
        Simheader.css
        Simbody.css
      /ASP Service
        Aheader.css
        Abody.css
      /Servicespage
        Sheader.css
        Sbody.css
      /AboutUs
        aboutus.css
      /ContactUs
        contactus.css
    /images                      # SVG and image assets
      logo.svg
      field.svg
      player-iconB.svg
      player-iconR.svg
      [30+ SVG files]
    /config
      api.js                     # API configuration with env vars
    /utils
      auth.js                    # Authentication utilities
      placements.js              # Player placement utilities
```

---

## 5. Project Root Structure

```
/GamePlanAI
  /backend                       # ML backend APIs
  /database                      # Authentication API
  /frontend                      # React frontend
  .gitignore
  render.yaml                    # Render deployment config
  DEPLOYMENT_GUIDE.md           # Deployment instructions
  API_DOCUMENTATION.md           # Complete API documentation
  ENV_VARIABLES_REFERENCE.md     # Environment variables guide
  PROJECT_STRUCTURE.md           # This file
  start_all_servers.bat          # Windows startup script
  start_all_servers.sh           # Linux/Mac startup script
  test_api_connections.py        # API testing script
  test_api_flow.py
  test_signup.py
  test_simple.py
```

---

## 6. API Endpoints Summary

### Authentication API (Port 5002)
- `GET  /api/auth/health`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET  /api/auth/verify`
- `GET  /api/auth/profile`
- `POST /api/auth/logout`
- `POST /api/auth/change-password`

### Corner Kick API (Port 5000)
- `GET  /api/health`
- `POST /api/optimize`
- `POST /api/simulate`
- `POST /api/corner/left`
- `POST /api/corner/right`
- `GET  /api/strategies`
- `GET  /api/strategy/<filename>`

### Free Kick API (Port 5001)
- `GET  /api/freekick/health`
- `POST /api/freekick/position`
- `POST /api/freekick/simulate`

---

## 7. Development Setup

### Backend
```bash
# Authentication API
cd database
pip install -r requirements.txt
python auth_api.py

# Corner Kick API
cd backend/hassaa/data
pip install -r requirements.txt
python api_server.py

# Free Kick API
cd backend/hassaa/data
python freekick_api.py
```

### Frontend
```bash
cd frontend
npm install
npm start
```

### All Servers (Windows)
```bash
.\start_all_servers.bat
```

### All Servers (Linux/Mac)
```bash
chmod +x start_all_servers.sh
./start_all_servers.sh
```

---

## 8. Deployment

### Backend (Render)
- 3 separate web services
- Python environment
- Environment variables for CORS and PORT

### Frontend (Vercel)
- Create React App
- Environment variables for API URLs
- Build output: `build/` directory

---

## 9. Key Technologies

| Component | Technology |
|-----------|-----------|
| Frontend Framework | React 19.1.0 |
| Frontend Build Tool | Create React App (react-scripts) |
| Routing | React Router DOM 7.5.3 |
| Backend Framework | Flask 2.3.3 |
| Database | SQLite |
| ML Framework | PyTorch + PyTorch Geometric |
| Authentication | JWT (PyJWT) + bcrypt |
| Data Visualization | Chart.js 4.4.9 |
| PDF Export | jsPDF 3.0.1 |
| Drag & Drop | Interact.js 1.10.27 |

---

## 10. File Naming Conventions

- **Flask entry files**: `*_api.py` or `api_server.py`
- **React components**: PascalCase (e.g., `Homepage.js`)
- **CSS files**: kebab-case matching component (e.g., `login-body.css`)
- **Utility files**: camelCase (e.g., `auth.js`)
- **Config files**: lowercase (e.g., `api.js`)

