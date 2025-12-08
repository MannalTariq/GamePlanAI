# GamePlanAI - Complete Deployment Guide

This guide walks you through deploying the entire GamePlanAI project to production, including frontend (Vercel), backend APIs (Render), and database.

## Prerequisites

- GitHub account with repository pushed
- Render account (free tier available)
- Vercel account (free tier available)
- All backend APIs tested locally

---

## Step 1: Deploy Backend APIs to Render

### 1.1 Authentication API (Port 5002)

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository**: `MannalTariq/GamePlanAI`
4. **Configure the service**:
   - **Name**: `gameplanai-auth`
   - **Region**: Oregon (or closest to you)
   - **Branch**: `main`
   - **Root Directory**: `database`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python auth_api.py`
   - **Plan**: Free

5. **Environment Variables**:
   ```
   PORT=5002
   JWT_SECRET_KEY=<generate-a-strong-secret-key>
   CORS_ORIGINS=https://your-frontend-url.vercel.app
   ```

6. **Click "Create Web Service"**
7. **Wait for deployment** (5-10 minutes)
8. **Copy the service URL** (e.g., `https://gameplanai-auth.onrender.com`)

### 1.2 Corner Kick API (Port 5000)

1. **Click "New +" → "Web Service"**
2. **Connect same GitHub repository**
3. **Configure**:
   - **Name**: `gameplanai-corner`
   - **Root Directory**: `backend/hassaa/data`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python api_server.py`
   - **Plan**: Free

4. **Environment Variables**:
   ```
   PORT=5000
   CORS_ORIGINS=https://your-frontend-url.vercel.app
   ```

5. **Deploy and copy URL** (e.g., `https://gameplanai-corner.onrender.com`)

### 1.3 Free Kick API (Port 5001)

1. **Click "New +" → "Web Service"**
2. **Configure**:
   - **Name**: `gameplanai-freekick`
   - **Root Directory**: `backend/hassaa/data`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python freekick_api.py`
   - **Plan**: Free

3. **Environment Variables**:
   ```
   PORT=5001
   CORS_ORIGINS=https://your-frontend-url.vercel.app
   ```

4. **Deploy and copy URL** (e.g., `https://gameplanai-freekick.onrender.com`)

---

## Step 2: Update Backend APIs for Production

### 2.1 Update Authentication API CORS

Edit `database/auth_api.py`:

```python
import os

# Update CORS configuration
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=CORS_ORIGINS)
```

### 2.2 Update Corner Kick API CORS

Edit `backend/hassaa/data/api_server.py`:

```python
import os

# Update CORS configuration
CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=CORS_ORIGINS)
```

### 2.3 Update Free Kick API CORS

Edit `backend/hassaa/data/freekick_api.py` (if exists):

```python
import os

CORS_ORIGINS = os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, origins=CORS_ORIGINS)
```

### 2.4 Update Port Configuration

All APIs should read port from environment:

```python
import os

PORT = int(os.environ.get('PORT', 5002))  # Default to original port
app.run(debug=False, host='0.0.0.0', port=PORT)
```

**Note**: Set `debug=False` for production!

---

## Step 3: Update Frontend Environment Variables

### 3.1 Create Production Environment File

Create `frontend/.env.production`:

```env
REACT_APP_AUTH_API=https://gameplanai-auth.onrender.com/api/auth
REACT_APP_CORNER_API=https://gameplanai-corner.onrender.com/api
REACT_APP_FK_API=https://gameplanai-freekick.onrender.com/api/freekick
```

**Replace URLs with your actual Render service URLs!**

### 3.2 Verify Frontend Code Uses Environment Variables

All frontend files should use:
- `process.env.REACT_APP_AUTH_API`
- `process.env.REACT_APP_CORNER_API`
- `process.env.REACT_APP_FK_API`

Check these files:
- ✅ `frontend/src/utils/auth.js`
- ✅ `frontend/src/js files/Login/login-body.js`
- ✅ `frontend/src/js files/Signup/signup-body.js`
- ✅ `frontend/src/js files/OPP-Service/Obody.js`
- ✅ `frontend/src/js files/Sim-Service/Simbody.js`
- ✅ `frontend/src/js files/ASP Service/Abody.js`

---

## Step 4: Deploy Frontend to Vercel

### 4.1 Connect Repository

1. **Go to Vercel Dashboard**: https://vercel.com
2. **Click "Add New Project"**
3. **Import GitHub repository**: `MannalTariq/GamePlanAI`

### 4.2 Configure Project

- **Framework Preset**: Create React App
- **Root Directory**: `frontend`
- **Build Command**: `npm run build`
- **Output Directory**: `build`
- **Install Command**: `npm install`

### 4.3 Add Environment Variables

In Vercel project settings → Environment Variables:

```
REACT_APP_AUTH_API=https://gameplanai-auth.onrender.com/api/auth
REACT_APP_CORNER_API=https://gameplanai-corner.onrender.com/api
REACT_APP_FK_API=https://gameplanai-freekick.onrender.com/api/freekick
```

**Important**: Add these for **Production**, **Preview**, and **Development** environments.

### 4.4 Deploy

1. **Click "Deploy"**
2. **Wait for build** (2-5 minutes)
3. **Copy deployment URL** (e.g., `https://gameplanai.vercel.app`)

---

## Step 5: Update Backend CORS with Frontend URL

### 5.1 Update Render Environment Variables

For each backend service on Render:

1. Go to service → **Environment** tab
2. Update `CORS_ORIGINS` to your Vercel URL:
   ```
   CORS_ORIGINS=https://gameplanai.vercel.app
   ```
3. **Redeploy** the service (or it will auto-redeploy)

---

## Step 6: Test Deployment

### 6.1 Test Authentication

1. Visit your Vercel URL
2. Try to sign up a new user
3. Try to log in
4. Verify JWT token is stored

### 6.2 Test Corner Kick API

1. Navigate to "Simulate Strategies"
2. Select "Corner Kick"
3. Place players and generate simulation
4. Verify API calls work

### 6.3 Test Free Kick API

1. Select "Free Kick"
2. Set ball position
3. Generate simulation
4. Verify API calls work

### 6.4 Test Optimization

1. Navigate to "Optimize Player Positioning"
2. Select team and set piece
3. Click "Optimize"
4. Verify results display

---

## Troubleshooting

### Backend APIs Not Responding

1. **Check Render logs**: Service → Logs tab
2. **Verify environment variables** are set correctly
3. **Check build logs** for dependency errors
4. **Verify Python version** (should be 3.9+)

### CORS Errors

1. **Verify CORS_ORIGINS** includes your Vercel URL
2. **Check frontend environment variables** are set
3. **Clear browser cache** and hard refresh
4. **Check browser console** for specific CORS error

### Frontend Build Fails

1. **Check Vercel build logs**
2. **Verify all dependencies** in `package.json`
3. **Check for TypeScript/ESLint errors**
4. **Ensure `.env.production`** is committed (or use Vercel env vars)

### Database Issues

1. **SQLite database** is created automatically on Render
2. **Database file** persists in Render's filesystem
3. **For production**, consider migrating to PostgreSQL (Render offers free tier)

---

## Production Checklist

- [ ] All three backend APIs deployed to Render
- [ ] Backend APIs use environment variables for CORS
- [ ] Backend APIs use PORT environment variable
- [ ] Frontend deployed to Vercel
- [ ] Frontend environment variables set in Vercel
- [ ] CORS_ORIGINS updated with Vercel URL
- [ ] All APIs tested and working
- [ ] Authentication flow works
- [ ] Corner kick simulation works
- [ ] Free kick simulation works
- [ ] Optimization service works
- [ ] Error handling works correctly

---

## Cost Estimate

### Free Tier (Sufficient for MVP)

- **Render**: 3 web services × Free tier = $0/month
- **Vercel**: Free tier = $0/month
- **Total**: $0/month

### Limitations

- Render free tier services **spin down after 15 minutes** of inactivity
- First request after spin-down takes **30-60 seconds** to wake up
- Consider upgrading to paid tier for production use

---

## Next Steps

1. **Set up custom domain** (optional)
2. **Enable HTTPS** (automatic on Vercel/Render)
3. **Set up monitoring** (Render/Vercel provide basic monitoring)
4. **Migrate database** to PostgreSQL for better reliability
5. **Add CI/CD** for automatic deployments
6. **Set up error tracking** (Sentry, etc.)

---

## Support

If you encounter issues:

1. Check Render service logs
2. Check Vercel build/deployment logs
3. Check browser console for frontend errors
4. Verify all environment variables are set correctly
5. Test APIs individually using Postman/curl

---

## Quick Reference

### Render Service URLs
- Auth API: `https://gameplanai-auth.onrender.com`
- Corner API: `https://gameplanai-corner.onrender.com`
- Free Kick API: `https://gameplanai-freekick.onrender.com`

### Vercel Frontend URL
- Frontend: `https://gameplanai.vercel.app`

### Environment Variables Template

**Frontend (.env.production):**
```env
REACT_APP_AUTH_API=https://gameplanai-auth.onrender.com/api/auth
REACT_APP_CORNER_API=https://gameplanai-corner.onrender.com/api
REACT_APP_FK_API=https://gameplanai-freekick.onrender.com/api/freekick
```

**Backend (Render Environment Variables):**
```
PORT=5002 (or 5000, 5001)
JWT_SECRET_KEY=<your-secret-key>
CORS_ORIGINS=https://gameplanai.vercel.app
```

