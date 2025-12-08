# Vercel Environment Variables Setup

## Required Environment Variables

Add these in **Vercel Dashboard** → **Your Project** → **Settings** → **Environment Variables**

### For Production Environment:

```
REACT_APP_AUTH_API=https://gameplanai-auth.onrender.com/api/auth
REACT_APP_CORNER_API=https://gameplanai-corner.onrender.com/api
REACT_APP_FK_API=https://gameplanai-freekick.onrender.com/api/freekick
```

### For Preview Environment (optional, same as production):

```
REACT_APP_AUTH_API=https://gameplanai-auth.onrender.com/api/auth
REACT_APP_CORNER_API=https://gameplanai-corner.onrender.com/api
REACT_APP_FK_API=https://gameplanai-freekick.onrender.com/api/freekick
```

### For Development Environment (optional, for local testing):

```
REACT_APP_AUTH_API=http://localhost:5002/api/auth
REACT_APP_CORNER_API=http://localhost:5000/api
REACT_APP_FK_API=http://localhost:5001/api/freekick
```

## Steps to Add in Vercel:

1. Go to https://vercel.com/dashboard
2. Select your **GamePlanAI** project
3. Click **Settings** tab
4. Click **Environment Variables** in the sidebar
5. Add each variable:
   - **Key**: `REACT_APP_AUTH_API`
   - **Value**: `https://gameplanai-auth.onrender.com/api/auth`
   - **Environment**: Select **Production**, **Preview**, and **Development**
   - Click **Save**
6. Repeat for `REACT_APP_CORNER_API` and `REACT_APP_FK_API`

## Important Notes:

- **Replace URLs** with your actual Render service URLs after deployment
- Environment variables starting with `REACT_APP_` are automatically available in React
- After adding variables, **redeploy** your project for changes to take effect
- Variables are case-sensitive

## Verify Setup:

After deployment, check browser console to verify API calls are using production URLs (not localhost).

## Files Already Updated:

All frontend files have been updated to use environment variables:
- ✅ `frontend/src/utils/auth.js`
- ✅ `frontend/src/js files/Login/login-body.js`
- ✅ `frontend/src/js files/Signup/signup-body.js`
- ✅ `frontend/src/js files/OPP-Service/Obody.js`
- ✅ `frontend/src/js files/Sim-Service/Simbody.js`
- ✅ `frontend/src/js files/ASP Service/Abody.js`
- ✅ `frontend/src/config/api.js`

No code changes needed - just add environment variables in Vercel!

