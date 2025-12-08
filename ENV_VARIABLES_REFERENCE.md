# Environment Variables Reference

## Frontend Environment Variables (Vercel)

Add these in Vercel Dashboard → Project Settings → Environment Variables

### Production
```env
REACT_APP_AUTH_API=https://gameplanai-auth.onrender.com/api/auth
REACT_APP_CORNER_API=https://gameplanai-corner.onrender.com/api
REACT_APP_FK_API=https://gameplanai-freekick.onrender.com/api/freekick
```

### Development (Local)
```env
REACT_APP_AUTH_API=http://localhost:5002/api/auth
REACT_APP_CORNER_API=http://localhost:5000/api
REACT_APP_FK_API=http://localhost:5001/api/freekick
```

---

## Backend Environment Variables (Render)

### Authentication API (`gameplanai-auth`)

**Required:**
```
PORT=5002
JWT_SECRET_KEY=<generate-strong-secret-key>
CORS_ORIGINS=https://your-frontend.vercel.app
```

**Optional:**
```
FLASK_DEBUG=False
```

### Corner Kick API (`gameplanai-corner`)

**Required:**
```
PORT=5000
CORS_ORIGINS=https://your-frontend.vercel.app
```

**Optional:**
```
FLASK_DEBUG=False
```

### Free Kick API (`gameplanai-freekick`)

**Required:**
```
PORT=5001
CORS_ORIGINS=https://your-frontend.vercel.app
```

**Optional:**
```
FLASK_DEBUG=False
```

---

## Generating JWT Secret Key

Use Python to generate a secure secret key:

```python
import secrets
print(secrets.token_urlsafe(32))
```

Or use OpenSSL:
```bash
openssl rand -base64 32
```

---

## Notes

- **CORS_ORIGINS**: Can be comma-separated list for multiple origins
- **PORT**: Render automatically sets PORT, but include it for clarity
- **FLASK_DEBUG**: Should be `False` in production
- Frontend env vars must start with `REACT_APP_` to be accessible in React

