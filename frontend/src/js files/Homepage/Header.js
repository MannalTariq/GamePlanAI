import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { authAPI } from '../../utils/auth';

const Header = () => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const checkAuth = async () => {
      const isAuthenticated = await authAPI.verifyToken();
      if (isAuthenticated) {
        const userData = authAPI.getUserData();
        setUser(userData);
      }
      setLoading(false);
    };
    
    checkAuth();
  }, []);

  const handleLogout = async () => {
    await authAPI.logout();
    setUser(null);
    navigate('/login');
  };

  return (
    <header className='header'>
      <div className="navbar-container">
        <nav>
          <ul>
            <li><Link to="/" className="navbar">Home</Link></li>
            <li><Link to="/services" className="navbar">Services</Link></li>
            <li><Link to="/about" className="navbar">About Us</Link></li>
            <li><Link to="/contact" className="navbar">Contact Us</Link></li>
          </ul>
        </nav>
        <div className="buttons-container">
          {loading ? (
            <div>Loading...</div>
          ) : user ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
              <span style={{ color: '#fff', fontSize: '14px' }}>
                Welcome, {user.name}
              </span>
              <button 
                className="signbutton" 
                onClick={handleLogout}
                style={{ backgroundColor: '#dc3545', border: 'none' }}
              >
                Logout
              </button>
            </div>
          ) : (
            <>
              <Link to="/signup">
                <button className="signbutton">Sign up</button>
              </Link>
              <Link to="/login">
                <button className="signbutton">Login</button>
              </Link>
            </>
          )}
        </div>
      </div>
      <Link to="/" style={{ textDecoration: 'none', color: 'inherit' }}>
        <div className="hero">GamePlan AI</div>
      </Link>
      <div className="hero-description">
        <h1>An AI solution for analyzing football set pieces using advanced AI techniques</h1>
        <div className="hero-buttons">
          <Link to="/services"><button>Strategize</button></Link>
          <Link to="/services"><button>Visualization</button></Link>
        </div>
      </div>
    </header>
  );
};

export default Header;