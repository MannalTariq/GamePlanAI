import React from 'react';
import logo from '../../images/logo.svg';
import fb from '../../images/fb.svg';
import instagram from '../../images/insta.svg';
import linkedin from '../../images/linkedin.svg';
import youtube from '../../images/youtube.svg';
import pitch from '../../images/footer.svg';
import { Link } from 'react-router-dom';


const Footer = () => {
  return (
    <footer>
      <div className="footer-top">
        <div className="footer-logo">
          <Link to="/">
            <img src={logo} alt="GamePlan AI Logo" />
          </Link>
        </div>

        <div className="footer-container">
          <div className="quick-links">
            <h3>Quick Links</h3>
            <ul>
              <li><Link to="/">Home</Link></li>
              <li><Link to="/services">Services</Link></li>
              <li><Link to="/about">About Us</Link></li>
              <li><Link to="/contact">Contact Us</Link></li>
            </ul>
          </div>

          <div className="about-us">
            <h3>About Us</h3>
            <p>GamePlan AI: Your Guide to Optimizing Set-Piece Strategies and Boosting Team Performance.</p>
            <div className="social-media">
              <a href="https://www.facebook.com" target="_blank" rel="noopener noreferrer">
                <img src={fb} alt="Facebook" />
              </a>
              <a href="https://www.instagram.com" target="_blank" rel="noopener noreferrer">
                <img src={instagram} alt="Instagram" />
              </a>
              <a href="https://www.linkedin.com" target="_blank" rel="noopener noreferrer">
                <img src={linkedin} alt="LinkedIn" />
              </a>
              <a href="https://www.youtube.com" target="_blank" rel="noopener noreferrer">
                <img src={youtube} alt="YouTube" />
              </a>
            </div>
          </div>

          <div className="newsletter">
            <h3>Newsletter</h3>
            <p>Stay up to date with our latest news, receive exclusive deals, and more.</p>
            <form className="newsletter-form">
              <input type="email" placeholder="Email Id" required className="email-input" />
              <button type="submit" className="submit-btn"></button>
            </form>
          </div>
        </div>

        <div className="footer-image">
          <img src={pitch} alt="Football Pitch" />
        </div>
      </div>

      <div className="footer-bottom">
        <p>
          Copyright ©2025 GAMEPLAN AI |{' '}
          <Link to="/terms">Terms & Conditions</Link> |{' '}
          <Link to="/privacy">Privacy Policy</Link>
        </p>
      </div>
    </footer>
  );
};

export default Footer;