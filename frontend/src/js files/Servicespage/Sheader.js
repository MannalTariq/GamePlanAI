import React from 'react';
import { Link } from 'react-router-dom';
import logo from '../../images/logo.svg'; // Adjust path to your logo image
import phoneIcon from '../../images/phoneicon.svg'; // Import phone SVG
import emailIcon from '../../images/emailicon.svg'; // Import email SVG

const Sheader = () => {
  return (
    <header className="sheader">
      {/* Top Bar */}
      <div className="top-bar">
        <div className="top-bar-left">
          <a href="#terms">TERMS & CONDITIONS</a>
          <a href="#privacy">PRIVACY POLICY</a>
          <a href="#contact">CONTACT US</a>
        </div>
        <div className="top-bar-right">
          <select className="language-selector">
            <option>LANGUAGE: English</option>
          </select>
        </div>
      </div>

      {/* Logo and Contact Info */}
      <div className="sheader-main">
        <div className="logo">
          <img src={logo} alt="GamePlan Logo" className="logo-image" />
        </div>
        <div className="contact-info">
          <div className="contact-item">
            <img src={phoneIcon} alt="Phone Icon" className="contact-icon" />
            <span className="contact-label">PHONE</span>
            <span className="contact-value">(050) 537 6669 Call</span>
          </div>
          <div className="contact-item">
            <img src={emailIcon} alt="Email Icon" className="contact-icon" />
            <span className="contact-label">EMAIL</span>
            <span className="contact-value">info@gameplan.net</span>
          </div>
        </div>
          <Link to="/contact" className="contact-button">
            Contact Us
          </Link>
        </div>

      <div className="nav-bar">
        <Link to="/" className="nav-link">HOME</Link>
        <Link to="/about" className="nav-link">About Us</Link>
        <Link to="/services" className="nav-link">Services</Link>
        <Link to="/contact" className="nav-link">Contact Us</Link>
        </div>

        <section className="sheader-hero"></section>
    </header>
  );
};

export default Sheader;