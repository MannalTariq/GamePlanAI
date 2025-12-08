import React from 'react';
import { Link } from 'react-router-dom';
import Header from '../js files/Homepage/Header';
import Footer from '../js files/Homepage/Footer';
import '../css files/AboutUs/aboutus.css';

const AboutUs = () => {
  return (
    <div>
      <Header />
      <div className="about-us-container">
        {/* Company Overview Section */}
        <div className="about-section">
          <div className="heading">
            <h1>ABOUT US</h1>
            <h2>Who We Are</h2>
          </div>
          <div className="content-section">
            <p className="overview-text">
              GamePlan AI is a cutting-edge technology company specializing in football analytics and artificial intelligence. 
              We operate in the sports technology industry, focusing on revolutionizing how football teams analyze and optimize 
              their set-piece strategies. Our platform combines advanced AI techniques with deep football knowledge to provide 
              coaches and teams with actionable insights that drive performance improvements.
            </p>
          </div>
        </div>

        {/* Mission & Vision Section */}
        <div className="mission-vision-section">
          <div className="section mission">
            <h3>OUR MISSION</h3>
            <p>
              At GamePlan AI, our mission is to empower football coaches by automating the analysis of set-pieces. 
              We aim to eliminate the limitations of subjective judgments, helping coaches and teams optimize strategies 
              for better positioning, improved decision making, and maximized scoring opportunities. Our goal is to 
              enhance team performance, ensuring that every critical moment in a match is backed by precise, actionable data.
            </p>
          </div>

          <div className="section vision">
            <h3>OUR VISION</h3>
            <p>
              Our vision is to transform football strategy by enhancing the analysis of set pieces with advanced technology. 
              We aim to provide coaches with real-time, automated insights to help teams improve their performance, optimize 
              strategies, and make informed decisions, setting new standards in the game.
            </p>
          </div>
        </div>

        {/* Story/Background Section */}
        <div className="story-section">
          <div className="heading">
            <h2>Our Story</h2>
          </div>
          <div className="content-section">
            <p>
              GamePlan AI was born from a passion for football and a recognition of the untapped potential in set-piece analysis. 
              Traditional methods of analyzing corner kicks, free kicks, and other set pieces relied heavily on manual observation 
              and subjective judgment, leaving significant room for improvement.
            </p>
            <p>
              Our founders, combining expertise in artificial intelligence, machine learning, and football analytics, set out to 
              create a solution that would revolutionize how teams approach set pieces. We developed advanced Graph Neural Network 
              (GNN) models trained on extensive football data to predict optimal player positioning, receiver selection, and shot 
              confidence with unprecedented accuracy.
            </p>
            <p>
              Since our inception, we've been committed to providing coaches and teams with the tools they need to make data-driven 
              decisions that translate directly to improved performance on the pitch.
            </p>
          </div>
        </div>

        {/* Values & Principles Section */}
        <div className="values-section">
          <div className="heading">
            <h2>Our Values & Principles</h2>
          </div>
          <div className="values-grid">
            <div className="value-card">
              <h4>Innovation</h4>
              <p>We continuously push the boundaries of AI and machine learning to deliver cutting-edge solutions.</p>
            </div>
            <div className="value-card">
              <h4>Accuracy</h4>
              <p>We prioritize precision and reliability in all our predictions and analyses.</p>
            </div>
            <div className="value-card">
              <h4>Transparency</h4>
              <p>We believe in clear, understandable insights that coaches can trust and act upon.</p>
            </div>
            <div className="value-card">
              <h4>Excellence</h4>
              <p>We are committed to delivering the highest quality tools and services to our users.</p>
            </div>
            <div className="value-card">
              <h4>User-Centric</h4>
              <p>Everything we build is designed with coaches and teams in mind, ensuring practical utility.</p>
            </div>
            <div className="value-card">
              <h4>Continuous Improvement</h4>
              <p>We constantly refine our models and features based on feedback and new data.</p>
            </div>
          </div>
        </div>

        {/* What Makes Us Unique Section */}
        <div className="unique-section">
          <div className="heading">
            <h2>What Makes Us Unique</h2>
          </div>
          <div className="content-section">
            <div className="unique-points">
              <div className="unique-point">
                <h4>Advanced AI Technology</h4>
                <p>Our Graph Neural Network models are specifically designed for football analytics, providing insights that traditional methods cannot match.</p>
              </div>
              <div className="unique-point">
                <h4>Real-Time Analysis</h4>
                <p>Get instant tactical recommendations and visualizations that adapt to your team's specific positioning and strategy.</p>
              </div>
              <div className="unique-point">
                <h4>Comprehensive Coverage</h4>
                <p>From corner kicks to free kicks, we analyze all types of set pieces with the same level of depth and accuracy.</p>
              </div>
              <div className="unique-point">
                <h4>Actionable Insights</h4>
                <p>Our platform doesn't just provide data—it delivers clear, actionable recommendations that coaches can implement immediately.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Call to Action Section */}
        <div className="cta-section">
          <div className="cta-content">
            <h2>Ready to Transform Your Set-Piece Strategy?</h2>
            <p>Join coaches and teams who are already using GamePlan AI to optimize their performance.</p>
            <div className="cta-buttons">
              <Link to="/services" className="cta-button primary">View Our Services</Link>
              <Link to="/contact" className="cta-button secondary">Get in Touch</Link>
            </div>
          </div>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default AboutUs;

