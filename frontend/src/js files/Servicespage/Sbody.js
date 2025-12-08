import React from 'react';
import { Link } from 'react-router-dom'; // Import Link for navigation
import tickSvg from '../../images/tick.svg';
import analyzeIcon from '../../images/set-piece_service.svg';
import optimizeIcon from '../../images/optimize_service.svg';
import simulateIcon from '../../images/simulate_service.svg';

const Sbody = () => {
  return (
    <div className="sbody">
      {/* Analyze set piece
      <Link to="/analyze-set-piece" className="card-link">
        <section className="sbody-section card-grid">
          <div className="card-header">
            <h2>Analyze set piece</h2>
            <img src={analyzeIcon} alt="Analyze Icon" className="section-icon" />
          </div>
          <p className="card-description">
            It processes the data and tracks how players move over time and looks at how effective the team's setup is
            compared to the opponent's defense. The system identifies patterns and weaknesses to analyze set-piece
            performance and provide insights.
          </p>
          <div className="card-columns">
            <div className="card">
              <div className="card-title">
                <img src={tickSvg} alt="Tick" className="tick-icon" />
                Strategy Development
              </div>
              <p className="card-text">
                Lorem fibble dronk zazzle. Wibberfloosh glimtack nebuloid splargh. Dingbleblat shnarf plomptoo kazoodle bramflit? Quorbat noogee snibblewump ferblax!
              </p>
            </div>
            <div className="card">
              <div className="card-title">
                <img src={tickSvg} alt="Tick" className="tick-icon" />
                Review
              </div>
              <p className="card-text">
                Lorem fibble dronk zazzle. Wibberfloosh glimtack nebuloid splargh. Dingbleblat shnarf plomptoo kazoodle bramflit? Quorbat noogee snibblewump ferblax!
              </p>
            </div>
          </div>
        </section>
      </Link>

      {/* Optimize player positioning */} 
{/*       
      <Link to="/optimize-player-positioning" className="card-link">
        <section className="sbody-section card-grid">
          <div className="card-header">
            <h2>Optimize player positioning</h2>
            <img src={optimizeIcon} alt="Optimize Icon" className="section-icon" />
          </div>
          <p className="card-description">
            The system uses predictive analytics to recommend the best player positioning. By analyzing provided data of player and opponent tactics etc. GamePlan AI suggests optimal placements for players in corner kicks, free kicks, and other set-piece scenarios. The system even provides a success probability for each setup, so coaches know exactly which positioning offers the best chances of success. This feature is designed to ensure that players are positioned in the most effective spots, making the team’s set-piece execution more powerful and impactful.
          </p>
          <div className="card-columns">
            <div className="card">
              <div className="card-title">
                <img src={tickSvg} alt="Tick" className="tick-icon" />
                Framework Development
              </div>
              <p className="card-text">
                Lorem fibble dronk razzle. Wibberfloosh glimtack nebuloid splargh. Dingbleblat shnarf plomptoo kazoodle bramflit? Quorbat noogee snibblewump ferblax!
              </p>
            </div>
            <div className="card">
              <div className="card-title">
                <img src={tickSvg} alt="Tick" className="tick-icon" />
                Review
              </div>
              <p className="card-text">
                Lorem fibble dronk razzle. Wibberfloosh glimtack nebuloid splargh. Dingbleblat shnarf plomptoo kazoodle bramflit? Quorbat noogee snibblewump ferblax!
              </p>
            </div>
          </div>
        </section>
      </Link> */ }

      {/* Simulate strategies */}
      <Link to="/simulate-strategies" className="card-link">
        <section className="sbody-section card-grid">
          <div className="card-header">
            <h2>Simulate strategies</h2>
            <img src={simulateIcon} alt="Simulate Icon" className="section-icon" />
          </div>
          <p className="card-description">
            GamePlan AI allows users to simulate different set-piece scenarios. By inputting various player movements and tactical setups, users can see how different strategies might play out in real-time.
          </p>
          <div className="card-columns">
            <div className="card">
              <div className="card-title">
                <img src={tickSvg} alt="Tick" className="tick-icon" />
                Tactic Creation
              </div>
              <p className="card-text">
                Design custom player setups and movements for corner and free kicks. Customize positions and roles to craft unique strategies. Perfect for tailoring plays to your team's strengths.
              </p>
            </div>
            <div className="card">
              <div className="card-title">
                <img src={tickSvg} alt="Tick" className="tick-icon" />
                Strategy Analysis
              </div>
              <p className="card-text">
                Evaluate predicted outcomes, success rates, and tactical alternatives in real-time. Review detailed insights to optimize your set-piece plans. Compare options to select the most effective approach.
              </p>
            </div>
          </div>
        </section>
      </Link>

      {/* Coming Soon */}
      <Link to="/" className="card-link">
        <section className="sbody-section card-grid">
          <div className="card-header">
            <h2>Coming Soon</h2>
            <img src={optimizeIcon} alt="Optimize Icon" className="section-icon" />
          </div>
          <p className="card-description">
            Lorem fibble dronk razzle. Wibberfloosh glimtack nebuloid splargh. Dingbleblat shnarf plomptoo kazoodle bramflit? Quorbat noogee snibblewump ferblax!
          </p>
          <div className="card-columns">
            <div className="card">
              <div className="card-title">
                <img src={tickSvg} alt="Tick" className="tick-icon" />
                Live Match
              </div>
              <p className="card-text">
                Lorem fibble dronk razzle. Wibberfloosh glimtack nebuloid splargh. Dingbleblat shnarf plomptoo kazoodle bramflit? Quorbat noogee snibblewump ferblax!
              </p>
            </div>
            <div className="card">
              <div className="card-title">
                <img src={tickSvg} alt="Tick" className="tick-icon" />
                Team Performance
              </div>
              <p className="card-text">
                Lorem fibble dronk razzle. Wibberfloosh glimtack nebuloid splargh. Dingbleblat shnarf plomptoo kazoodle bramflit? Quorbat noogee snibblewump ferblax!
              </p>
            </div>
          </div>
        </section>
      </Link>



    </div>
  );
};

export default Sbody;