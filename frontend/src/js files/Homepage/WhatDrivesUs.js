import React from 'react';
import whatDrivesUsImage from '../../images/whatDrivesUs.svg';

const WhatDrivesUs = () => {
  return (
    <div className="mission-vision-container">
      {/* Heading */}
      <div className="heading">
        <h1>WHAT DRIVES US</h1>
        <h2>We are here to help you</h2>
      </div>

      {/* Content Wrapper */}
      <div className="content-wrapper">
        {/* Mission and Vision Sections */}
        <div className="text-sections">
          {/* Mission */}
          <div className="section mission">
            <h3>OUR MISSION</h3>
            <p>
              At GamePlan AI, Our Mission Is To Empower Football Coaches By
              Automating The Analysis Of Set-Pieces. We Aim To Eliminate The
              Limitations Of Subjective Judgments, Helping Coaches And Teams
              Optimize Strategies For Better Positioning, Improved Decision
              Making, And Maximized Scoring Opportunities. Our Goal Is To
              Enhance Team Performance, Ensuring That Every Critical Moment In A
              Match Is Backed By Precise, Actionable Data.
            </p>
          </div>

          {/* Vision */}
          <div className="section vision">
            <h3>OUR VISION</h3>
            <p>
              Our Vision Is To Transform Football Strategy By Enhancing The
              Analysis Of Set Pieces With Advanced Technology. We Aim To Provide
              Coaches With Real-Time, Automated Insights To Help Teams Improve
              Their Performance, Optimize Strategies, And Make Informed
              Decisions, Setting New Standards In The Game.
            </p>
          </div>
        </div>

        {/* Football Field Image */}
        <div className="image-section">
          <img src={whatDrivesUsImage} alt="Football Field" />
        </div>
      </div>
    </div>
  );
};

export default WhatDrivesUs;
