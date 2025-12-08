import React from 'react';
import Header from '../js files/Homepage/Header';
import WhatDrivesUs from '../js files/Homepage/WhatDrivesUs';
import Services from '../js files/Homepage/Services';
import FAQ from '../js files/Homepage/FAQ';
import Footer from '../js files/Homepage/Footer';




const HomePage = () => {
  return (
    <div>
      <Header />
      <WhatDrivesUs />
      <Services />
      <FAQ />
      <Footer />
    </div>
  );
};

export default HomePage;