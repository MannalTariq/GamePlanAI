import React, { useState, useRef } from 'react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { FaChevronDown } from 'react-icons/fa';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';
import playerIconB from '../../images/player-iconB.svg'
import playerIconR from '../../images/player-iconR.svg'

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const Obody = () => {
  const [selectedTeam, setSelectedTeam] = useState('');
  const [selectedSetPiece, setSelectedSetPiece] = useState('');
  const [isOptimized, setIsOptimized] = useState(false);
  const [isTeamOpen, setIsTeamOpen] = useState(false);
  const [isSetPieceOpen, setIsSetPieceOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const chartRef = useRef(null);

  const teamOptions = ['Team A', 'Team B'];
  const setPieceOptions = ['Corner Kick', 'Free Kick'];

  const players = [
    { id: 1, name: 'Ivor John Allchurch', role: 'Central Attacking Midfielder / Forward' },
    { id: 2, name: 'Viv Anderson', role: 'Right-back or Right-wing Back' },
    { id: 3, name: 'Kenneth George Aston', role: 'Central Defender' },
  ];

  const optimizationInsight = `
    Best Player Positioning for Maximum Success: Ivor Allchurch should be positioned at the near post, where he has the best chance.
  `;

  const [successRateData, setSuccessRateData] = useState({
    labels: ['Success', 'Failure'],
    datasets: [
      {
        data: [78, 22],
        backgroundColor: ['#4caf50', '#C41616'],
        borderWidth: 0,
      },
    ],
  });

  const exportToPDF = () => {
    const doc = new jsPDF({
      orientation: 'portrait',
      unit: 'mm',
      format: 'a4',
    });

    const tempContainer = document.createElement('div');
    tempContainer.style.position = 'absolute';
    tempContainer.style.left = '-9999px';
    document.body.appendChild(tempContainer);

    const reportSection = document.querySelector('.report-section');
    if (reportSection) {
      const reportClone = reportSection.cloneNode(true);
      const printButton = reportClone.querySelector('.print-button');
      if (printButton) printButton.remove();
      tempContainer.appendChild(reportClone);

      const chartCanvas = reportClone.querySelector('.success-rate-chart canvas');
      if (chartCanvas && chartRef.current) {
        const chartInstance = chartRef.current.chartInstance;
        if (chartInstance) chartInstance.render();
      }

      setTimeout(() => {
        html2canvas(tempContainer, { useCORS: true, scale: 2 }).then((canvas) => {
          const imgData = canvas.toDataURL('image/png');
          const imgWidth = 210;
          const pageHeight = 297;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;
          let heightLeft = imgHeight;
          let position = 0;

          doc.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
          heightLeft -= pageHeight;

          while (heightLeft > 0) {
            position = heightLeft - imgHeight;
            doc.addPage();
            doc.addImage(imgData, 'PNG', 0, position, imgWidth, imgHeight);
            heightLeft -= pageHeight;
          }

          doc.save('report.pdf');
          document.body.removeChild(tempContainer);
        });
      }, 500);
    }
  };

  const handleOptimize = async () => {
    if (!selectedTeam || !selectedSetPiece) {
      alert('Please select both team and set-piece');
      return;
    }

    try {
      setIsLoading(true);
      
      // Default player positions for optimization
      const defaultPlayers = [
        { id: 1, x: 90, y: 30, team: 'attacker' },
        { id: 2, x: 85, y: 35, team: 'attacker' },
        { id: 3, x: 88, y: 40, team: 'attacker' },
        { id: 4, x: 92, y: 25, team: 'attacker' },
        { id: 5, x: 87, y: 50, team: 'defender' },
        { id: 6, x: 83, y: 45, team: 'defender' },
        { id: 7, x: 89, y: 55, team: 'defender' },
      ];

      const cornerPosition = selectedSetPiece === 'Corner Kick' 
        ? { x: 105, y: 0 } 
        : { x: 50, y: 30 };

      const apiUrl = selectedSetPiece === 'Free Kick'
        ? 'http://localhost:5001/api/freekick/position'
        : 'http://localhost:5000/api/optimize';

      const requestData = selectedSetPiece === 'Free Kick'
        ? { freekickPosition: { x: 50, y: 30 } }
        : {
            team: selectedTeam,
            setPiece: selectedSetPiece,
            players: defaultPlayers,
            cornerPosition: cornerPosition
          };

      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
      });

      if (!response.ok) {
        throw new Error(`API request failed: ${response.status}`);
      }

      const result = await response.json();
      setOptimizationResult(result);
      
      // Update success rate based on result
      if (result.success && result.predictions) {
        const confidence = result.predictions.shot_confidence 
          ? Math.round(result.predictions.shot_confidence * 100)
          : 78;
        setSuccessRateData({
          labels: ['Success', 'Failure'],
          datasets: [
            {
              data: [confidence, 100 - confidence],
              backgroundColor: ['#4caf50', '#C41616'],
              borderWidth: 0,
            },
          ],
        });
      }
      
      setIsOptimized(true);
    } catch (error) {
      console.error('Error optimizing:', error);
      alert('Error optimizing player positions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="Obody">
      {/* Selection Section */}
      <div className="obody-section selection-section">
        <div className="controls-container">
          <div className="dropdowns-row">
            <div className="custom-dropdown">
              <div
                className={`custom-dropdown-header ${isTeamOpen ? 'open' : ''}`}
                onClick={() => setIsTeamOpen(!isTeamOpen)}
              >
                {selectedTeam || 'Select Team'}
                <FaChevronDown className="arrow" />
              </div>
              {isTeamOpen && (
                <ul className="custom-dropdown-options">
                  {teamOptions.map((team, index) => (
                    <li
                      key={index}
                      className="custom-option"
                      onClick={() => {
                        setSelectedTeam(team);
                        setIsTeamOpen(false);
                      }}
                    >
                      {team}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div className="custom-dropdown">
              <div
                className={`custom-dropdown-header ${isSetPieceOpen ? 'open' : ''}`}
                onClick={() => setIsSetPieceOpen(!isSetPieceOpen)}
              >
                {selectedSetPiece || 'Select Set-piece'}
                <FaChevronDown className="arrow" />
              </div>
              {isSetPieceOpen && (
                <ul className="custom-dropdown-options">
                  {setPieceOptions.map((option, index) => (
                    <li
                      key={index}
                      className="custom-option"
                      onClick={() => {
                        setSelectedSetPiece(option);
                        setIsSetPieceOpen(false);
                      }}
                    >
                      {option}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          <div className="button-row">
            <button 
              className="optimize-button" 
              onClick={handleOptimize}
              disabled={!selectedTeam || !selectedSetPiece || isLoading}
            >
              {isLoading ? 'Optimizing...' : 'Optimize'}
            </button>
          </div>
        </div>
      </div>

      {/* Simulation Section */}
      {isOptimized && (
        <div className="obody-section simulation-section">
          <div className="field-simulation">
            <div className="field">
              <img
                src={playerIconB}
                alt="Player Blue"
                className="player player-blue"
                style={{ left: '20%', top: '30%' }}
              />
              <img
                src={playerIconB}
                alt="Player Blue"
                className="player player-blue"
                style={{ left: '30%', top: '40%' }}
              />
              <img
                src={playerIconB}
                alt="Player Blue"
                className="player player-blue"
                style={{ left: '40%', top: '50%' }}
              />
              <img
                src={playerIconB}
                alt="Player Blue"
                className="player player-blue"
                style={{ left: '47%', top: '4%' }}
              />
              <img
                src={playerIconR}
                alt="Player Red"
                className="player player-red"
                style={{ left: '42%', top: '55%' }}
              />
              <img
                src={playerIconR}
                alt="Player Red"
                className="player player-red"
                style={{ left: '32%', top: '45%' }}
              />
              <img
                src={playerIconR}
                alt="Player Red"
                className="player player-red"
                style={{ left: '22%', top: '35%' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Report Section */}
      {isOptimized && (
        <div className="obody-section report-section">
          <div className="report-content">
            <div className="report-text">
              <h3>Player Positioning</h3>
              <ul className="player-list">
                {players.map((player) => (
                  <li key={player.id}>
                    {player.name} - Role: {player.role}
                  </li>
                ))}
              </ul>
              <div className="optimization-insights">
                <h4>Optimization Insights</h4>
                <p>
                  {optimizationResult && optimizationResult.predictions ? (
                    <>
                      Primary Receiver: Player {optimizationResult.predictions.primary_receiver?.player_id || 'N/A'}<br/>
                      Shot Confidence: {optimizationResult.predictions.shot_confidence ? Math.round(optimizationResult.predictions.shot_confidence * 100) : 'N/A'}<br/>
                      Tactical Decision: {optimizationResult.predictions.tactical_decision || 'N/A'}<br/>
                      {optimizationResult.debug_info?.decision_reason && (
                        <>Decision Reason: {optimizationResult.debug_info.decision_reason}</>
                      )}
                    </>
                  ) : (
                    optimizationInsight
                  )}
                </p>
              </div>
            </div>
            <div className="success-rate-container">
              <h3 className="success-rate-title">Success Rate</h3>
              <div className="success-rate-chart">
                <Doughnut
                  ref={chartRef}
                  data={successRateData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    cutout: '70%',
                  }}
                />
              </div>
            </div>
          </div>
          <button className="print-button" onClick={exportToPDF}>
            Print Report
          </button>
        </div>
      )}
    </div>
  );
};

export default Obody;