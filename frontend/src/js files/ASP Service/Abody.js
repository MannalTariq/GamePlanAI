import React, { useState, useRef, useEffect } from 'react';
import { Doughnut } from 'react-chartjs-2';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { FaChevronDown } from 'react-icons/fa';
import jsPDF from 'jspdf';
import html2canvas from 'html2canvas';

// Register Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend);

const Abody = () => {
  const [selectedTeam, setSelectedTeam] = useState('');
  const [selectedStrategy, setSelectedStrategy] = useState('');
  const [selectedSetPiece, setSelectedSetPiece] = useState('');
  const [isAnalyzed, setIsAnalyzed] = useState(false);
  const [isTeamOpen, setIsTeamOpen] = useState(false);
  const [isSetPieceOpen, setIsSetPieceOpen] = useState(false);
  const [isStrategyOpen, setIsStrategyOpen] = useState(false);
  const [availableStrategies, setAvailableStrategies] = useState([]);
  const [strategyData, setStrategyData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const chartRef = useRef(null);

  const players = [
    { id: 1, name: 'Ivor John Allchurch', role: 'Central Attacking Midfielder / Forward' },
    { id: 2, name: 'Viv Anderson', role: 'Right-back or Right-wing Back' },
    { id: 3, name: 'Kenneth George Aston', role: 'Central Defender' },
  ];

  const optimizationInsight = `
    Best Player Positioning for Maximum Success: Ivor Allchurch should be positioned at the near post, where he has the best chance. 
    Additional analysis indicates that his positioning could be further optimized by coordinating with midfield runners to create space. 
    Viv Anderson's versatility allows for dynamic shifts between right-back and right-wing back, enhancing defensive stability and offensive support. 
    Kenneth George Aston's central defender role is critical for organizing the backline, and his positioning should focus on intercepting long balls. 
    Further tactical adjustments include increased pressure on the opposition's left flank to exploit weaknesses. 
    Training drills should emphasize quick transitions and set-piece execution to maximize these positional benefits. 
    Regular review of match footage will help refine these strategies over time.
  `;

  const cornerKickStrategies = [
    "In-swinger", "Out-swinger", "Short corner",
    "Zonal marking vs. Man marking (defensively)",
    "Near-post runs / Far-post overload",
    "Screening and blocking defenders",
    "Edge-of-the-box shooter",
    "Crowding the goalkeeper",
    "Decoy runs / Dummy runners",
    "Flick-ons / Knock-downs"
  ];

  const freeKickStrategies = [
    "Direct shot on goal", "Lay-off and shot", "Dummy run / Fake kick",
    "Wall splitting (using runners)", "Cross into the box",
    "Low-driven cross or pass", "Quick restart / Fast execution",
    "Over-the-wall curling shot"
  ];

  const teamOptions = [
    "Team A", "Team B"
  ];

  // Fetch available strategies when set piece is selected
  useEffect(() => {
    if (selectedSetPiece) {
      fetchStrategies();
    }
  }, [selectedSetPiece]);

  const fetchStrategies = async () => {
    try {
      setIsLoading(true);
      const response = await fetch(`${process.env.REACT_APP_CORNER_API || 'http://localhost:5000/api'}/strategies`);
      if (response.ok) {
        const data = await response.json();
        setAvailableStrategies(data.strategies || []);
      }
    } catch (error) {
      console.error('Error fetching strategies:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedStrategy) {
      alert('Please select a strategy to analyze');
      return;
    }

    try {
      setIsLoading(true);
      // Find the strategy filename from available strategies
      const strategy = availableStrategies.find(s => 
        s.filename === selectedStrategy || 
        s.filename.replace('.json', '') === selectedStrategy
      );
      
      const filename = strategy ? strategy.filename : selectedStrategy;
      const response = await fetch(`${process.env.REACT_APP_CORNER_API || 'http://localhost:5000/api'}/strategy/${filename}`);
      
      if (response.ok) {
        const data = await response.json();
        setStrategyData(data);
        setIsAnalyzed(true);
        
        // Update success rate based on actual data
        if (data.predictions && data.predictions.shot_confidence) {
          const confidence = Math.round(data.predictions.shot_confidence * 100);
          successRateData.datasets[0].data = [confidence, 100 - confidence];
        }
      } else {
        alert('Failed to load strategy data');
      }
    } catch (error) {
      console.error('Error analyzing strategy:', error);
      alert('Error analyzing strategy. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const successRateData = {
    labels: ['Success', 'Failure'],
    datasets: [
      {
        data: [68, 32],
        backgroundColor: ['#4caf50', '#C41616'],
        borderWidth: 0,
      },
    ],
  };

  const exportToPDF = () => {
    const doc = new jsPDF({
      orientation: 'portrait',
      unit: "mm",
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
      if (printButton) {
        printButton.remove();
      }

      tempContainer.appendChild(reportClone);

      const chartCanvas = reportClone.querySelector('.success-rate-chart canvas');
      if (chartCanvas && chartRef.current) {
        const chartInstance = chartRef.current.chartInstance;
        if (chartInstance) {
          chartInstance.render();
        }
      }

      setTimeout(() => {
        html2canvas(tempContainer, {
          useCORS: true,
          scale: 2,
        }).then((canvas) => {
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

  return (
    <div className="abody">
      {/* Selection Section */}
      <div className="abody-section selection-section">
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
                {selectedSetPiece || 'Select Set-Piece'}
                <FaChevronDown className="arrow" />
              </div>
              {isSetPieceOpen && (
                <ul className="custom-dropdown-options">
                  <li
                    className="custom-option"
                    onClick={() => {
                      setSelectedSetPiece('Corner Kick');
                      setIsSetPieceOpen(false);
                    }}
                  >
                    Corner Kick
                  </li>
                  <li
                    className="custom-option"
                    onClick={() => {
                      setSelectedSetPiece('Free Kick');
                      setIsSetPieceOpen(false);
                    }}
                  >
                    Free Kick
                  </li>
                </ul>
              )}
            </div>

            <div className="custom-dropdown">
              <div
                className={`custom-dropdown-header ${isStrategyOpen ? 'open' : ''} ${!selectedSetPiece ? 'disabled' : ''}`}
                onClick={() => selectedSetPiece && setIsStrategyOpen(!isStrategyOpen)}
              >
                {selectedStrategy || (isLoading ? 'Loading...' : 'Select Strategy')}
                <FaChevronDown className="arrow" />
              </div>
              {isStrategyOpen && selectedSetPiece && !isLoading && (
                <ul className="custom-dropdown-options">
                  {availableStrategies.length > 0 ? (
                    availableStrategies.map((strategy, index) => (
                      <li
                        key={index}
                        className="custom-option"
                        onClick={() => {
                          setSelectedStrategy(strategy.filename);
                          setIsStrategyOpen(false);
                        }}
                      >
                        {strategy.filename} - {new Date(strategy.timestamp).toLocaleString()}
                      </li>
                    ))
                  ) : (
                    <li className="custom-option disabled">No strategies available</li>
                  )}
                </ul>
              )}
            </div>
          </div>
          <div className="button-row">
            <button 
              className="analyze-button" 
              onClick={handleAnalyze}
              disabled={!selectedStrategy || isLoading}
            >
              {isLoading ? 'Analyzing...' : 'Analyze'}
            </button>
          </div>
        </div>
      </div>

      {/* Combined Report Section */}
      {isAnalyzed && (
        <div className="abody-section report-section">
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
                  {strategyData && strategyData.predictions ? (
                    <>
                      Primary Receiver: Player {strategyData.predictions.primary_receiver?.player_id || 'N/A'}<br/>
                      Shot Confidence: {strategyData.predictions.shot_confidence ? Math.round(strategyData.predictions.shot_confidence * 100) : 'N/A'}<br/>
                      Tactical Decision: {strategyData.predictions.tactical_decision || 'N/A'}<br/>
                      {strategyData.debug_info?.decision_reason && (
                        <>Decision Reason: {strategyData.debug_info.decision_reason}</>
                      )}
                    </>
                  ) : (
                    optimizationInsight
                  )}
                </p>
              </div>
            </div>
            <div className="success-rate-container">
              
              <div className="success-rate-chart">
                <Doughnut
                  ref={chartRef}
                  data={successRateData}
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    
                  }}
                />
              </div>
              <h3 className="success-rate-title">Success Rate</h3>
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

export default Abody;

