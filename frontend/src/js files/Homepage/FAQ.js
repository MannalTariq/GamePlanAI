import React, { useState } from 'react';
import faqimage from '../../images/FAQ.svg';

const FAQ = () => {
  const [activeIndex, setActiveIndex] = useState(null);

  const toggleAnswer = (index) => {
    setActiveIndex(activeIndex === index ? null : index);
  };

  const faqs = [
    {
      question: 'Is AI suitable for football predictions?',
      answer: 'GamePlan AI analyzes match data and player positioning to predict success probabilities for different strategies.',
    },
    {
      question: 'What are set-pieces, and why are they important?',
      answer: 'Set-pieces are crucial in football for creating scoring opportunities. AI can optimize them by analyzing historical data.',
    },
    {
      question: 'How accurate are the predictions of GamePlan AI?',
      answer: 'GamePlan AI has high accuracy, powered by detailed data analysis of past games and player positions.',
    },
    {
      question: 'How does GamePlan AI work?',
      answer: 'GamePlan AI predicts set-piece success by analyzing detailed historical match data and player positions. The system calculates success probabilities for different strategies.',
    },
  ];

  return (
    <section className="faq-container">
      <div className="faq-left">
        <h1>FAQ'S</h1>
        <div className="faq-list">
          {faqs.map((faq, index) => (
            <div key={index} className="faq-item">
              <div className={`faq-question ${activeIndex === index ? 'active' : ''}`}
                onClick={() => toggleAnswer(index)}>
                <span>{faq.question}</span>
                <span className={`arrow ${activeIndex === index ? 'rotate' : ''}`}>▸</span>
              </div>
              {activeIndex === index && <div className="faq-answer">{faq.answer}</div>}
            </div>
          ))}
        </div>
      </div>
      <div className="faq-right">
        <img src={faqimage} alt="Question Mark" />
      </div>
    </section>
  );
};

export default FAQ;