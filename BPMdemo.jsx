import React, { useState } from 'react';
import { Play, RotateCcw, ArrowRight } from 'lucide-react';

const BPMXORDemo = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isAnimating, setIsAnimating] = useState(false);

  // XOR data points
  const points = [
    { x: 0, y: 0, label: 'a', class: 0, color: '#ef4444' },
    { x: 0, y: 1, label: 'b', class: 1, color: '#3b82f6' },
    { x: 1, y: 0, label: 'c', class: 1, color: '#3b82f6' },
    { x: 1, y: 1, label: 'd', class: 0, color: '#ef4444' }
  ];

  // Border pairs for each step
  const borderPairs = [
    { from: 'a', to: 'b', reason: 'Different classes (0→1), distance=1' },
    { from: 'a', to: 'c', reason: 'Different classes (0→1), distance=1' },
    { from: 'b', to: 'd', reason: 'Different classes (1→0), distance=1' },
    { from: 'c', to: 'd', reason: 'Different classes (1→0), distance=1' }
  ];

  const steps = [
    {
      title: "XOR Problem Setup",
      description: "Four data points representing XOR logic operation",
      content: "Points a,d (red) = class 0, Points b,c (blue) = class 1"
    },
    {
      title: "Step 1: Identify Border Pairs",
      description: "Find adjacent points of different classes",
      content: "Border pairs: (a,b), (a,c), (b,d), (c,d)"
    },
    {
      title: "Step 2: First Hyperplane",
      description: "Separate border pairs (a,b) and (a,c)",
      content: "Hyperplane: x + y = 0.5 (separates point 'a' from b,c,d)"
    },
    {
      title: "Step 3: Check Homogeneity",
      description: "Area left of line contains only class 0 (homogeneous)",
      content: "Right area still contains mixed classes → needs second hyperplane"
    },
    {
      title: "Step 4: Second Hyperplane", 
      description: "Separate remaining border pairs (b,d) and (c,d)",
      content: "Hyperplane: x + y = 1.5 (separates point 'd' from b,c)"
    },
    {
      title: "Step 5: Layer 1 Complete",
      description: "Two neurons create three homogeneous regions",
      content: "Region A: only 'a' (class 0), Region B: 'b','c' (class 1), Region C: only 'd' (class 0)"
    },
    {
      title: "Step 6: Layer 2 Construction",
      description: "Create truth table from homogeneous regions",
      content: "3 regions → 3-row truth table for final classification"
    },
    {
      title: "Step 7: Final Network",
      description: "Complete 2-layer network without any hyperparameter tuning!",
      content: "BPM automatically determines optimal architecture"
    }
  ];

  const SVG_SIZE = 300;
  const MARGIN = 50;
  const SCALE = 200;

  const scaleX = (x) => MARGIN + x * SCALE;
  const scaleY = (y) => SVG_SIZE - MARGIN - y * SCALE;

  const nextStep = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const prevStep = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  const reset = () => {
    setCurrentStep(0);
  };

  const renderBorderPairs = () => {
    if (currentStep < 1) return null;
    
    return borderPairs.map((pair, index) => {
      const pointFrom = points.find(p => p.label === pair.from);
      const pointTo = points.find(p => p.label === pair.to);
      
      return (
        <line
          key={index}
          x1={scaleX(pointFrom.x)}
          y1={scaleY(pointFrom.y)}
          x2={scaleX(pointTo.x)}
          y2={scaleY(pointTo.y)}
          stroke="#fbbf24"
          strokeWidth="2"
          strokeDasharray="5,5"
          opacity={currentStep >= 1 ? 0.7 : 0}
        />
      );
    });
  };

  const renderHyperplanes = () => {
    const hyperplanes = [];
    
    // First hyperplane: x + y = 0.5
    if (currentStep >= 2) {
      hyperplanes.push(
        <line
          key="h1"
          x1={scaleX(0.5)}
          y1={scaleY(0)}
          x2={scaleX(0)}
          y2={scaleY(0.5)}
          stroke="#10b981"
          strokeWidth="3"
          opacity={0.8}
        />
      );
    }
    
    // Second hyperplane: x + y = 1.5  
    if (currentStep >= 4) {
      hyperplanes.push(
        <line
          key="h2"
          x1={scaleX(1)}
          y1={scaleY(0.5)}
          x2={scaleX(0.5)}
          y2={scaleY(1)}
          stroke="#8b5cf6"
          strokeWidth="3"
          opacity={0.8}
        />
      );
    }
    
    return hyperplanes;
  };

  const renderRegionLabels = () => {
    if (currentStep < 5) return null;
    
    return (
      <>
        <text x={scaleX(0.15)} y={scaleY(0.15)} fill="#374151" fontSize="14" fontWeight="bold">
          Region A
        </text>
        <text x={scaleX(0.5)} y={scaleY(0.7)} fill="#374151" fontSize="14" fontWeight="bold">
          Region B  
        </text>
        <text x={scaleX(0.85)} y={scaleY(0.85)} fill="#374151" fontSize="14" fontWeight="bold">
          Region C
        </text>
      </>
    );
  };

  const renderNeuralNetwork = () => {
    const networkSVG_WIDTH = 400;
    const networkSVG_HEIGHT = 300;
    const layerSpacing = 120;
    const neuronRadius = 25;
    
    // Input layer position
    const inputLayer = {
      x: 50,
      neurons: [
        { y: 100, label: 'X', active: true },
        { y: 200, label: 'Y', active: true }
      ]
    };
    
    // Hidden layer position  
    const hiddenLayer = {
      x: 170,
      neurons: [
        { y: 80, label: 'N11', active: currentStep >= 2, equation: 'x+y≥0.5' },
        { y: 220, label: 'N12', active: currentStep >= 4, equation: 'x+y≥1.5' }
      ]
    };
    
    // Output layer position
    const outputLayer = {
      x: 290,
      neurons: [
        { y: 150, label: 'N21', active: currentStep >= 7, equation: 'OR logic' }
      ]
    };

    return (
      <svg width={networkSVG_WIDTH} height={networkSVG_HEIGHT} className="border border-gray-300 bg-white rounded">
        {/* Connections - Input to Hidden */}
        {currentStep >= 2 && hiddenLayer.neurons.map((hiddenNeuron, hIndex) => (
          hiddenNeuron.active && inputLayer.neurons.map((inputNeuron, iIndex) => (
            <line
              key={`ih-${iIndex}-${hIndex}`}
              x1={inputLayer.x + neuronRadius}
              y1={inputNeuron.y}
              x2={hiddenLayer.x - neuronRadius}
              y2={hiddenNeuron.y}
              stroke="#94a3b8"
              strokeWidth="2"
              opacity={0.6}
            />
          ))
        ))}
        
        {/* Connections - Hidden to Output */}
        {currentStep >= 7 && outputLayer.neurons.map((outputNeuron, oIndex) => (
          hiddenLayer.neurons.map((hiddenNeuron, hIndex) => (
            hiddenNeuron.active && (
              <line
                key={`ho-${hIndex}-${oIndex}`}
                x1={hiddenLayer.x + neuronRadius}
                y1={hiddenNeuron.y}
                x2={outputLayer.x - neuronRadius}
                y2={outputNeuron.y}
                stroke="#94a3b8"
                strokeWidth="2"
                opacity={0.6}
              />
            )
          ))
        ))}
        
        {/* Input Layer */}
        <text x={inputLayer.x} y={30} textAnchor="middle" fill="#374151" fontSize="14" fontWeight="bold">
          Input Layer
        </text>
        {inputLayer.neurons.map((neuron, index) => (
          <g key={`input-${index}`}>
            <circle
              cx={inputLayer.x}
              cy={neuron.y}
              r={neuronRadius}
              fill={neuron.active ? "#e2e8f0" : "#f1f5f9"}
              stroke={neuron.active ? "#475569" : "#cbd5e1"}
              strokeWidth="2"
            />
            <text
              x={inputLayer.x}
              y={neuron.y + 5}
              textAnchor="middle"
              fill="#374151"
              fontSize="14"
              fontWeight="bold"
            >
              {neuron.label}
            </text>
          </g>
        ))}
        
        {/* Hidden Layer */}
        <text x={hiddenLayer.x} y={30} textAnchor="middle" fill="#374151" fontSize="14" fontWeight="bold">
          Hidden Layer
        </text>
        {hiddenLayer.neurons.map((neuron, index) => (
          <g key={`hidden-${index}`}>
            <circle
              cx={hiddenLayer.x}
              cy={neuron.y}
              r={neuronRadius}
              fill={neuron.active ? "#ddd6fe" : "#f1f5f9"}
              stroke={neuron.active ? "#7c3aed" : "#cbd5e1"}
              strokeWidth="2"
              opacity={neuron.active ? 1 : 0.3}
            />
            <text
              x={hiddenLayer.x}
              y={neuron.y + 5}
              textAnchor="middle"
              fill={neuron.active ? "#374151" : "#94a3b8"}
              fontSize="12"
              fontWeight="bold"
            >
              {neuron.label}
            </text>
            {neuron.active && (
              <text
                x={hiddenLayer.x}
                y={neuron.y + 45}
                textAnchor="middle"
                fill="#7c3aed"
                fontSize="10"
              >
                {neuron.equation}
              </text>
            )}
          </g>
        ))}
        
        {/* Output Layer */}
        <text x={outputLayer.x} y={30} textAnchor="middle" fill="#374151" fontSize="14" fontWeight="bold">
          Output Layer
        </text>
        {outputLayer.neurons.map((neuron, index) => (
          <g key={`output-${index}`}>
            <circle
              cx={outputLayer.x}
              cy={neuron.y}
              r={neuronRadius}
              fill={neuron.active ? "#dcfce7" : "#f1f5f9"}
              stroke={neuron.active ? "#16a34a" : "#cbd5e1"}
              strokeWidth="2"
              opacity={neuron.active ? 1 : 0.3}
            />
            <text
              x={outputLayer.x}
              y={neuron.y + 5}
              textAnchor="middle"
              fill={neuron.active ? "#374151" : "#94a3b8"}
              fontSize="12"
              fontWeight="bold"
            >
              {neuron.label}
            </text>
            {neuron.active && (
              <text
                x={outputLayer.x}
                y={neuron.y + 45}
                textAnchor="middle"
                fill="#16a34a"
                fontSize="10"
              >
                {neuron.equation}
              </text>
            )}
          </g>
        ))}
      </svg>
    );
  };

  return (
    <div className="max-w-7xl mx-auto p-6 bg-white">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Border Pairs Method (BPM) - XOR Demonstration
        </h1>
        <p className="text-lg text-gray-600">
          By Dr. Bojan Ploj, Slovenia - Automatic Neural Network Construction
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* Problem Space Visualization */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">Problem Space</h2>
          
          <svg width={SVG_SIZE} height={SVG_SIZE} className="border border-gray-300 bg-white rounded">
            {/* Grid */}
            <defs>
              <pattern id="grid" width="50" height="50" patternUnits="userSpaceOnUse">
                <path d="M 50 0 L 0 0 0 50" fill="none" stroke="#e5e7eb" strokeWidth="1"/>
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#grid)" />
            
            {/* Axes */}
            <line x1={MARGIN} y1={SVG_SIZE - MARGIN} x2={SVG_SIZE - MARGIN} y2={SVG_SIZE - MARGIN} 
                  stroke="#374151" strokeWidth="2" markerEnd="url(#arrowhead)"/>
            <line x1={MARGIN} y1={SVG_SIZE - MARGIN} x2={MARGIN} y2={MARGIN} 
                  stroke="#374151" strokeWidth="2" markerEnd="url(#arrowhead)"/>
            
            {/* Arrow markers */}
            <defs>
              <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                      refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#374151" />
              </marker>
            </defs>
            
            {/* Axis labels */}
            <text x={SVG_SIZE - 20} y={SVG_SIZE - 30} fill="#374151" fontSize="14">X</text>
            <text x={30} y={20} fill="#374151" fontSize="14">Y</text>
            
            {/* Border pairs */}
            {renderBorderPairs()}
            
            {/* Hyperplanes */}
            {renderHyperplanes()}
            
            {/* Region labels */}
            {renderRegionLabels()}
            
            {/* Data points */}
            {points.map((point, index) => (
              <g key={index}>
                <circle
                  cx={scaleX(point.x)}
                  cy={scaleY(point.y)}
                  r="8"
                  fill={point.color}
                  stroke="#fff"
                  strokeWidth="2"
                />
                <text
                  x={scaleX(point.x)}
                  y={scaleY(point.y) - 15}
                  textAnchor="middle"
                  fill="#374151"
                  fontSize="14"
                  fontWeight="bold"
                >
                  {point.label}({point.x},{point.y})
                </text>
                <text
                  x={scaleX(point.x)}
                  y={scaleY(point.y) + 25}
                  textAnchor="middle"
                  fill={point.color}
                  fontSize="12"
                  fontWeight="bold"
                >
                  Class {point.class}
                </text>
              </g>
            ))}
          </svg>
        </div>

        {/* Neural Network Construction */}
        <div className="bg-gray-50 p-6 rounded-lg">
          <h2 className="text-xl font-semibold mb-4">Neural Network Construction</h2>
          {renderNeuralNetwork()}
          <div className="mt-4 text-sm text-gray-600">
            <p><strong>Architecture:</strong> 2 → {currentStep >= 4 ? '2' : currentStep >= 2 ? '1' : '0'} → {currentStep >= 7 ? '1' : '0'}</p>
            <p><strong>Total Neurons:</strong> {2 + (currentStep >= 4 ? 2 : currentStep >= 2 ? 1 : 0) + (currentStep >= 7 ? 1 : 0)}</p>
          </div>
        </div>

        {/* Step Information */}
        <div className="space-y-6">
          <div className="bg-blue-50 p-6 rounded-lg">
            <h2 className="text-xl font-semibold mb-4">
              Step {currentStep + 1} of {steps.length}
            </h2>
            <h3 className="text-lg font-medium text-blue-800 mb-2">
              {steps[currentStep].title}
            </h3>
            <p className="text-gray-700 mb-3">
              {steps[currentStep].description}
            </p>
            <p className="text-sm text-blue-600 bg-blue-100 p-3 rounded">
              {steps[currentStep].content}
            </p>
          </div>

          {/* Controls */}
          <div className="flex gap-4 justify-center">
            <button
              onClick={reset}
              className="flex items-center gap-2 px-4 py-2 bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition-colors"
            >
              <RotateCcw size={16} />
              Reset
            </button>
            <button
              onClick={prevStep}
              disabled={currentStep === 0}
              className="px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors"
            >
              Previous
            </button>
            <button
              onClick={nextStep}
              disabled={currentStep === steps.length - 1}
              className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded disabled:bg-gray-300 disabled:cursor-not-allowed hover:bg-blue-600 transition-colors"
            >
              Next
              <ArrowRight size={16} />
            </button>
          </div>

          {/* Border Pairs Info */}
          {currentStep >= 1 && (
            <div className="bg-yellow-50 p-4 rounded-lg">
              <h3 className="font-medium text-yellow-800 mb-2">Border Pairs Identified:</h3>
              <ul className="text-sm text-yellow-700 space-y-1">
                {borderPairs.map((pair, index) => (
                  <li key={index}>
                    • ({pair.from},{pair.to}): {pair.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Final Truth Table */}
          {currentStep >= 6 && (
            <div className="bg-green-50 p-4 rounded-lg">
              <h3 className="font-medium text-green-800 mb-2">Layer 2 Truth Table:</h3>
              <table className="w-full text-sm border border-green-200">
                <thead>
                  <tr className="bg-green-100">
                    <th className="border border-green-200 p-2">Region</th>
                    <th className="border border-green-200 p-2">N11</th>
                    <th className="border border-green-200 p-2">N12</th>
                    <th className="border border-green-200 p-2">Output</th>
                  </tr>
                </thead>
                <tbody className="text-green-700">
                  <tr>
                    <td className="border border-green-200 p-2 text-center">A</td>
                    <td className="border border-green-200 p-2 text-center">0</td>
                    <td className="border border-green-200 p-2 text-center">0</td>
                    <td className="border border-green-200 p-2 text-center">0</td>
                  </tr>
                  <tr>
                    <td className="border border-green-200 p-2 text-center">B</td>
                    <td className="border border-green-200 p-2 text-center">1</td>
                    <td className="border border-green-200 p-2 text-center">0</td>
                    <td className="border border-green-200 p-2 text-center">1</td>
                  </tr>
                  <tr>
                    <td className="border border-green-200 p-2 text-center">C</td>
                    <td className="border border-green-200 p-2 text-center">1</td>
                    <td className="border border-green-200 p-2 text-center">1</td>
                    <td className="border border-green-200 p-2 text-center">0</td>
                  </tr>
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* Summary */}
      {currentStep >= 7 && (
        <div className="mt-8 bg-gradient-to-r from-purple-50 to-blue-50 p-6 rounded-lg">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">
            🎉 BPM Success - XOR Problem Solved!
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="font-semibold text-purple-800 mb-2">BPM Advantages:</h3>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>✅ No hyperparameter tuning</li>
                <li>✅ Deterministic construction</li>
                <li>✅ Minimal architecture</li>
                <li>✅ Geometric intuition</li>
                <li>✅ No random initialization</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-blue-800 mb-2">Backpropagation Issues:</h3>
              <ul className="text-sm text-gray-700 space-y-1">
                <li>❌ Random weight initialization</li>
                <li>❌ Local minima problems</li>
                <li>❌ Hyperparameter guessing</li>
                <li>❌ Thousands of iterations</li>
                <li>❌ Architecture uncertainty</li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default BPMXORDemo;
