"use client";

import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [formData, setFormData] = useState({
    "Active component type formation energy": 0,
    "Active component type density": 0,
    "Active component content (wt percent)": 0,
    "Promoter type formation energy": 0,
    "Promoter type density": 0,
    "Promoter content (wt percent)": 0,
    "Support a type formation energy": 0,
    "Support a type density": 0,
    "Support a content (wt percent)": 0,
    "Support b type formation energy": 0,
    "Support b type density": 0,
    "Calcination Temperature (C)": 0,
    "Calcination time (h)": 0,
    "Reduction Temperature (C)": 0,
    "Reduction Pressure (bar)": 0,
    "Reduction time (h)": 0,
    "Reduced hydrogen content (vol percent)": 0,
    "Temperature (C)": 0,
    "Pressure (bar)": 0,
    "Weight hourly space velocity [mgcat/(min·ml)]": 0,
    "Content of inert components in raw materials (vol percent)": 0,
    "h2/co2 ratio (mol/mol)": 0
  });

  const [predictions, setPredictions] = useState({
    co2_conversion_ratio: "",
    ch4_yield: "",
    ch4_selectivity: ""
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSliderChange = (e) => {
    const { name, value } = e.target;
    setFormData({ ...formData, [name]: value });
  };

  const predict = async (endpoint) => {
    setLoading(true);
    setError(null); // Reset any previous errors

    try {
      const response = await axios.post(
        `http://127.0.0.1:5000/predict/${endpoint}`,
        formData
      );
      setPredictions({ ...predictions, ...response.data });
    } catch (error) {
      console.error("Prediction Error:", error);
      setError("An error occurred while fetching the prediction. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container mx-auto p-8 bg-gray-50 rounded-xl shadow-lg">
      <h1 className="text-3xl font-semibold text-center mb-6 text-blue-600">
        CO₂ Methanation Prediction
      </h1>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {Object.keys(formData).map((key, index) => (
          <div key={index} className="space-y-4">
            <label className="block text-sm font-medium text-gray-700">{key}</label>

            <div className="flex items-center space-x-4">
              <input
                type="range"
                name={key}
                value={formData[key]}
                min="0"
                max="100" // Adjust the max and min based on your expected value ranges
                step="0.1"
                onChange={handleSliderChange}
                className="w-full"
              />
              <input
                type="number"
                name={key}
                value={formData[key]}
                min="0"
                max="100"
                step="0.1"
                onChange={handleChange}
                className="w-24 p-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
        ))}
      </div>

      <div className="flex justify-center space-x-4 mt-6">
        <button
          onClick={() => predict("co2_conversion")}
          disabled={loading}
          className={`bg-blue-600 text-white px-6 py-3 rounded-lg shadow-md hover:bg-blue-700 transition duration-300 ease-in-out ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          {loading ? "Loading..." : "Predict CO₂ Conversion"}
        </button>
        <button
          onClick={() => predict("ch4_yield")}
          disabled={loading}
          className={`bg-green-600 text-white px-6 py-3 rounded-lg shadow-md hover:bg-green-700 transition duration-300 ease-in-out ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          {loading ? "Loading..." : "Predict CH₄ Yield"}
        </button>
        <button
          onClick={() => predict("ch4_selectivity")}
          disabled={loading}
          className={`bg-purple-600 text-white px-6 py-3 rounded-lg shadow-md hover:bg-purple-700 transition duration-300 ease-in-out ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
        >
          {loading ? "Loading..." : "Predict CH₄ Selectivity"}
        </button>
      </div>

      {error && (
        <div className="mt-6 p-4 bg-red-100 text-red-700 rounded-lg">
          <p>{error}</p>
        </div>
      )}

      <div className="mt-8">
        <h2 className="text-lg font-semibold text-gray-700">Predictions:</h2>
        <div className="mt-4 space-y-2">
          <p className="text-sm font-medium text-gray-800">
            <span className="font-semibold">CO₂ Conversion Ratio:</span> {predictions.co2_conversion_ratio}
          </p>
          <p className="text-sm font-medium text-gray-800">
            <span className="font-semibold">CH₄ Yield:</span> {predictions.ch4_yield}
          </p>
          <p className="text-sm font-medium text-gray-800">
            <span className="font-semibold">CH₄ Selectivity:</span> {predictions.ch4_selectivity}
          </p>
        </div>
      </div>
    </div>
  );
}
