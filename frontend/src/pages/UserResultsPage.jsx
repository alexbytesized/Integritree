import React, { useEffect } from 'react' // 1. Added useEffect
import { Link } from 'react-router-dom'
import './UserResultsPage.css'
import backButton from '../assets/back-button.png'
import FraudDonut from "../components/FraudDonut";
import { useState } from "react";
import SearchBar from "../components/SearchBar";
import TransactionTable from "../components/TransactionTable";
import AOS from "aos";
import "aos/dist/aos.css";


const UserResultsPage = () => {
    const [searchTerm, setSearchTerm] = useState("");

    useEffect(() => {
    if (window.particlesJS) {
        window.particlesJS.load('particles-js', '/particles.json', function() {
        console.log('callback - particles.js config loaded');
        });
    }
    }, []);

    useEffect(() => {
    AOS.init({
        duration: 1000,
        once: false,
        offset: 100,
    });
    }, []);

    return(
        <div className='user-results-container'>
            <div id="particles-js"></div>

            <div data-aos="fade-up">
                 <Link to="/" className="backButtonContainer">
                <img src={backButton} alt="Back Button" className="backButton" />
            </Link>

            <div className='results'>
                <div className='results-title'>
                    Results Overview
                </div>

                <div className='results-upper'>
                    <div className='csv'>
                        <p><strong>CSV File Name:</strong> dataset.csv</p>
                    </div>
                    <div className='csv'>
                        <p><strong>Total Records:</strong> 6,368,659</p>
                    </div>
                </div>

                <div className='results-proper'>
                    <div className='model'>
                        <div className='model-1'>
                            <p className='model-title'>Benchmark RF</p>
                            <p className='legitimate-numbers' >6012120</p>
                            <p className='records'>Legitimate Records</p>

                            <p className='fraudulent-numbers'>356549</p>
                            <p className='records'>Fraudulent Records</p>

                            <FraudDonut value={5.6} />
                        </div>
                        <div className='model-1'>
                            <p className='model-title'>RF-SMOTE</p>
                            <p className='legitimate-numbers'>6308787</p>
                            <p className='records'>Legitimate Records</p>

                            <p className='fraudulent-numbers'>59872</p>
                            <p className='records'>Fraudulent Records</p>

                            <FraudDonut value={0.94} />
                        </div>
                    </div>

                    <div className='model-right'>
                        <p className='model-title'>Ground Truth</p>
                        <p className='legitimate-numbers'>6354407</p>
                        <p className='records'>Legitimate Records</p>

                        <p className='fraudulent-numbers'>8213</p>
                        <p className='records'>Fraudulent Records</p>

                        <FraudDonut value={0.13} />
                    </div>
                </div>
            </div>

                <div className="transaction">
                    <div className="results-title">Transaction Records</div>

                    <SearchBar
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                    placeholder="Search Transactions"
                    />

                    <TransactionTable searchTerm={searchTerm} />
                </div>
            </div>
        </div>
    )
}

export default UserResultsPage;