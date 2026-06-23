import UserResultsPage from "./UserResultsPage";
import './ResearcherResultsPage.css'

const ResearcherResultsPage = () => {
    return(
        <div className="main">
            <UserResultsPage/>

            <div className="statistics">
                <div className='statistics-title'>
                    Performance Statistics of Both Models
                </div>

                <div className="statistics-upper">
                    <div className="stat-model">
                        <p className="stat-title">Benchmark RF</p>
                        <div className="column">
                            <div className="row">
                                <strong>Precision:</strong>
                                <p>87.8%</p>
                            </div>
                        </div>

                        <div className="column">
                            <div className="row">
                                <strong>Recall:</strong>
                                <p>75.2%</p>
                            </div>
                        </div>

                        <div className="column">
                            <div className="row">
                                <strong>F1-Score:</strong>
                                <p>81.0%</p>
                            </div>
                        </div>

                        <div className="column">
                            <div className="row">
                                <strong>MCC:</strong>
                                <p>80.5%</p>
                            </div>
                        </div>

                        <div className="column">
                            <div className="row">
                                <strong>PR-AUC:</strong>
                                <p>82.4%</p>
                            </div>
                        </div>
                    </div>

                    <div className="stat-model">
                        <p className="stat-title">RF-SMOTE</p>
                        <div className="column">
                            <div className="row">
                                <strong>Precision:</strong>
                                <p>98.5%</p>
                            </div>
                        </div>

                        <div className="column">
                            <div className="row">
                                <strong>Recall:</strong>
                                <p>99.0%</p>
                            </div>
                        </div>

                        <div className="column">
                            <div className="row">
                                <strong>F1-Score:</strong>
                                <p>98.7%</p>
                            </div>
                        </div>

                        <div className="column">
                            <div className="row">
                                <strong>MCC:</strong>
                                <p>98.2%</p>
                            </div>
                        </div>

                        <div className="column">
                            <div className="row">
                                <strong>PR-AUC:</strong>
                                <p>99.1%</p>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="middle">
                    <p className="test-title">Percentage Difference</p>

                    <div className="percentage-grid">
                        <div className="percentage-card">
                            <p><strong>11.49%</strong> difference</p>
                            <h3>Precision</h3>
                        </div>

                        <div className="percentage-card">
                            <p><strong>27.32%</strong> difference</p>
                            <h3>Recall</h3>
                        </div>

                        <div className="percentage-card">
                            <p><strong>19.70%</strong> difference</p>
                            <h3>F1-Score</h3>
                        </div>

                        <div className="percentage-card">
                            <p><strong>19.81%</strong> difference</p>
                            <h3>MCC</h3>
                        </div>

                        <div className="percentage-card">
                            <p><strong>18.40%</strong> difference</p>
                            <h3>PR-AUC</h3>
                        </div>
                    </div>
                </div>

                <div className="middle">
                    <p className="test-title">McNemar's Test</p>

                    <div className="mcnemar-content">
                        <div className="mcnemar-table">
                            <div className="empty-cell"></div>
                            <div className="table-header">RF-SMOTE Model Correct</div>
                            <div className="table-header">RF-SMOTE Model Incorrect</div>

                            <div className="row-header">RF Model Correct</div>
                            <div className="table-value">6005000</div>
                            <div className="table-value">125000</div>

                            <div className="row-header">RF Model Incorrect</div>
                            <div className="table-value">345000</div>
                            <div className="table-value">345000</div>
                        </div>

                        <div className="mcnemar-result">
                            <div className="pvalue-box">
                                <p>Resulting p-value:</p>
                                <strong>&lt; 0.0001</strong>
                            </div>

                            <div className="evaluation">
                                <p>Evaluation:</p>
                                <p>
                                The difference between the benchmark RF model and the RF-SMOTE model
                                is statistically significant.
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default ResearcherResultsPage;