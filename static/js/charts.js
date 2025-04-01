// Charts.js integration for the Heart Disease Prediction System
document.addEventListener('DOMContentLoaded', function() {
    // Function to create a doughnut chart for metrics
    function createMetricChart(elementId, value, label) {
        const element = document.getElementById(elementId);
        
        if (!element) return;
        
        const ctx = element.getContext('2d');
        
        return new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: [label, 'Remainder'],
                datasets: [{
                    data: [value, 1 - value],
                    backgroundColor: [
                        '#007bff',  // Primary color for the metric
                        '#e9ecef'   // Light gray for remainder
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `${context.label}: ${(context.raw * 100).toFixed(2)}%`;
                            }
                        }
                    }
                },
                animation: {
                    animateRotate: true,
                    animateScale: true
                }
            }
        });
    }
    
    // Create charts for metrics if they exist on the page
    const metricIds = ['accuracy-chart', 'precision-chart', 'recall-chart', 'f1-chart'];
    const metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1 Score'];
    
    metricIds.forEach((id, index) => {
        const element = document.getElementById(id);
        if (element) {
            const value = parseFloat(element.dataset.value);
            if (!isNaN(value)) {
                createMetricChart(id, value, metricLabels[index]);
            }
        }
    });
    
    // Function to create comparison charts on the history page
    function createComparisonChart() {
        const element = document.getElementById('model-comparison-chart');
        
        if (!element) return;
        
        // Get data from the table
        const table = document.getElementById('history-table');
        if (!table) return;
        
        const rows = table.getElementsByTagName('tr');
        const models = {};
        
        // Skip header row
        for (let i = 1; i < rows.length; i++) {
            const cells = rows[i].getElementsByTagName('td');
            if (cells.length >= 5) {
                const modelType = cells[2].textContent.trim();
                const accuracy = parseFloat(cells[3].textContent);
                
                if (!models[modelType]) {
                    models[modelType] = {
                        accuracies: [],
                        count: 0
                    };
                }
                
                models[modelType].accuracies.push(accuracy);
                models[modelType].count++;
            }
        }
        
        // Calculate average accuracy for each model type
        const labels = [];
        const data = [];
        const backgroundColors = [
            'rgba(0, 123, 255, 0.7)',    // Blue for ML
            'rgba(40, 167, 69, 0.7)',    // Green for DL
            'rgba(23, 162, 184, 0.7)',   // Cyan for QML
            'rgba(255, 193, 7, 0.7)'     // Yellow for QNN
        ];
        
        for (const [type, info] of Object.entries(models)) {
            labels.push(type);
            const avgAccuracy = info.accuracies.reduce((a, b) => a + b, 0) / info.count;
            data.push(avgAccuracy);
        }
        
        const ctx = element.getContext('2d');
        
        return new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Average Accuracy',
                    data: data,
                    backgroundColor: backgroundColors.slice(0, labels.length),
                    borderColor: backgroundColors.slice(0, labels.length).map(color => color.replace('0.7', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Accuracy: ${(context.raw * 100).toFixed(2)}%`;
                            }
                        }
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    // Create comparison chart if we're on the history page
    if (document.getElementById('model-comparison-chart')) {
        createComparisonChart();
    }
});
