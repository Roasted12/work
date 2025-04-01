// Charts.js - Visualization scripts for the Heart Disease Prediction System

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all chart elements on the page
    initMetricCharts();
    initComparisonChart();
});

// Initialize metric gauge charts
function initMetricCharts() {
    const metricCharts = document.querySelectorAll('[id$="-chart"]');
    
    if (!metricCharts.length || typeof Chart === 'undefined') return;
    
    metricCharts.forEach((canvas, index) => {
        // Get value from data attribute
        const value = parseFloat(canvas.dataset.value);
        if (isNaN(value)) return;
        
        // Set color and label based on chart ID
        let color, label;
        const id = canvas.id;
        
        if (id.includes('accuracy')) {
            color = '#0d6efd'; // Blue
            label = 'Accuracy';
        } else if (id.includes('precision')) {
            color = '#198754'; // Green
            label = 'Precision';
        } else if (id.includes('recall')) {
            color = '#fd7e14'; // Orange
            label = 'Recall';
        } else if (id.includes('f1')) {
            color = '#6f42c1'; // Purple
            label = 'F1 Score';
        } else {
            color = '#6c757d'; // Secondary
            label = 'Metric';
        }
        
        // Create gauge chart with animation
        const chart = new Chart(canvas, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, 1], // Start with 0, will animate to actual value
                    backgroundColor: [color, 'rgba(200, 200, 200, 0.2)'],
                    borderWidth: 0
                }]
            },
            options: {
                cutout: '75%',
                responsive: true,
                maintainAspectRatio: true,
                circumference: 180,
                rotation: -90,
                plugins: {
                    tooltip: {
                        enabled: false
                    },
                    legend: {
                        display: false
                    }
                },
                animation: {
                    duration: 1500,
                    easing: 'easeOutCubic'
                }
            }
        });
        
        // Animate the chart after a delay based on index
        setTimeout(() => {
            chart.data.datasets[0].data = [value, 1 - value];
            chart.update();
        }, 300 * index);
        
        // Add center text if possible
        addCenterText(canvas, `${(value * 100).toFixed(0)}%`, label);
    });
}

// Add text to the center of a doughnut chart
function addCenterText(canvas, value, label) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // This function will be called after the chart is rendered
    const addText = () => {
        // Get canvas dimensions
        const width = canvas.width;
        const height = canvas.height;
        
        // Save the current state
        ctx.save();
        
        // Clear the center area (not needed for a new chart but useful for updates)
        ctx.clearRect(width * 0.2, height * 0.2, width * 0.6, height * 0.6);
        
        // Add value text
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.font = 'bold 16px sans-serif';
        ctx.fillStyle = '#fff';
        ctx.fillText(value, width / 2, height / 2);
        
        // Restore the state
        ctx.restore();
    };
    
    // Add the text now and also set it to run after any animations
    addText();
    canvas.parentElement.addEventListener('chartjs-animation-complete', addText);
}

// Initialize model comparison chart if present
function initComparisonChart() {
    const comparisonChart = document.getElementById('comparison-chart');
    
    if (!comparisonChart || typeof Chart === 'undefined') return;
    
    // Get chart data from data attributes
    const labels = JSON.parse(comparisonChart.dataset.labels || '[]');
    const accuracyValues = JSON.parse(comparisonChart.dataset.accuracy || '[]');
    const precisionValues = JSON.parse(comparisonChart.dataset.precision || '[]');
    const recallValues = JSON.parse(comparisonChart.dataset.recall || '[]');
    const f1Values = JSON.parse(comparisonChart.dataset.f1 || '[]');
    
    if (!labels.length) return;
    
    // Create the chart
    new Chart(comparisonChart, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Accuracy',
                    data: accuracyValues,
                    backgroundColor: 'rgba(13, 110, 253, 0.7)',
                    borderColor: 'rgba(13, 110, 253, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Precision',
                    data: precisionValues,
                    backgroundColor: 'rgba(25, 135, 84, 0.7)',
                    borderColor: 'rgba(25, 135, 84, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Recall',
                    data: recallValues,
                    backgroundColor: 'rgba(253, 126, 20, 0.7)',
                    borderColor: 'rgba(253, 126, 20, 1)',
                    borderWidth: 1
                },
                {
                    label: 'F1 Score',
                    data: f1Values,
                    backgroundColor: 'rgba(111, 66, 193, 0.7)',
                    borderColor: 'rgba(111, 66, 193, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1,
                    ticks: {
                        callback: function(value) {
                            return (value * 100) + '%';
                        }
                    }
                }
            },
            plugins: {
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return context.dataset.label + ': ' + (context.raw * 100).toFixed(2) + '%';
                        }
                    }
                },
                legend: {
                    position: 'top',
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeOutQuart'
            }
        }
    });
}

// Create animated metric counter element
function createMetricCounter(elementId, targetValue, prefix = '', suffix = '', duration = 1500) {
    const element = document.getElementById(elementId);
    if (!element) return;
    
    // Format for display
    const formattedTarget = typeof targetValue === 'number' 
        ? targetValue.toFixed(2) 
        : targetValue;
    
    // Set starting value
    let startValue = 0;
    let currentValue = startValue;
    
    // Calculate step size based on duration and refresh rate
    const fps = 60;
    const refreshInterval = 1000 / fps;
    const steps = Math.ceil(duration / refreshInterval);
    const increment = targetValue / steps;
    
    // Start the animation
    const counter = setInterval(() => {
        currentValue += increment;
        
        // Check if reached or exceeded target
        if (currentValue >= targetValue) {
            currentValue = targetValue;
            clearInterval(counter);
        }
        
        // Update the element text
        element.textContent = prefix + currentValue.toFixed(2) + suffix;
    }, refreshInterval);
}