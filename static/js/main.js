// Main JavaScript functionality for the Heart Disease Prediction System
document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const fileInput = document.getElementById('file-input');
    const fileLabel = document.querySelector('.custom-file-label');
    const uploadForm = document.getElementById('upload-form');
    const loadingContainer = document.getElementById('loading-container');
    const modelCards = document.querySelectorAll('.model-card');
    const modelInput = document.getElementById('model-type');
    
    // File input change handler
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const fileName = this.files[0] ? this.files[0].name : 'Choose file';
            if (fileLabel) {
                fileLabel.textContent = fileName;
            }
        });
    }
    
    // Model card selection
    if (modelCards.length > 0) {
        modelCards.forEach(card => {
            card.addEventListener('click', function() {
                // Remove selected class from all cards
                modelCards.forEach(c => c.classList.remove('selected'));
                
                // Add selected class to clicked card
                this.classList.add('selected');
                
                // Update hidden input value
                if (modelInput) {
                    modelInput.value = this.dataset.model;
                }
            });
        });
    }
    
    // Form submission
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            // Check if file is selected
            if (fileInput && fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select a CSV file to upload.');
                return;
            }
            
            // Check if model is selected
            if (modelInput && !modelInput.value) {
                e.preventDefault();
                alert('Please select a model for prediction.');
                return;
            }
            
            // Show loading animation
            if (loadingContainer) {
                loadingContainer.style.display = 'block';
            }
        });
    }
    
    // Export results button
    const exportButton = document.getElementById('export-button');
    if (exportButton) {
        exportButton.addEventListener('click', function() {
            const sessionId = this.dataset.sessionId;
            if (sessionId) {
                window.location.href = `/export/${sessionId}`;
            }
        });
    }
    
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggerList.length > 0) {
        [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl));
    }
    
    // Initialize popovers
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    if (popoverTriggerList.length > 0) {
        [...popoverTriggerList].map(popoverTriggerEl => new bootstrap.Popover(popoverTriggerEl));
    }
});

// Function to filter the history table
function filterHistory() {
    const input = document.getElementById('history-search');
    const filter = input.value.toUpperCase();
    const table = document.getElementById('history-table');
    const tr = table.getElementsByTagName('tr');
    
    for (let i = 1; i < tr.length; i++) {  // Start from 1 to skip header row
        let visible = false;
        const td = tr[i].getElementsByTagName('td');
        
        for (let j = 0; j < td.length; j++) {
            const cell = td[j];
            if (cell) {
                const txtValue = cell.textContent || cell.innerText;
                if (txtValue.toUpperCase().indexOf(filter) > -1) {
                    visible = true;
                    break;
                }
            }
        }
        
        tr[i].style.display = visible ? '' : 'none';
    }
}
