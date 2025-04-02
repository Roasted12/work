// Main JavaScript functionality for the Heart Disease Prediction System
document.addEventListener('DOMContentLoaded', function() {
    // Initialize Wizard
    initWizard();
    
    // Initialize other components
    initFileUpload();
    initModelSelection();
    initFormValidation();
    initResultsView();
    initHistoryFilter();
    
    // Initialize Bootstrap components
    initBootstrapComponents();
});

// Initialize the step wizard
function initWizard() {
    const wizardSteps = document.querySelectorAll('.wizard-step');
    const wizardPanes = document.querySelectorAll('.wizard-pane');
    const prevBtn = document.getElementById('prev-step');
    const nextBtn = document.getElementById('next-step');
    const submitBtn = document.getElementById('submit-prediction');
    const progressBar = document.querySelector('.wizard-progress');
    let currentStep = 0;
    
    if (!wizardSteps.length) return;
    
    // Function to update wizard state
    function updateWizard() {
        // Update steps
        wizardSteps.forEach((step, index) => {
            step.classList.remove('active', 'completed');
            if (index === currentStep) {
                step.classList.add('active');
            } else if (index < currentStep) {
                step.classList.add('completed');
            }
        });
        
        // Update content panes
        wizardPanes.forEach((pane, index) => {
            pane.classList.remove('active');
            if (index === currentStep) {
                pane.classList.add('active');
            }
        });
        
        // Update progress bar
        if (progressBar) {
            const progress = (currentStep / (wizardSteps.length - 1)) * 100;
            progressBar.style.width = `${progress}%`;
        }
        
        // Update buttons
        if (prevBtn) {
            prevBtn.disabled = currentStep === 0;
        }
        
        if (nextBtn) {
            nextBtn.style.display = currentStep < wizardSteps.length - 1 ? 'block' : 'none';
        }
        
        if (submitBtn) {
            submitBtn.style.display = currentStep === wizardSteps.length - 1 ? 'block' : 'none';
        }
    }
    
    // Initialize wizard
    updateWizard();
    
    // Set up step navigation
    wizardSteps.forEach((step, index) => {
        step.addEventListener('click', () => {
            // Only allow going to completed steps or the next available step
            if (index <= currentStep + 1 && validateStep(currentStep)) {
                currentStep = index;
                updateWizard();
            }
        });
    });
    
    // Next button click handler
    if (nextBtn) {
        nextBtn.addEventListener('click', () => {
            if (currentStep < wizardSteps.length - 1 && validateStep(currentStep)) {
                currentStep++;
                updateWizard();
                // Scroll to top of wizard
                document.querySelector('.wizard-container').scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
    
    // Previous button click handler
    if (prevBtn) {
        prevBtn.addEventListener('click', () => {
            if (currentStep > 0) {
                currentStep--;
                updateWizard();
                // Scroll to top of wizard
                document.querySelector('.wizard-container').scrollIntoView({ behavior: 'smooth' });
            }
        });
    }
}

// Validate the current step
function validateStep(stepIndex) {
    switch(stepIndex) {
        case 0: // File Upload step
            const fileInput = document.getElementById('file-input');
            if (!fileInput || fileInput.files.length === 0) {
                showNotification('Please select a CSV file to upload.', 'warning');
                return false;
            }
            return true;
            
        case 1: // Model Selection step
            const modelInput = document.getElementById('model-type');
            if (!modelInput || !modelInput.value) {
                showNotification('Please select a prediction model.', 'warning');
                return false;
            }
            return true;
            
        default:
            return true;
    }
}

// Initialize file upload functionality
function initFileUpload() {
    const fileInput = document.getElementById('file-input');
    const fileUploadContainer = document.querySelector('.file-upload-container');
    const fileNameDisplay = document.getElementById('file-name-display');
    const fileSizeDisplay = document.getElementById('file-size-display');
    const fileDetailsContainer = document.querySelector('.file-details');
    
    if (!fileInput) return;
    
    // Cache the last file for persistence after form submission
    let lastFile = {
        name: null,
        size: null
    };
    
    // Check if previously stored filename/size is in sessionStorage
    const savedFileName = sessionStorage.getItem('lastUploadedFileName');
    const savedFileSize = sessionStorage.getItem('lastUploadedFileSize');
    
    // If we have a saved file name, restore the UI state
    if (savedFileName && savedFileSize) {
        if (fileNameDisplay) {
            fileNameDisplay.textContent = savedFileName;
        }
        
        if (fileSizeDisplay) {
            fileSizeDisplay.textContent = savedFileSize;
        }
        
        // Show file details
        if (fileDetailsContainer) {
            fileDetailsContainer.classList.add('show');
        }
        
        // Add 'has-file' class to container
        if (fileUploadContainer) {
            fileUploadContainer.classList.add('has-file');
        }
    }
    
    // Handle file selection
    fileInput.addEventListener('change', function() {
        if (this.files.length > 0) {
            const file = this.files[0];
            
            // Update file details
            if (fileNameDisplay) {
                fileNameDisplay.textContent = file.name;
                // Save to sessionStorage
                sessionStorage.setItem('lastUploadedFileName', file.name);
            }
            
            if (fileSizeDisplay) {
                const fileSize = formatFileSize(file.size);
                fileSizeDisplay.textContent = fileSize;
                // Save to sessionStorage 
                sessionStorage.setItem('lastUploadedFileSize', fileSize);
            }
            
            // Cache the file info
            lastFile.name = file.name;
            lastFile.size = formatFileSize(file.size);
            
            // Show file details
            if (fileDetailsContainer) {
                fileDetailsContainer.classList.add('show');
            }
            
            // Add 'has-file' class to container
            if (fileUploadContainer) {
                fileUploadContainer.classList.add('has-file');
            }
        } else {
            // Reset UI when no file is selected
            if (fileDetailsContainer) {
                fileDetailsContainer.classList.remove('show');
            }
            
            if (fileUploadContainer) {
                fileUploadContainer.classList.remove('has-file');
            }
            
            // Clear sessionStorage
            sessionStorage.removeItem('lastUploadedFileName');
            sessionStorage.removeItem('lastUploadedFileSize');
            
            // Reset cached file info
            lastFile.name = null;
            lastFile.size = null;
        }
    });
    
    // Handle drag and drop
    if (fileUploadContainer) {
        // Prevent default behavior for drag events
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileUploadContainer.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        // Add visual cues for drag events
        ['dragenter', 'dragover'].forEach(eventName => {
            fileUploadContainer.addEventListener(eventName, () => {
                fileUploadContainer.classList.add('dragover');
            }, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileUploadContainer.addEventListener(eventName, () => {
                fileUploadContainer.classList.remove('dragover');
            }, false);
        });
        
        // Handle file drop
        fileUploadContainer.addEventListener('drop', function(e) {
            const droppedFiles = e.dataTransfer.files;
            if (droppedFiles.length > 0) {
                fileInput.files = droppedFiles;
                // Trigger change event
                const event = new Event('change');
                fileInput.dispatchEvent(event);
            }
        }, false);
    }
}

// Format file size into human-readable format
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Initialize model selection
function initModelSelection() {
    const modelCards = document.querySelectorAll('.model-card');
    const modelInput = document.getElementById('model-type');
    const modelTypeContainers = document.querySelectorAll('.model-type-container');
    const reviewModelName = document.getElementById('review-model-name');
    const reviewModelType = document.getElementById('review-model-type');
    const reviewModelIcon = document.getElementById('review-model-icon');
    
    if (!modelCards.length || !modelInput) return;
    
    // Handle model card selection
    modelCards.forEach(card => {
        card.addEventListener('click', function() {
            // Remove selected class from all cards
            modelCards.forEach(c => c.classList.remove('selected'));
            
            // Add selected class to clicked card
            this.classList.add('selected');
            
            // Update hidden input value
            const modelName = this.dataset.model;
            modelInput.value = modelName;
            
            // Highlight selected model type
            if (modelTypeContainers.length) {
                const modelType = this.closest('.model-type-container').dataset.modelType;
                highlightModelType(modelType);
                
                // Update review section
                updateReviewSection(modelName, modelType);
            }
        });
    });
    
    // Function to highlight selected model type
    function highlightModelType(modelType) {
        modelTypeContainers.forEach(container => {
            if (container.dataset.modelType === modelType) {
                container.classList.add('selected-type');
            } else {
                container.classList.remove('selected-type');
            }
        });
    }
    
    // Function to update review section
    function updateReviewSection(modelName, modelType) {
        // Only update if review elements exist (on step 3)
        if (reviewModelName && reviewModelType && reviewModelIcon) {
            // Update model name in review with proper formatting
            let displayName = modelName.replace(/_/g, ' ');
            displayName = displayName.split(' ').map(word => 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');
            
            reviewModelName.textContent = displayName;
            
            // Update model type
            let typeText = '';
            let iconHTML = '';
            
            switch(modelType) {
                case 'ml':
                    typeText = 'Machine Learning';
                    iconHTML = '<i class="bi bi-diagram-3 text-primary" style="font-size: 2rem;"></i>';
                    break;
                case 'dl':
                    typeText = 'Deep Learning';
                    iconHTML = '<i class="bi bi-cpu-fill text-success" style="font-size: 2rem;"></i>';
                    break;
                case 'qml':
                    typeText = 'Quantum Machine Learning';
                    iconHTML = '<i class="bi bi-tsunami text-info" style="font-size: 2rem;"></i>';
                    break;
                default:
                    typeText = 'Model';
                    iconHTML = '<i class="bi bi-question-circle text-secondary" style="font-size: 2rem;"></i>';
            }
            
            reviewModelType.textContent = typeText;
            reviewModelIcon.innerHTML = iconHTML;
        }
    }
}

// Initialize form validation
function initFormValidation() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('file-input');
    const modelInput = document.getElementById('model-type');
    const loadingContainer = document.getElementById('loading-container');
    
    if (!uploadForm) return;
    
    uploadForm.addEventListener('submit', function(e) {
        // Prevent default form submission
        e.preventDefault();
        
        // Validate file selection - check both file input and saved filename
        const savedFileName = sessionStorage.getItem('lastUploadedFileName');
        if ((!fileInput || fileInput.files.length === 0) && !savedFileName) {
            showNotification('Please select a CSV file to upload.', 'warning');
            return;
        }
        
        // Validate model selection
        if (!modelInput || !modelInput.value) {
            showNotification('Please select a prediction model.', 'warning');
            return;
        }
        
        // Make sure we preserve file name in session storage
        if (fileInput && fileInput.files.length > 0) {
            const file = fileInput.files[0];
            sessionStorage.setItem('lastUploadedFileName', file.name);
            sessionStorage.setItem('lastUploadedFileSize', formatFileSize(file.size));
        }
        
        // Show loading animation
        if (loadingContainer) {
            loadingContainer.style.display = 'flex';
        }
        
        // Submit the form after a short delay for better UX
        setTimeout(() => {
            this.submit();
        }, 500);
    });
}

// Initialize results view with charts and animations
function initResultsView() {
    const exportButton = document.getElementById('export-button');
    const metricsCharts = document.querySelectorAll('[id$="-chart"]');
    
    // Handle export button
    if (exportButton) {
        exportButton.addEventListener('click', function() {
            const sessionId = this.dataset.sessionId;
            if (sessionId) {
                window.location.href = `/export/${sessionId}`;
            }
        });
    }
    
    // Initialize metric charts
    if (metricsCharts.length && typeof Chart !== 'undefined') {
        metricsCharts.forEach(canvas => {
            const value = parseFloat(canvas.dataset.value);
            const id = canvas.id;
            let color = '#0d6efd';
            
            // Set color based on metric type
            if (id.includes('accuracy')) {
                color = '#0d6efd';
            } else if (id.includes('precision')) {
                color = '#198754';
            } else if (id.includes('recall')) {
                color = '#fd7e14';
            } else if (id.includes('f1')) {
                color = '#6f42c1';
            }
            
            // Create gauge chart
            new Chart(canvas, {
                type: 'doughnut',
                data: {
                    datasets: [{
                        data: [value, 1 - value],
                        backgroundColor: [color, 'rgba(200, 200, 200, 0.2)'],
                        borderWidth: 0
                    }]
                },
                options: {
                    cutout: '80%',
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
                    }
                }
            });
        });
    }
    
    // Animate results elements
    const resultsElements = document.querySelectorAll('.results-container .card');
    if (resultsElements.length) {
        resultsElements.forEach((el, index) => {
            setTimeout(() => {
                el.classList.add('animate-in');
            }, 300 * index);
        });
    }
}

// Initialize history table filter
function initHistoryFilter() {
    const historySearch = document.getElementById('history-search');
    
    if (historySearch) {
        historySearch.addEventListener('input', filterHistory);
    }
}

// Filter history table based on search input
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

// Initialize Bootstrap components
function initBootstrapComponents() {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    if (tooltipTriggerList.length > 0 && typeof bootstrap !== 'undefined') {
        tooltipTriggerList.forEach(el => new bootstrap.Tooltip(el));
    }
    
    // Initialize popovers
    const popoverTriggerList = document.querySelectorAll('[data-bs-toggle="popover"]');
    if (popoverTriggerList.length > 0 && typeof bootstrap !== 'undefined') {
        popoverTriggerList.forEach(el => new bootstrap.Popover(el));
    }
}

// Show notification message
function showNotification(message, type = 'info') {
    const notificationContainer = document.getElementById('notification-container');
    
    if (!notificationContainer) {
        // Create notification container if it doesn't exist
        const container = document.createElement('div');
        container.id = 'notification-container';
        container.style.position = 'fixed';
        container.style.top = '20px';
        container.style.right = '20px';
        container.style.zIndex = '9999';
        document.body.appendChild(container);
    }
    
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add notification to container
    document.getElementById('notification-container').appendChild(notification);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}
