// static/js/batch.js

document.addEventListener('DOMContentLoaded', function() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const browseFilesBtn = document.getElementById('browse-files-btn');

    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false); // For entire body
    });

    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    // Handle dropped files
    dropArea.addEventListener('drop', handleDrop, false);

    // Handle browse button click
    browseFilesBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle file input change (when file is selected via browse)
    fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        handleFiles(files);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropArea.classList.add('highlight');
    }

    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    }

    function handleFiles(files) {
        if (files.length > 0) {
            const file = files;
            const formData = new FormData();
            formData.append('file', file);

            // Show a temporary processing indicator or update the queue immediately
            // For a real-time progress, you'd need WebSockets/SSE and a background task queue (e.g., Celery)
            // For now, we'll just submit and let the server redirect/flash message.
            
            // Submit the form programmatically
            fetch('/batch_upload', {
                method: 'POST',
                body: formData
            })
           .then(response => response.text()) // Get response as text to handle Flask's redirect
           .then(html => {
                // Flask redirects, so we need to reload the page to see the flash message and updated queue
                window.location.reload(); 
            })
           .catch(error => {
                console.error('Error uploading file:', error);
                alert('Error uploading file. Please try again.');
            });
        }
    }
});