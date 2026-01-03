// static/js/dashboard.js

document.addEventListener('DOMContentLoaded', function() {
    fetchDashboardData();
    // Optional: Refresh data every 60 seconds
    // setInterval(fetchDashboardData, 60000);
});

function fetchDashboardData() {
    fetch('/api/dashboard_data')
       .then(response => response.json())
       .then(data => {
            if (data.error) {
                console.error("Error fetching dashboard data:", data.error);
                return;
            }
            
            // Update cards
            document.getElementById('detection-accuracy').textContent = data.detection_accuracy.toFixed(2) + '%';
            document.getElementById('urls-analyzed-today').textContent = data.urls_analyzed_today.toLocaleString();
            document.getElementById('threats-blocked').textContent = data.threats_blocked_today.toLocaleString();
            document.getElementById('system-uptime').textContent = data.system_uptime;

            // Calculate and update change indicators (simplified example)
            const urlsChange = data.urls_analyzed_today - data.urls_analyaled_yesterday;
            const urlsChangePercent = data.urls_analyaled_yesterday > 0? (urlsChange / data.urls_analyaled_yesterday * 100).toFixed(1) : '0.0';
            const urlsChangeElement = document.querySelector('#urls-analyzed-today +.card-footer.change-indicator');
            urlsChangeElement.innerHTML = `<i class="fas fa-arrow-${urlsChange >= 0? 'up' : 'down'}"></i> ${Math.abs(urlsChangePercent)}%`;
            urlsChangeElement.className = `change-indicator ${urlsChange >= 0? 'positive' : 'negative'}`;

            const threatsChange = data.threats_blocked_today - data.threats_blocked_yesterday;
            const threatsChangePercent = data.threats_blocked_yesterday > 0? (threatsChange / data.threats_blocked_yesterday * 100).toFixed(1) : '0.0';
            const threatsChangeElement = document.querySelector('#threats-blocked +.card-footer.change-indicator');
            threatsChangeElement.innerHTML = `<i class="fas fa-arrow-${threatsChange >= 0? 'up' : 'down'}"></i> ${Math.abs(threatsChangePercent)}%`;
            threatsChangeElement.className = `change-indicator ${threatsChange >= 0? 'positive' : 'negative'}`;


            // Render Detection Trends Chart
            renderDetectionTrendChart(data.detection_trend);

            // Update recent threat detections (if not already handled by Jinja)
            // This part is currently handled by Jinja on initial load.
            // If you want dynamic updates, you'd fetch this data via AJAX too.
        })
       .catch(error => console.error('Error:', error));
}

function renderDetectionTrendChart(trendData) {
    const ctx = document.getElementById('detectionTrendChart').getContext('2d');

    const labels = trendData.map(item => item.date);
    const phishingData = trendData.map(item => item.phishing);
    const legitimateData = trendData.map(item => item.legitimate);

    if (window.detectionTrendChartInstance) {
        window.detectionTrendChartInstance.destroy(); // Destroy existing chart instance
    }

    window.detectionTrendChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets:
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'var(--text-color-dark)'
                    }
                },
                y: {
                    beginAtZero: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'var(--text-color-dark)'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: 'var(--text-color-light)'
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'var(--accent-blue)',
                    borderWidth: 1
                }
            }
        }
    });
}