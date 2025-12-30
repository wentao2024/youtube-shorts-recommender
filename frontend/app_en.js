// API Configuration
const API_BASE_URL = window.location.origin.includes('localhost') 
    ? 'http://localhost:8000' 
    : window.location.origin.replace(/:\d+$/, ':8000');

// Global state
let currentRecommendations = [];

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    initNavigation();
    checkHealth();
    
    // Custom recall configuration toggle
    document.getElementById('customRecall').addEventListener('change', function() {
        const config = document.getElementById('recallConfig');
        config.style.display = this.checked ? 'block' : 'none';
    });
});

// Navigation switching
function initNavigation() {
    const navItems = document.querySelectorAll('.nav-item');
    const tabContents = document.querySelectorAll('.tab-content');

    navItems.forEach(item => {
        item.addEventListener('click', function() {
            const tab = this.dataset.tab;
            
            // Update navigation state
            navItems.forEach(nav => nav.classList.remove('active'));
            this.classList.add('active');
            
            // Update content display
            tabContents.forEach(content => content.classList.remove('active'));
            document.getElementById(`${tab}-tab`).classList.add('active');
        });
    });
}

// Health check
async function checkHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        
        const indicator = document.getElementById('statusIndicator');
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');
        
        if (data.status === 'ok') {
            dot.classList.add('online');
            text.textContent = 'Service Online';
        } else {
            dot.classList.remove('online');
            text.textContent = 'Service Error';
        }
    } catch (error) {
        const indicator = document.getElementById('statusIndicator');
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('.status-text');
        dot.classList.remove('online');
        text.textContent = 'Connection Failed';
    }
}

// Show loading
function showLoading() {
    document.getElementById('loading').style.display = 'flex';
}

// Hide loading
function hideLoading() {
    document.getElementById('loading').style.display = 'none';
}

// Show toast message
function showToast(message, type = 'success') {
    const toast = document.getElementById('toast');
    toast.textContent = message;
    toast.className = `toast ${type} show`;
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// Get recommendations
async function getRecommendations() {
    const userId = parseInt(document.getElementById('userId').value);
    const topK = parseInt(document.getElementById('topK').value);
    const customRecall = document.getElementById('customRecall').checked;
    
    if (!userId || userId < 1 || userId > 943) {
        showToast('Please enter a valid User ID (1-943)', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const requestBody = {
            user_id: userId,
            top_k: topK
        };
        
        if (customRecall) {
            requestBody.recall_nums = {
                cf: parseInt(document.getElementById('recallCf').value) || 200,
                popular: parseInt(document.getElementById('recallPopular').value) || 100,
                high_rating: parseInt(document.getElementById('recallHighRating').value) || 100,
                similarity: parseInt(document.getElementById('recallSimilarity').value) || 100
            };
        }
        
        const response = await fetch(`${API_BASE_URL}/recommend`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        currentRecommendations = data.recommendations;
        displayRecommendations(data);
        showToast(`Successfully retrieved ${data.total_candidates} recommendations`, 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showToast('Failed to get recommendations: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Display recommendations
function displayRecommendations(data) {
    const resultsDiv = document.getElementById('recommendResults');
    const listDiv = document.getElementById('recommendationsList');
    const countSpan = document.getElementById('resultsCount');
    
    resultsDiv.style.display = 'block';
    countSpan.textContent = `Total: ${data.total_candidates} recommendations`;
    
    listDiv.innerHTML = '';
    
    if (data.recommendations.length === 0) {
        listDiv.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No recommendations found</p>';
        return;
    }
    
    data.recommendations.forEach((rec, index) => {
        const item = document.createElement('div');
        item.className = 'recommendation-item';
        item.innerHTML = `
            <div style="display: flex; align-items: center; flex: 1;">
                <span class="recommendation-rank">#${index + 1}</span>
                <div class="recommendation-info">
                    <div class="recommendation-title">Video ID: ${rec.video_id}</div>
                    <div class="recommendation-id">Score: ${rec.score.toFixed(4)}</div>
                </div>
            </div>
            <div class="recommendation-score">${(rec.score * 100).toFixed(1)}%</div>
        `;
        listDiv.appendChild(item);
    });
}

// Get recall analysis
async function getRecallAnalysis() {
    const userId = parseInt(document.getElementById('recallUserId').value);
    
    if (!userId || userId < 1 || userId > 943) {
        showToast('Please enter a valid User ID (1-943)', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/recall`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                user_id: userId
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayRecallAnalysis(data);
        showToast('Recall analysis completed', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showToast('Failed to get recall analysis: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Display recall analysis
function displayRecallAnalysis(data) {
    const resultsDiv = document.getElementById('recallResults');
    const statsDiv = document.getElementById('recallStats');
    const detailsDiv = document.getElementById('recallDetails');
    
    resultsDiv.style.display = 'block';
    
    // Statistics
    const recallTypes = Object.keys(data.recall_results);
    statsDiv.innerHTML = '';
    
    recallTypes.forEach(type => {
        const count = data.recall_results[type].length;
        const statCard = document.createElement('div');
        statCard.className = 'recall-stat-card';
        statCard.innerHTML = `
            <div class="recall-stat-label">${getRecallTypeName(type)}</div>
            <div class="recall-stat-value">${count}</div>
        `;
        statsDiv.appendChild(statCard);
    });
    
    // Detailed results
    detailsDiv.innerHTML = '';
    recallTypes.forEach(type => {
        const section = document.createElement('div');
        section.className = 'recall-section';
        section.innerHTML = `
            <h4><i class="fas fa-list"></i> ${getRecallTypeName(type)} (${data.recall_results[type].length} items)</h4>
            <div class="recall-items"></div>
        `;
        
        const itemsDiv = section.querySelector('.recall-items');
        data.recall_results[type].slice(0, 10).forEach(item => {
            const itemDiv = document.createElement('div');
            itemDiv.className = 'recall-item';
            itemDiv.innerHTML = `
                <span>Video ID: ${item.video_id}</span>
                <span style="font-weight: 600; color: var(--primary-color);">Score: ${item.score.toFixed(4)}</span>
            `;
            itemsDiv.appendChild(itemDiv);
        });
        
        if (data.recall_results[type].length > 10) {
            const moreDiv = document.createElement('div');
            moreDiv.style.textAlign = 'center';
            moreDiv.style.padding = '10px';
            moreDiv.style.color = 'var(--text-secondary)';
            moreDiv.textContent = `... and ${data.recall_results[type].length - 10} more results`;
            itemsDiv.appendChild(moreDiv);
        }
        
        detailsDiv.appendChild(section);
    });
}

// Get recall type name
function getRecallTypeName(type) {
    const names = {
        'collaborative_filtering': 'Collaborative Filtering',
        'popularity': 'Popularity',
        'high_rating': 'High Rating',
        'user_similarity': 'User Similarity'
    };
    return names[type] || type;
}

// Get user statistics
async function getUserStats() {
    const userId = parseInt(document.getElementById('statsUserId').value);
    
    if (!userId || userId < 1 || userId > 943) {
        showToast('Please enter a valid User ID (1-943)', 'error');
        return;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE_URL}/user/${userId}/stats`);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        displayUserStats(data);
        showToast('User statistics retrieved successfully', 'success');
        
    } catch (error) {
        console.error('Error:', error);
        showToast('Failed to get user statistics: ' + error.message, 'error');
    } finally {
        hideLoading();
    }
}

// Display user statistics
function displayUserStats(data) {
    const resultsDiv = document.getElementById('userStatsResults');
    const gridDiv = document.getElementById('statsGrid');
    
    resultsDiv.style.display = 'block';
    gridDiv.innerHTML = '';
    
    if (data.message) {
        gridDiv.innerHTML = `<p style="text-align: center; color: var(--text-secondary);">${data.message}</p>`;
        return;
    }
    
    const stats = [
        { label: 'User ID', value: data.user_id },
        { label: 'Rating Count', value: data.rating_count },
        { label: 'Average Rating', value: data.average_rating.toFixed(2) },
        { label: 'Rated Videos', value: data.rated_videos_count },
        { label: 'Unique Videos', value: data.unique_videos }
    ];
    
    stats.forEach(stat => {
        const card = document.createElement('div');
        card.className = 'stat-card';
        card.innerHTML = `
            <div class="stat-label">${stat.label}</div>
            <div class="stat-value">${stat.value}</div>
        `;
        gridDiv.appendChild(card);
    });
}

// Export results
function exportResults() {
    if (currentRecommendations.length === 0) {
        showToast('No data to export', 'error');
        return;
    }
    
    const csv = [
        ['Rank', 'Video ID', 'Score'],
        ...currentRecommendations.map((rec, index) => [
            index + 1,
            rec.video_id,
            rec.score.toFixed(4)
        ])
    ].map(row => row.join(',')).join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', `recommendations_${Date.now()}.csv`);
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showToast('Export successful', 'success');
}

// Periodic health check
setInterval(checkHealth, 30000); // Check every 30 seconds

