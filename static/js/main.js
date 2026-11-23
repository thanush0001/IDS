/**
 * Intelligent IoT IDS - Interactive Frontend Controller
 * Handles AJAX requests, Chart rendering, Real-time indicators, and Dynamic UI actions.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Animate stat counters
  animateCounters();
});

function animateCounters() {
  const counters = document.querySelectorAll('.counter-value');
  counters.forEach(counter => {
    const target = parseFloat(counter.getAttribute('data-target') || '0');
    const isFloat = counter.getAttribute('data-float') === 'true';
    const duration = 1200;
    const startTime = performance.now();

    function update(now) {
      const elapsed = now - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = target * ease;

      counter.textContent = isFloat ? current.toFixed(2) : Math.floor(current).toLocaleString();

      if (progress < 1) {
        requestAnimationFrame(update);
      } else {
        counter.textContent = isFloat ? target.toFixed(2) : target.toLocaleString();
      }
    }
    requestAnimationFrame(update);
  });
}

/**
 * Standardized AJAX caller
 */
async function apiCall(url, method = 'GET', body = null) {
  try {
    const options = {
      method: method,
      headers: {}
    };

    if (body instanceof FormData) {
      options.body = body;
    } else if (body) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }

    const response = await fetch(url, options);
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('API Call Error:', error);
    return { success: false, message: error.message };
  }
}

/**
 * Render Toast Notification
 */
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container') || createToastContainer();
  const toast = document.createElement('div');
  toast.className = `alert alert-${type}`;
  toast.style.boxShadow = '0 4px 15px rgba(0,0,0,0.4)';
  toast.innerHTML = `<i class="fas fa-info-circle"></i> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transition = 'opacity 0.5s';
    setTimeout(() => toast.remove(), 500);
  }, 4000);
}

function createToastContainer() {
  const div = document.createElement('div');
  div.id = 'toast-container';
  div.style.position = 'fixed';
  div.style.top = '20px';
  div.style.right = '20px';
  div.style.zIndex = '9999';
  document.body.appendChild(div);
  return div;
}

/**
 * Acknowledge an intrusion alert
 */
async function acknowledgeAlert(alertId, btnElement) {
  if (btnElement) {
    btnElement.disabled = true;
    btnElement.innerHTML = '<span class="spinner"></span>';
  }

  const res = await apiCall(`/api/alerts/review/${alertId}`, 'POST');
  if (res.success) {
    showToast(`Alert #${alertId} marked as reviewed.`, 'success');
    const row = document.getElementById(`alert-row-${alertId}`);
    if (row) {
      const badge = row.querySelector('.status-badge');
      if (badge) {
        badge.className = 'badge badge-normal';
        badge.textContent = 'Reviewed';
      }
      if (btnElement) btnElement.remove();
    }
  } else {
    showToast(res.message || 'Failed to acknowledge alert.', 'danger');
    if (btnElement) {
      btnElement.disabled = false;
      btnElement.textContent = 'Review';
    }
  }
}
