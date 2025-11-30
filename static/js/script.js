/*
=================================================================
Indonesia Heart Attack Prediction - Custom JavaScript
=================================================================
*/

// ==================== Global Variables ====================
const API_BASE_URL = window.location.origin;

// ==================== Document Ready ====================
$(document).ready(function() {
    console.log('Indonesia Heart Attack Prediction - Initialized');
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize smooth scroll
    initializeSmoothScroll();
    
    // Initialize form validation
    initializeFormValidation();
    
    // Add active class to current nav item
    highlightActiveNav();
    
    // Initialize animations on scroll
    initializeScrollAnimations();
});

// ==================== Tooltip Initialization ====================
function initializeTooltips() {
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// ==================== Smooth Scroll ====================
function initializeSmoothScroll() {
    $('a[href*="#"]:not([href="#"])').click(function() {
        if (location.pathname.replace(/^\//, '') == this.pathname.replace(/^\//, '') 
            && location.hostname == this.hostname) {
            var target = $(this.hash);
            target = target.length ? target : $('[name=' + this.hash.slice(1) + ']');
            if (target.length) {
                $('html, body').animate({
                    scrollTop: target.offset().top - 100
                }, 800);
                return false;
            }
        }
    });
}

// ==================== Highlight Active Nav ====================
function highlightActiveNav() {
    var currentPath = window.location.pathname;
    $('.navbar-nav .nav-link').each(function() {
        var linkPath = $(this).attr('href');
        if (currentPath === linkPath || (currentPath === '/' && linkPath === '/')) {
            $(this).addClass('active');
        }
    });
}

// ==================== Scroll Animations ====================
function initializeScrollAnimations() {
    // Fade in elements on scroll
    $(window).scroll(function() {
        $('.fade-in-scroll').each(function() {
            var elementTop = $(this).offset().top;
            var elementBottom = elementTop + $(this).outerHeight();
            var viewportTop = $(window).scrollTop();
            var viewportBottom = viewportTop + $(window).height();
            
            if (elementBottom > viewportTop && elementTop < viewportBottom) {
                $(this).addClass('visible');
            }
        });
    });
}

// ==================== Form Validation ====================
function initializeFormValidation() {
    // Bootstrap form validation
    var forms = document.querySelectorAll('.needs-validation');
    Array.prototype.slice.call(forms).forEach(function (form) {
        form.addEventListener('submit', function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                event.stopPropagation();
            }
            form.classList.add('was-validated');
        }, false);
    });
}

// ==================== Form Reset Handler ====================
$(document).on('click', 'button[type="reset"]', function() {
    setTimeout(function() {
        $('.was-validated').removeClass('was-validated');
        $('#resultCard').hide();
    }, 100);
});

// ==================== Number Input Validation ====================
// DISABLED - Biarkan user mengetik bebas tanpa validasi otomatis
// Validasi hanya akan jalan saat submit form (HTML5 native validation)

// ==================== Loading State Management ====================
function showLoading(element) {
    $(element).html('<i class="fas fa-spinner fa-spin"></i> Loading...');
    $(element).prop('disabled', true);
}

function hideLoading(element, originalText) {
    $(element).html(originalText);
    $(element).prop('disabled', false);
}

// ==================== API Call Helper ====================
async function makeAPICall(endpoint, method = 'GET', data = null) {
    try {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        const response = await fetch(API_BASE_URL + endpoint, options);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        throw error;
    }
}

// ==================== Notification System ====================
function showNotification(message, type = 'info') {
    const alertClass = `alert-${type}`;
    const iconClass = {
        'success': 'fa-check-circle',
        'danger': 'fa-exclamation-circle',
        'warning': 'fa-exclamation-triangle',
        'info': 'fa-info-circle'
    }[type] || 'fa-info-circle';
    
    const notification = $(`
        <div class="alert ${alertClass} alert-dismissible fade show position-fixed" 
             role="alert" style="top: 80px; right: 20px; z-index: 9999; min-width: 300px;">
            <i class="fas ${iconClass}"></i> ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        </div>
    `);
    
    $('body').append(notification);
    
    setTimeout(function() {
        notification.alert('close');
    }, 5000);
}

// ==================== Copy to Clipboard ====================
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        showNotification('Copied to clipboard!', 'success');
    }, function(err) {
        showNotification('Failed to copy', 'danger');
    });
}

// ==================== Download as PDF (if needed) ====================
function downloadAsPDF(elementId, filename) {
    // This would require jsPDF library
    // Placeholder for future implementation
    showNotification('PDF download feature coming soon!', 'info');
}

// ==================== Print Result ====================
function printResult() {
    window.print();
}

// ==================== Local Storage Helpers ====================
function saveToLocalStorage(key, value) {
    try {
        localStorage.setItem(key, JSON.stringify(value));
        return true;
    } catch (error) {
        console.error('LocalStorage Error:', error);
        return false;
    }
}

function getFromLocalStorage(key) {
    try {
        const item = localStorage.getItem(key);
        return item ? JSON.parse(item) : null;
    } catch (error) {
        console.error('LocalStorage Error:', error);
        return null;
    }
}

function removeFromLocalStorage(key) {
    try {
        localStorage.removeItem(key);
        return true;
    } catch (error) {
        console.error('LocalStorage Error:', error);
        return false;
    }
}

// ==================== Form Data Persistence ====================
// Save form data to localStorage on input change
$('#predictionForm input, #predictionForm select').on('change', function() {
    const formData = $('#predictionForm').serializeArray();
    const formObject = {};
    
    $.each(formData, function(i, field) {
        formObject[field.name] = field.value;
    });
    
    saveToLocalStorage('predictionFormData', formObject);
});

// Restore form data from localStorage on page load
function restoreFormData() {
    const savedData = getFromLocalStorage('predictionFormData');
    
    if (savedData) {
        $.each(savedData, function(name, value) {
            $(`[name="${name}"]`).val(value);
        });
        
        // Show notification
        const restoreAlert = $(`
            <div class="alert alert-info alert-dismissible fade show" role="alert">
                <i class="fas fa-info-circle"></i> 
                Previous form data restored. 
                <button type="button" class="btn btn-sm btn-outline-primary ms-2" id="clearSavedData">
                    Clear Saved Data
                </button>
                <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
            </div>
        `);
        
        $('#predictionForm').prepend(restoreAlert);
        
        $('#clearSavedData').click(function() {
            removeFromLocalStorage('predictionFormData');
            $('#predictionForm')[0].reset();
            restoreAlert.alert('close');
            showNotification('Saved data cleared', 'success');
        });
    }
}

// Restore form data if on predict page
if (window.location.pathname.includes('predict')) {
    $(document).ready(function() {
        restoreFormData();
    });
}

// ==================== Input Formatting ====================
// Format numbers with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

// ==================== Validation Helpers ====================
function validateAge(age) {
    return age >= 25 && age <= 90;
}

function validateBloodPressure(systolic, diastolic) {
    return systolic > diastolic && systolic >= 80 && systolic <= 200 && diastolic >= 40 && diastolic <= 130;
}

function validateCholesterol(total, hdl, ldl) {
    return total > 0 && hdl > 0 && ldl > 0 && (hdl + ldl) <= total;
}

// ==================== Back to Top Button ====================
$(window).scroll(function() {
    if ($(this).scrollTop() > 300) {
        if ($('#backToTop').length === 0) {
            $('body').append(`
                <button id="backToTop" class="btn btn-primary rounded-circle" 
                        style="position: fixed; bottom: 30px; right: 30px; z-index: 1000; 
                               width: 50px; height: 50px; display: none;">
                    <i class="fas fa-arrow-up"></i>
                </button>
            `);
            
            $('#backToTop').fadeIn().click(function() {
                $('html, body').animate({ scrollTop: 0 }, 600);
                return false;
            });
        } else {
            $('#backToTop').fadeIn();
        }
    } else {
        $('#backToTop').fadeOut();
    }
});

// ==================== Keyboard Shortcuts ====================
$(document).keydown(function(e) {
    // Ctrl + K: Focus on search/form
    if (e.ctrlKey && e.keyCode === 75) {
        e.preventDefault();
        $('input:visible:first').focus();
    }
    
    // Escape: Close modals/alerts
    if (e.keyCode === 27) {
        $('.alert').alert('close');
        $('.modal').modal('hide');
    }
});

// ==================== Dynamic Year in Footer ====================
$(document).ready(function() {
    const currentYear = new Date().getFullYear();
    $('footer').find('.current-year').text(currentYear);
});

// ==================== Helper: Check if element is in viewport ====================
function isInViewport(element) {
    const elementTop = $(element).offset().top;
    const elementBottom = elementTop + $(element).outerHeight();
    const viewportTop = $(window).scrollTop();
    const viewportBottom = viewportTop + $(window).height();
    return elementBottom > viewportTop && elementTop < viewportBottom;
}

// ==================== Statistics Counter Animation ====================
function animateCounter(element, target, duration = 2000) {
    const start = 0;
    const increment = target / (duration / 16);
    let current = start;
    
    const timer = setInterval(function() {
        current += increment;
        if (current >= target) {
            current = target;
            clearInterval(timer);
        }
        $(element).text(Math.floor(current));
    }, 16);
}

// Animate counters when visible - FIXED VERSION
$(window).scroll(function() {
    $('.stat-card h3').each(function() {
        if (isInViewport(this) && !$(this).hasClass('animated')) {
            $(this).addClass('animated');
            const target = parseInt($(this).text());
            if (!isNaN(target)) {
                $(this).text('0');
                animateCounter(this, target);
            }
        }
    });
});

// ==================== Error Handling ====================
window.onerror = function(msg, url, lineNo, columnNo, error) {
    console.error('Error: ' + msg + '\nURL: ' + url + '\nLine: ' + lineNo + '\nColumn: ' + columnNo + '\nError object: ' + JSON.stringify(error));
    return false;
};

// ==================== Console Log Styling ====================
console.log('%c Indonesia Heart Attack Prediction ', 'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; font-size: 20px; padding: 10px;');
console.log('%c Developed for AI Course Project ', 'background: #28a745; color: white; font-size: 14px; padding: 5px;');
console.log('%c System ready! ', 'background: #17a2b8; color: white; font-size: 12px; padding: 5px;');

// ==================== Performance Monitoring ====================
if (window.performance) {
    window.addEventListener('load', function() {
        setTimeout(function() {
            const perfData = window.performance.timing;
            const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
            console.log('Page Load Time: ' + pageLoadTime + 'ms');
        }, 0);
    });
}

// ==================== End of Script ====================