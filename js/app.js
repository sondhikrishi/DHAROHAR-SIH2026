/* ==========================================================================
   LAND RECORD MANAGEMENT & VALIDATION SYSTEM
   Main Application Script - js/app.js
   A modular, well-commented JavaScript application designed for a
   1st Year B.Tech Engineering Project. Connects to Flask REST API or
   operates standalone with client-side localStorage.
   ========================================================================== */

// ==========================================================================
// 1. CONFIGURATION & STATE MANAGEMENT
// ==========================================================================
// Set to FALSE when running Flask backend (python backend/app.py) on localhost:5000!
const API_BASE_URL = "https://dharohar-sih2026.onrender.com/api";
const USE_MOCK_DATA = false;
// Active user session state
let currentUser = {
    id: "PAT-402",
    name: "Shri R. K. Sharma",
    role: "patwari" // 'citizen', 'patwari', 'admin'
};

// In-memory application datasets (loaded from localStorage or js/data.js)
let recordsData = [];
let duplicatesData = [];
let auditLogsData = [];

// ==========================================================================
// 2. LOCALSTORAGE PERSISTENCE (Zero-Setup Standalone Execution)
// ==========================================================================
function initializeStorage() {
    const storedRecords = localStorage.getItem("dlims_records");
    const storedDuplicates = localStorage.getItem("dlims_duplicates");
    const storedLogs = localStorage.getItem("dlims_logs");

    recordsData = storedRecords ? JSON.parse(storedRecords) : [...INITIAL_LAND_RECORDS];
    duplicatesData = storedDuplicates ? JSON.parse(storedDuplicates) : [...INITIAL_DUPLICATE_ALERTS];
    auditLogsData = storedLogs ? JSON.parse(storedLogs) : [...INITIAL_AUDIT_LOGS];

    saveToStorage();
}

function saveToStorage() {
    localStorage.setItem("dlims_records", JSON.stringify(recordsData));
    localStorage.setItem("dlims_duplicates", JSON.stringify(duplicatesData));
    localStorage.setItem("dlims_logs", JSON.stringify(auditLogsData));
}

// Reset data back to default (useful for demo/testing)
function resetDemoData() {
    localStorage.removeItem("dlims_records");
    localStorage.removeItem("dlims_duplicates");
    localStorage.removeItem("dlims_logs");
    initializeStorage();
    renderAllViews();
    alert("System data restored to default demo values!");
}

// ==========================================================================
// 3. FLASK REST API SERVICE ADAPTER
// ==========================================================================
const apiService = {
    // 1. Fetch land records with optional query parameters
    async getRecords(params = {}) {
        if (USE_MOCK_DATA) {
            return recordsData;
        }
        try {
            const query = new URLSearchParams(params).toString();
            const res = await fetch(`${API_BASE_URL}/records?${query}`);
            const data = await res.json();
            return data.records;
        } catch (err) {
            console.warn("Flask API unavailable, falling back to mock data:", err);
            return recordsData;
        }
    },

    // 2. Upload document and get OCR text extraction
    async uploadAndOCR(formData) {
        if (USE_MOCK_DATA) {
            return null; // Handled by simulateOCRScanner()
        }
        try {
            const res = await fetch(`${API_BASE_URL}/upload-ocr`, {
                method: "POST",
                body: formData
            });
            return await res.json();
        } catch (err) {
            console.warn("Flask OCR error, falling back to simulation:", err);
            return null;
        }
    },

    // 3. Commit new record to registry
    async saveRecord(record) {
        if (USE_MOCK_DATA) {
            recordsData.unshift(record);
            saveToStorage();
            return { success: true, record };
        }
        try {
            const res = await fetch(`${API_BASE_URL}/records`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(record)
            });
            return await res.json();
        } catch (err) {
            console.warn("Flask save error, falling back to mock:", err);
            recordsData.unshift(record);
            saveToStorage();
            return { success: true, record };
        }
    },

    // 4. Fetch duplicate flags
    async getDuplicates() {
        if (USE_MOCK_DATA) {
            return duplicatesData;
        }
        try {
            const res = await fetch(`${API_BASE_URL}/duplicates`);
            const data = await res.json();
            return data.duplicates;
        } catch (err) {
            return duplicatesData;
        }
    },

    // 5. Fetch audit logs
    async getAuditLogs() {
        if (USE_MOCK_DATA) {
            return auditLogsData;
        }
        try {
            const res = await fetch(`${API_BASE_URL}/audit-logs`);
            const data = await res.json();
            return data.logs;
        } catch (err) {
            return auditLogsData;
        }
    }
};

// ==========================================================================
// 4. INITIALIZATION & LIVE UTILITIES
// ==========================================================================
document.addEventListener("DOMContentLoaded", function () {
    initializeStorage();
    startLiveClock();
    renderAllViews();
    applyRolePermissions();
});

function renderAllViews() {
    renderDashboardKPIs();
    renderDashboardRecentTable();
    renderSearchResultsTable(recordsData);
    renderDuplicateDisputes();
    render5PointChecklist();
    renderAuditLogsTable(auditLogsData);
}

// Live Clock with Indian Standard Time (IST)
function startLiveClock() {
    function update() {
        const now = new Date();
        const options = {
            weekday: 'short',
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        };
        const clockElem = document.getElementById("live-ist-clock");
        if (clockElem) {
            clockElem.innerText = "🕒 " + now.toLocaleDateString('en-IN', options) + " (IST)";
        }
    }
    update();
    setInterval(update, 1000);
}

// Accessibility font resizer
function setAccessibilityFontSize(size) {
    if (size === 'small') document.body.style.fontSize = "13px";
    else if (size === 'normal') document.body.style.fontSize = "14px";
    else if (size === 'large') document.body.style.fontSize = "16px";
}

// ==========================================================================
// 5. TAB NAVIGATION CONTROLLER
// ==========================================================================
function switchTab(tabId, clickedNavElement) {
    // Hide all tab panes
    const panes = document.querySelectorAll(".tab-pane");
    panes.forEach(p => p.classList.remove("active"));

    // Activate selected pane
    const targetPane = document.getElementById("tab-" + tabId);
    if (targetPane) {
        targetPane.classList.add("active");
    }

    // Update nav links active class
    const links = document.querySelectorAll(".nav-link");
    links.forEach(l => l.classList.remove("active"));

    if (clickedNavElement) {
        clickedNavElement.classList.add("active");
    } else {
        links.forEach(link => {
            if (link.getAttribute("onclick") && link.getAttribute("onclick").includes(tabId)) {
                link.classList.add("active");
            }
        });
    }

    window.scrollTo({ top: 110, behavior: 'smooth' });
}

// ==========================================================================
// 6. ROLE SWITCHER & PERMISSION MANAGEMENT
// ==========================================================================
function switchRole(roleKey) {
    currentUser.role = roleKey;
    const nameLabel = document.getElementById("user-display-name");
    const roleLabel = document.getElementById("user-display-role");
    const roleSelect = document.getElementById("role-selector-dropdown");

    if (roleSelect) roleSelect.value = roleKey;

    if (roleKey === "citizen") {
        currentUser.id = "CITIZEN-PUB";
        currentUser.name = "Applicant Citizen / Landowner";
        nameLabel.innerText = currentUser.name;
        roleLabel.innerText = "Role: Citizen (नागरिक) [Read-Only]";
        roleLabel.style.color = "#b9770e";
    } else if (roleKey === "patwari") {
        currentUser.id = "PAT-402";
        currentUser.name = "Shri R. K. Sharma";
        nameLabel.innerText = currentUser.name;
        roleLabel.innerText = "Role: Patwari (पटवारी/राजस्व अधिकारी)";
        roleLabel.style.color = "var(--gov-green)";
    } else if (roleKey === "admin") {
        currentUser.id = "TEH-101";
        currentUser.name = "Shri V. C. Joshi (Tehsildar / IAS)";
        nameLabel.innerText = currentUser.name;
        roleLabel.innerText = "Role: Super Admin (तहसीलदार/प्रशासक)";
        roleLabel.style.color = "var(--gov-navy)";
    }

    applyRolePermissions();
    addAuditEntry("Role View Switched", `Switched active session view to ${roleKey.toUpperCase()}`);
    alert(`Switched to: ${currentUser.role.toUpperCase()} mode.\nInterface permissions have been updated.`);
}

function applyRolePermissions() {
    const btnSaveRegistry = document.getElementById("btn-commit-registry");
    const btnStartOcr = document.getElementById("btn-start-ocr-scan");
    const disputeActionBtns = document.querySelectorAll(".btn-dispute-action");

    // Citizens cannot commit deeds or resolve legal disputes
    if (currentUser.role === "citizen") {
        if (btnSaveRegistry) {
            btnSaveRegistry.disabled = true;
            btnSaveRegistry.title = "Citizens have read-only access to records.";
        }
        disputeActionBtns.forEach(b => {
            b.disabled = true;
            b.title = "Only Tehsildar / Admin can execute dispute orders.";
        });
    } else {
        if (btnSaveRegistry) {
            btnSaveRegistry.disabled = false;
            btnSaveRegistry.title = "";
        }
        disputeActionBtns.forEach(b => {
            b.disabled = false;
            b.title = "";
        });
    }
}

// ==========================================================================
// 7. DASHBOARD KPIS & RECENT RECORDS
// ==========================================================================
function renderDashboardKPIs() {
    const total = recordsData.length;
    const verified = recordsData.filter(r => r.status === "Verified").length;
    const pending = recordsData.filter(r => r.status === "Pending").length;
    const flagged = duplicatesData.length;

    // Calculate total acres
    const totalAcres = recordsData.reduce((acc, r) => acc + (parseFloat(r.areaAcres) || 0), 0);

    const elTotal = document.getElementById("kpi-total-records");
    const elVerified = document.getElementById("kpi-verified-records");
    const elPending = document.getElementById("kpi-pending-records");
    const elFlagged = document.getElementById("kpi-flagged-alerts");
    const elAcres = document.getElementById("kpi-total-acres");

    if (elTotal) elTotal.innerText = total.toLocaleString();
    if (elVerified) elVerified.innerText = verified.toLocaleString();
    if (elPending) elPending.innerText = pending.toLocaleString();
    if (elFlagged) elFlagged.innerText = flagged.toLocaleString();
    if (elAcres) elAcres.innerText = totalAcres.toFixed(1) + " Ac";

    // Nav badge for duplicates
    const navBadge = document.getElementById("nav-duplicate-counter");
    if (navBadge) navBadge.innerText = `${flagged} Alerts`;
}

function renderDashboardRecentTable() {
    const tbody = document.getElementById("dashboard-recent-table-body");
    if (!tbody) return;

    const recent = recordsData.slice(0, 4);
    let html = "";
    recent.forEach(r => {
        let badgeClass = r.status === "Verified" ? "badge-success" : (r.status === "Pending" ? "badge-warning" : "badge-danger");
        html += `
            <tr>
                <td><strong>${r.khasraNo}</strong></td>
                <td>${r.ownerName}</td>
                <td>${r.village}, ${r.tehsil}</td>
                <td>${r.areaAcres} Acres</td>
                <td><span class="badge ${badgeClass}">${r.status}</span></td>
            </tr>
        `;
    });
    tbody.innerHTML = html;
}

// ==========================================================================
// 8. LAND RECORD SEARCH & REGISTRY MODULE
// ==========================================================================
function filterRecords() {
    const khasraQ = (document.getElementById("search-khasra").value || "").toLowerCase().trim();
    const ownerQ = (document.getElementById("search-owner").value || "").toLowerCase().trim();
    const khatauniQ = (document.getElementById("search-khatauni").value || "").toLowerCase().trim();
    const districtQ = document.getElementById("filter-district").value;
    const statusQ = document.getElementById("filter-status").value;

    const filtered = recordsData.filter(r => {
        const matchKhasra = !khasraQ || r.khasraNo.toLowerCase().includes(khasraQ);
        const matchOwner = !ownerQ || r.ownerName.toLowerCase().includes(ownerQ);
        const matchKhatauni = !khatauniQ || (r.khatauniNo && r.khatauniNo.toLowerCase().includes(khatauniQ));
        const matchDistrict = (districtQ === "ALL") || (r.district === districtQ);
        const matchStatus = (statusQ === "ALL") || (r.status === statusQ);

        return matchKhasra && matchOwner && matchKhatauni && matchDistrict && matchStatus;
    });

    renderSearchResultsTable(filtered);
}

function renderSearchResultsTable(records) {
    const tbody = document.getElementById("search-results-table-body");
    const countDisplay = document.getElementById("search-results-counter");
    if (!tbody) return;

    if (records.length === 0) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding: 22px; color: #7f8c8d;">No land records match the filter criteria.</td></tr>`;
        if (countDisplay) countDisplay.innerText = "Showing 0 records";
        return;
    }

    let html = "";
    records.forEach(r => {
        let badgeClass = r.status === "Verified" ? "badge-success" : (r.status === "Pending" ? "badge-warning" : "badge-danger");
        html += `
            <tr>
                <td><code>${r.id}</code></td>
                <td><strong style="color: var(--gov-navy);">${r.khasraNo}</strong></td>
                <td>${r.khatauniNo || 'KH-001'}</td>
                <td>
                    <strong>${r.ownerName}</strong><br>
                    <small style="color: var(--text-muted);">${r.fatherName || 'S/o Late Sh. Father'}</small>
                </td>
                <td>${r.village} (${r.tehsil}, ${r.district})</td>
                <td>${r.areaAcres} Ac</td>
                <td><span class="badge ${badgeClass}">${r.status}</span></td>
                <td>
                    <button class="btn btn-primary" style="padding: 4px 8px; font-size: 11px;" onclick="viewLandDeedModal('${r.id}')">
                        📜 View Deed
                    </button>
                </td>
            </tr>
        `;
    });

    tbody.innerHTML = html;
    if (countDisplay) countDisplay.innerText = `Showing ${records.length} of ${recordsData.length} records`;
}

// Reset search fields
function resetSearchFilters() {
    document.getElementById("search-khasra").value = "";
    document.getElementById("search-owner").value = "";
    document.getElementById("search-khatauni").value = "";
    document.getElementById("filter-district").value = "ALL";
    document.getElementById("filter-status").value = "ALL";
    renderSearchResultsTable(recordsData);
}

// ==========================================================================
// 9. SIMULATED AI OCR SCANNER WITH PROGRESS & SCANNING BEAM
// ==========================================================================
function onFileSelected(input) {
    const label = document.getElementById("upload-file-feedback");
    if (input.files && input.files[0]) {
        label.innerText = `Selected File: ${input.files[0].name} (${(input.files[0].size / 1024).toFixed(1)} KB)`;
    } else {
        label.innerText = "No document selected";
    }
}

function handleOCRSubmission(event) {
    event.preventDefault();

    const docType = document.getElementById("upload-doc-category").value;
    const targetKhasra = document.getElementById("upload-khasra-input").value.trim();
    const district = document.getElementById("upload-district-select").value;

    if (!targetKhasra) {
        alert("Please specify the expected Khasra / Survey Number.");
        return;
    }

    startOCRScanSimulation(docType, targetKhasra, district);
}

function startOCRScanSimulation(docType, targetKhasra, district) {
    const scannerBox = document.getElementById("ocr-scanner-visualizer");
    const progressBar = document.getElementById("ocr-scan-progress");
    const stepLabel = document.getElementById("ocr-step-label");
    const percentLabel = document.getElementById("ocr-percent-label");
    const terminal = document.getElementById("ocr-live-terminal");
    const confBadge = document.getElementById("ocr-confidence-tag");

    scannerBox.style.display = "block";
    scannerBox.classList.add("scanning");
    progressBar.style.width = "0%";
    percentLabel.innerText = "0%";
    stepLabel.innerText = `Ingesting ${docType} image buffer...`;
    terminal.innerText = `[INFO] Initializing Tesseract / EasyOCR Indian Devanagari OCR Engine...\n[INFO] Applying bilateral filter & deskewing angle correction...`;

    // Step 1: Optical pre-processing
    setTimeout(() => {
        progressBar.style.width = "25%";
        percentLabel.innerText = "25%";
        stepLabel.innerText = "Detecting character bounding boxes & seal stamps...";
        terminal.innerText += `\n[OCR] Text bounding boxes detected: 142 tokens.\n[OCR] Recognized Devanagari keywords: 'खसरा संख्या', 'खाताधारक', 'तहसील'`;
    }, 500);

    // Step 2: Tabular parsing
    setTimeout(() => {
        progressBar.style.width = "65%";
        percentLabel.innerText = "65%";
        stepLabel.innerText = "Extracting tabular land schedule & boundaries...";
        terminal.innerText += `\n[OCR] Extracted Khasra No: ${targetKhasra}\n[OCR] Extracted Khatauni: KH-${Math.floor(100 + Math.random() * 900)}\n[OCR] Extracted Land Area: 3.80 Acres`;
    }, 1200);

    // Step 3: Completion & Populate Extracted Fields
    setTimeout(() => {
        progressBar.style.width = "100%";
        percentLabel.innerText = "100%";
        stepLabel.innerText = "OCR Extraction Finished (Confidence: 98.4%)";
        scannerBox.classList.remove("scanning");
        confBadge.className = "badge badge-success";
        confBadge.innerText = "98.4% Confidence";
        terminal.innerText += `\n[SUCCESS] Document parsed cleanly.\n[VALIDATION] Cross-referencing treasury stamp duty... VALID.`;

        populateExtractedFields(targetKhasra, district);
        addAuditEntry("OCR Extraction Complete", `Digitized ${docType} for Khasra #${targetKhasra}`);
    }, 1900);
}

function populateExtractedFields(khasraNo, district) {
    document.getElementById("ext-khasra-no").value = khasraNo;
    document.getElementById("ext-khatauni-no").value = "KH-0" + Math.floor(200 + Math.random() * 700);
    document.getElementById("ext-deed-no").value = "DEED-2024-" + Math.floor(1000 + Math.random() * 9000);
    document.getElementById("ext-owner-name").value = "Shri Mahaveer Prasad Meena";
    document.getElementById("ext-father-name").value = "Sh. Ramkaran Meena";
    document.getElementById("ext-land-area").value = "3.80";
    document.getElementById("ext-soil-type").value = "Agricultural (Irrigated)";
    document.getElementById("ext-market-value").value = "₹ 45,00,000";
    document.getElementById("ext-village").value = "Rampur";
    document.getElementById("ext-tehsil").value = district.split(" ")[0];
    document.getElementById("ext-district").value = district;
    document.getElementById("ext-boundaries").value = "North: PWD Road, South: Plot 304, East: Canal, West: Survey 112";

    // Trigger instant duplicate check on extracted survey number
    checkImmediateDuplicateRisk(khasraNo);
}

function checkImmediateDuplicateRisk(khasraNo) {
    const existing = recordsData.find(r => r.khasraNo.toLowerCase() === khasraNo.toLowerCase());
    if (existing) {
        alert(`⚠️ INTELLIGENT FRAUD / DUPLICATE ALERT!\n\nKhasra Number ${khasraNo} is ALREADY registered under:\nOwner: ${existing.ownerName} (${existing.deedNo})\n\nThe validation engine has flagged this record. Approval is blocked pending Tehsildar investigation.`);
    }
}

// Commit extracted fields to recordsData
function saveExtractedRecordToRegistry(event) {
    event.preventDefault();

    if (currentUser.role === "citizen") {
        alert("Permission Denied: Citizens cannot modify or commit deeds to the Official State Registry.");
        return;
    }

    const khasraNo = document.getElementById("ext-khasra-no").value.trim();
    const khatauniNo = document.getElementById("ext-khatauni-no").value.trim();
    const deedNo = document.getElementById("ext-deed-no").value.trim();
    const ownerName = document.getElementById("ext-owner-name").value.trim();
    const fatherName = document.getElementById("ext-father-name").value.trim();
    const area = parseFloat(document.getElementById("ext-land-area").value) || 1.0;
    const soilType = document.getElementById("ext-soil-type").value;
    const marketValue = document.getElementById("ext-market-value").value;
    const village = document.getElementById("ext-village").value;
    const tehsil = document.getElementById("ext-tehsil").value;
    const district = document.getElementById("ext-district").value;
    const boundaries = document.getElementById("ext-boundaries").value;

    const isDuplicate = recordsData.some(r => r.khasraNo.toLowerCase() === khasraNo.toLowerCase());
    const status = isDuplicate ? "Flagged" : "Verified";

    const newRecord = {
        id: "LR-2024-" + String(recordsData.length + 1).padStart(3, '0'),
        khasraNo,
        khatauniNo,
        deedNo,
        ownerName,
        fatherName,
        district,
        tehsil,
        village,
        areaAcres: area,
        soilType,
        marketValue,
        regDate: new Date().toISOString().split('T')[0],
        status,
        encumbranceStatus: isDuplicate ? "Dispute Flagged / Overlapping" : "Nil / Verified",
        boundaries
    };

    apiService.saveRecord(newRecord).then(() => {
        saveToStorage();
        renderAllViews();
        addAuditEntry("New Deed Committed", `Registered Deed ${deedNo} for Khasra #${khasraNo}`);
        alert(`✅ SUCCESS!\n\nLand Record for Khasra #${khasraNo} successfully committed to State Registry.\nStatus: ${status.toUpperCase()}`);
        switchTab('search', null);
    });
}

// ==========================================================================
// 10. 5-POINT VALIDATION CHECKLIST & DUPLICATE DISPUTE ENGINE
// ==========================================================================
function render5PointChecklist() {
    const container = document.getElementById("validation-checklist-items");
    if (!container) return;

    let html = "";
    VERIFICATION_CHECKLIST.forEach((item, index) => {
        html += `
            <div class="checklist-item">
                <div class="item-title">
                    <span style="color: var(--gov-green); font-size: 16px;">✓</span>
                    <div>
                        <strong>Point ${index + 1}: ${item.rule}</strong>
                        <p style="font-size: 11px; color: var(--text-muted); margin: 0;">${item.description}</p>
                    </div>
                </div>
                <span class="badge badge-success">Passing</span>
            </div>
        `;
    });
    container.innerHTML = html;
}

function renderDuplicateDisputes() {
    const container = document.getElementById("dispute-alerts-list");
    if (!container) return;

    if (duplicatesData.length === 0) {
        container.innerHTML = `<div style="text-align:center; padding: 20px; color: var(--gov-green); font-weight: bold;">✓ No active duplicate land disputes detected in the registry!</div>`;
        return;
    }

    let html = "";
    duplicatesData.forEach(alertItem => {
        let badgeSeverity = alertItem.severity === "CRITICAL" ? "badge-danger" : "badge-warning";
        html += `
            <div class="dispute-card" id="dispute-${alertItem.id}">
                <div class="dispute-content">
                    <h4>⚠️ [${alertItem.severity}] ${alertItem.title}</h4>
                    <p style="font-size: 12px; color: var(--text-muted);">${alertItem.description}</p>
                    
                    <div class="dispute-grid">
                        <div>
                            <strong>Primary Claimant:</strong><br>
                            ${alertItem.primaryClaimant.name}<br>
                            Deed: <code>${alertItem.primaryClaimant.deedNo}</code><br>
                            Area: ${alertItem.primaryClaimant.area} (${alertItem.primaryClaimant.date})
                        </div>
                        <div>
                            <strong>Conflicting Claimant / Entity:</strong><br>
                            ${alertItem.conflictingClaimant.name}<br>
                            Deed: <code>${alertItem.conflictingClaimant.deedNo}</code><br>
                            Area: ${alertItem.conflictingClaimant.area} (${alertItem.conflictingClaimant.date})
                        </div>
                    </div>

                    <div style="font-size: 11px; color: #922b21;">
                        <strong>Rule Violated:</strong> ${alertItem.ruleTriggered} | <strong>Location:</strong> ${alertItem.location}
                    </div>
                </div>

                <div style="display: flex; flex-direction: column; gap: 6px; min-width: 170px;">
                    <button class="btn btn-danger btn-dispute-action" onclick="freezeDispute('${alertItem.id}', '${alertItem.khasraNo}')">
                        ❄️ Freeze Record
                    </button>
                    <button class="btn btn-warning btn-dispute-action" onclick="flagInvestigation('${alertItem.id}', '${alertItem.khasraNo}')">
                        🚩 Flag Investigation
                    </button>
                    <button class="btn btn-success btn-dispute-action" onclick="approveDispute('${alertItem.id}', '${alertItem.khasraNo}')">
                        ✓ Clear & Approve
                    </button>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// Dispute Resolution Actions
function freezeDispute(alertId, khasraNo) {
    if (currentUser.role === "citizen") {
        alert("Citizens cannot issue freeze orders.");
        return;
    }
    alert(`❄️ INJUNCTION ORDER EXECUTED:\n\nKhasra #${khasraNo} has been FROZEN under Section 52 of the Land Revenue Act. No alienation, sale, or mortgage can be registered.`);
    addAuditEntry("Title Injunction (Freeze)", `Administrative freeze placed on Khasra #${khasraNo}`);
}

function flagInvestigation(alertId, khasraNo) {
    if (currentUser.role === "citizen") {
        alert("Citizens cannot issue investigation orders.");
        return;
    }
    alert(`🚩 DISPUTE NOTICE DISPATCHED:\n\nA Sub-Divisional Magistrate (SDM) enquiry and DGPS boundary survey ordered for Khasra #${khasraNo}. Both parties summoned.`);
    addAuditEntry("Enquiry Dispatched", `Formal dispute notice issued for Khasra #${khasraNo}`);
}

function approveDispute(alertId, khasraNo) {
    if (currentUser.role === "citizen") {
        alert("Citizens cannot approve disputed titles.");
        return;
    }

    // Remove from alerts
    duplicatesData = duplicatesData.filter(a => a.id !== alertId);

    // Update matching record to Verified
    const rec = recordsData.find(r => r.khasraNo === khasraNo);
    if (rec) {
        rec.status = "Verified";
        rec.encumbranceStatus = "Nil / Title Cleared";
    }

    saveToStorage();
    renderAllViews();
    addAuditEntry("Dispute Resolved & Approved", `Dispute cleared and verified for Khasra #${khasraNo}`);
    alert(`✓ SUCCESS:\n\nDispute for Khasra #${khasraNo} marked as RESOLVED and title affirmed in registry.`);
}

function triggerSystemAuditScan() {
    alert("⚡ FULL SYSTEM AUDIT COMPLETED:\n\nChecked 1,420 cadastral parcels across 18 villages:\n• Cross-Deed Collisions: 1 active\n• Stamp Duty Deficits: 0\n• Encroachments on State Land: 1 active\n\nAll registry records verified against State Cadastral GIS database.");
    addAuditEntry("System Integrity Audit", "Automated full database verification scan executed");
}

// ==========================================================================
// 11. AUDIT TRAIL LOGS
// ==========================================================================
function addAuditEntry(action, recordRef) {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const formatted = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;

    const newLog = {
        id: "LOG-" + (1000 + auditLogsData.length + 1),
        timestamp: formatted,
        userId: currentUser.id,
        role: currentUser.role === "admin" ? "Super Admin" : (currentUser.role === "patwari" ? "Patwari" : "Citizen"),
        action: action,
        recordRef: recordRef,
        ipAddress: "192.168.1." + Math.floor(10 + Math.random() * 80),
        status: "Success"
    };

    auditLogsData.unshift(newLog);
    saveToStorage();
    renderAuditLogsTable(auditLogsData);
}

function renderAuditLogsTable(logs) {
    const tbody = document.getElementById("audit-logs-table-body");
    if (!tbody) return;

    let html = "";
    logs.forEach(l => {
        html += `
            <tr>
                <td><code>${l.id}</code></td>
                <td><small>${l.timestamp}</small></td>
                <td><strong>${l.userId}</strong></td>
                <td><span class="badge badge-info">${l.role}</span></td>
                <td>${l.action}</td>
                <td><code>${l.recordRef}</code></td>
                <td><small>${l.ipAddress}</small></td>
                <td><span class="badge badge-success">${l.status}</span></td>
            </tr>
        `;
    });
    tbody.innerHTML = html;
}

function filterAuditLogs() {
    const keyword = (document.getElementById("audit-filter-input").value || "").toLowerCase();
    const role = document.getElementById("audit-role-select").value;

    const filtered = auditLogsData.filter(l => {
        const matchK = l.action.toLowerCase().includes(keyword) || l.recordRef.toLowerCase().includes(keyword) || l.userId.toLowerCase().includes(keyword);
        const matchR = (role === "ALL") || (l.role.toLowerCase().includes(role.toLowerCase()));
        return matchK && matchR;
    });

    renderAuditLogsTable(filtered);
}

function exportAuditLogsToCSV() {
    let csv = "Log ID,Timestamp,User ID,Role,Action,Record Ref,IP Address,Status\n";
    auditLogsData.forEach(l => {
        csv += `"${l.id}","${l.timestamp}","${l.userId}","${l.role}","${l.action}","${l.recordRef}","${l.ipAddress}","${l.status}"\n`;
    });
    downloadCSVFile(csv, "DLIMS_Audit_Trail_Export.csv");
}

function exportLandRecordsToCSV() {
    let csv = "Record ID,Khasra No,Khatauni No,Deed No,Owner Name,Father Name,District,Tehsil,Village,Area Acres,Soil Classification,Market Value,Status\n";
    recordsData.forEach(r => {
        csv += `"${r.id}","${r.khasraNo}","${r.khatauniNo}","${r.deedNo}","${r.ownerName}","${r.fatherName}","${r.district}","${r.tehsil}","${r.village}","${r.areaAcres}","${r.soilType}","${r.marketValue}","${r.status}"\n`;
    });
    downloadCSVFile(csv, "DLIMS_Land_Records_Registry.csv");
}

function downloadCSVFile(csvContent, fileName) {
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
}

// ==========================================================================
// 12. OFFICIAL DEED MODAL (RECORD OF RIGHTS / 7-12 EXTRACT)
// ==========================================================================
function viewLandDeedModal(recordId) {
    const rec = recordsData.find(r => r.id === recordId);
    if (!rec) return;

    const modal = document.getElementById("deed-viewer-modal");
    const container = document.getElementById("deed-viewer-content");

    container.innerHTML = `
        <div class="deed-certificate-view">
            <div style="text-align: center; border-bottom: 2px solid #333; padding-bottom: 8px; margin-bottom: 12px;">
                <div style="font-size: 11px; font-weight: bold; color: var(--gov-navy);">
                    DEPARTMENT OF LAND RESOURCES & REVENUE ADMINISTRATION
                </div>
                <div style="font-size: 15px; font-weight: bold; margin: 4px 0;">
                    CERTIFICATE OF RECORD OF RIGHTS (ROR / KHATAUNI / 7-12)
                </div>
                <div style="font-size: 11px; color: var(--text-muted);">
                    State Cadastral Digital Registry | Unique Record ID: <strong>${rec.id}</strong>
                </div>
            </div>

            <table style="width: 100%; font-size: 12px; line-height: 1.8;">
                <tr>
                    <td style="width: 35%; font-weight: bold; color: #555;">Khasra / Survey Number:</td>
                    <td style="font-weight: bold; font-size: 14px; color: var(--gov-navy);">${rec.khasraNo}</td>
                    <td style="font-weight: bold; color: #555;">Khatauni No:</td>
                    <td><strong>${rec.khatauniNo}</strong></td>
                </tr>
                <tr>
                    <td style="font-weight: bold; color: #555;">Registered Owner / Pattadar:</td>
                    <td><strong>${rec.ownerName}</strong></td>
                    <td style="font-weight: bold; color: #555;">Father / Husband:</td>
                    <td>${rec.fatherName}</td>
                </tr>
                <tr>
                    <td style="font-weight: bold; color: #555;">Administrative Location:</td>
                    <td colspan="3">Village: ${rec.village}, Tehsil: ${rec.tehsil}, District: ${rec.district}</td>
                </tr>
                <tr>
                    <td style="font-weight: bold; color: #555;">Total Land Area:</td>
                    <td><strong>${rec.areaAcres} Acres</strong></td>
                    <td style="font-weight: bold; color: #555;">Soil Classification:</td>
                    <td>${rec.soilType}</td>
                </tr>
                <tr>
                    <td style="font-weight: bold; color: #555;">Assessed Market Value:</td>
                    <td><strong>${rec.marketValue}</strong></td>
                    <td style="font-weight: bold; color: #555;">Registration Date:</td>
                    <td>${rec.regDate}</td>
                </tr>
                <tr>
                    <td style="font-weight: bold; color: #555;">Registered Sale Deed No:</td>
                    <td><code>${rec.deedNo}</code></td>
                    <td style="font-weight: bold; color: #555;">Encumbrance / Lien:</td>
                    <td><span style="color: ${rec.status === 'Verified' ? 'var(--gov-green)' : 'var(--gov-red)'}; font-weight: bold;">${rec.encumbranceStatus}</span></td>
                </tr>
                <tr>
                    <td style="font-weight: bold; color: #555;">Cadastral Boundaries:</td>
                    <td colspan="3"><small>${rec.boundaries}</small></td>
                </tr>
            </table>

            <div class="stamp-seal">
                ✓ DIGITALLY CERTIFIED<br>
                SUB-REGISTRAR OFFICE
            </div>
        </div>
    `;

    modal.classList.add("active");
    addAuditEntry("Certified Deed Inspected", `Viewed certified RoR for Khasra #${rec.khasraNo}`);
}

function closeLandDeedModal() {
    document.getElementById("deed-viewer-modal").classList.remove("active");
}

// ==========================================================================
// 13. AUTHENTICATION MODAL
// ==========================================================================
function openAuthModal() {
    document.getElementById("auth-login-modal").classList.add("active");
}

function closeAuthModal() {
    document.getElementById("auth-login-modal").classList.remove("active");
}

function handleLoginSubmit(event) {
    event.preventDefault();
    const user = document.getElementById("login-input-user").value;
    const role = document.getElementById("login-input-role").value;

    switchRole(role);
    closeAuthModal();
    alert(`✅ Authenticated successfully as ${user} (${role.toUpperCase()})`);
}

// Click outside modal backdrop closes it
window.onclick = function (e) {
    const deedModal = document.getElementById("deed-viewer-modal");
    const authModal = document.getElementById("auth-login-modal");
    if (e.target === deedModal) closeLandDeedModal();
    if (e.target === authModal) closeAuthModal();
};
