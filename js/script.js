const pages = ["dashboard", "upload", "validation", "result", "profile"];

const state = {
  selectedFile: null,
  objectUrl: null
};

const loginPage = document.getElementById("loginPage");
const appShell = document.getElementById("appShell");
const toast = document.getElementById("toast");
const pageTitle = document.getElementById("pageTitle");
const sidebar = document.getElementById("sidebar");
const overlay = document.getElementById("overlay");

const pageNames = {
  dashboard: "Dashboard",
  upload: "Upload & OCR",
  validation: "Validation",
  result: "Land Record Result",
  profile: "Profile"
};

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("show");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => toast.classList.remove("show"), 2600);
}

function showPage(page) {
  if (!pages.includes(page)) page = "dashboard";

  document.querySelectorAll(".page").forEach(el => el.classList.remove("active-page"));
  const target = document.getElementById(`${page}Page`);
  if (target) target.classList.add("active-page");

  document.querySelectorAll(".nav-item[data-page]").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.page === page);
  });

  pageTitle.textContent = pageNames[page];
  closeSidebar();
  window.scrollTo({ top: 0, behavior: "smooth" });

  history.replaceState(null, "", `#${page}`);
}

function openApp() {
  loginPage.classList.add("hidden");
  appShell.classList.remove("hidden");
  showPage(location.hash.replace("#", "") || "dashboard");
}

function logout() {
  appShell.classList.add("hidden");
  loginPage.classList.remove("hidden");
  document.getElementById("loginForm").reset();
  history.replaceState(null, "", "#login");
  showToast("You have been signed out.");
}

document.getElementById("loginForm").addEventListener("submit", (event) => {
  event.preventDefault();
  openApp();
});

document.getElementById("togglePassword").addEventListener("click", () => {
  const password = document.getElementById("password");
  const button = document.getElementById("togglePassword");
  const isPassword = password.type === "password";
  password.type = isPassword ? "text" : "password";
  button.textContent = isPassword ? "Hide" : "Show";
  button.setAttribute("aria-label", isPassword ? "Hide password" : "Show password");
});

document.getElementById("forgotPassword").addEventListener("click", () => {
  showToast("Password recovery can be started from this option.");
});

document.getElementById("logoutBtn").addEventListener("click", logout);

document.querySelectorAll("[data-page]").forEach(button => {
  button.addEventListener("click", () => showPage(button.dataset.page));
});

document.querySelectorAll("[data-go]").forEach(button => {
  button.addEventListener("click", () => showPage(button.dataset.go));
});

function openSidebar() {
  sidebar.classList.add("open");
  overlay.classList.add("show");
}
function closeSidebar() {
  sidebar.classList.remove("open");
  overlay.classList.remove("show");
}
document.getElementById("openSidebar").addEventListener("click", openSidebar);
document.getElementById("closeSidebar").addEventListener("click", closeSidebar);
overlay.addEventListener("click", closeSidebar);

const fileInput = document.getElementById("fileInput");
const dropZone = document.getElementById("dropZone");
const fileCard = document.getElementById("fileCard");
const previewArea = document.getElementById("previewArea");
const previewContent = document.getElementById("previewContent");
const fileName = document.getElementById("fileName");
const fileMeta = document.getElementById("fileMeta");
const fileIcon = document.getElementById("fileIcon");
const previewType = document.getElementById("previewType");
const ocrStatus = document.getElementById("ocrStatus");
const ocrMessage = document.getElementById("ocrMessage");
const startOcr = document.getElementById("startOcr");

const allowedTypes = ["image/jpeg", "image/png", "application/pdf"];
const allowedExtensions = [".jpg", ".jpeg", ".png", ".pdf"];
const maxSize = 10 * 1024 * 1024;

function formatSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function isAllowed(file) {
  const lower = file.name.toLowerCase();
  const extensionOk = allowedExtensions.some(ext => lower.endsWith(ext));
  return allowedTypes.includes(file.type) || extensionOk;
}

function resetFile() {
  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
  state.selectedFile = null;
  state.objectUrl = null;
  fileInput.value = "";
  fileCard.classList.add("hidden");
  previewArea.classList.add("hidden");
  previewContent.innerHTML = "";
  ocrStatus.textContent = "Ready";
  ocrStatus.classList.remove("neutral");
  ocrMessage.textContent = "Upload a land record to begin OCR.";
  startOcr.disabled = false;
}

function handleFile(file) {
  if (!file) return;

  if (!isAllowed(file)) {
    showToast("Please upload a JPG, PNG, JPEG or PDF file.");
    return;
  }

  if (file.size > maxSize) {
    showToast("File size exceeds the allowed limit.");
    return;
  }

  if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);

  state.selectedFile = file;
  state.objectUrl = URL.createObjectURL(file);

  fileName.textContent = file.name;
  fileMeta.textContent = `${file.type || "File"} · ${formatSize(file.size)}`;
  fileIcon.textContent = file.name.toLowerCase().endsWith(".pdf") ? "PDF" : "IMG";
  fileCard.classList.remove("hidden");

  previewArea.classList.remove("hidden");
  previewType.textContent = file.name.toLowerCase().endsWith(".pdf") ? "PDF document" : "Image preview";

  if (file.type.startsWith("image/")) {
    previewContent.innerHTML = "";
    const img = document.createElement("img");
    img.src = state.objectUrl;
    img.alt = "Selected land record preview";
    previewContent.appendChild(img);
  } else {
    previewContent.innerHTML = `
      <div class="pdf-preview">
        <strong>PDF document selected</strong><br>
        <span>The selected PDF is ready for the document-processing workflow.</span>
      </div>
    `;
  }

  ocrStatus.textContent = "Ready";
  ocrMessage.textContent = "OCR is ready to process the selected document.";
  showToast("Land record selected.");
}

fileInput.addEventListener("change", event => handleFile(event.target.files[0]));
document.getElementById("removeFile").addEventListener("click", resetFile);

["dragenter", "dragover"].forEach(type => {
  dropZone.addEventListener(type, event => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach(type => {
  dropZone.addEventListener(type, event => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", event => handleFile(event.dataTransfer.files[0]));

startOcr.addEventListener("click", () => {
  if (!state.selectedFile) {
    showToast("Please select a land record first.");
    return;
  }
  ocrStatus.textContent = "Ready";
  ocrMessage.textContent = "OCR is ready to process the selected document.";
  showToast("OCR is ready to process the selected document.");
});

document.getElementById("validateBtn").addEventListener("click", () => {
  showToast("Validation is ready to process the extracted record.");
});

window.addEventListener("hashchange", () => {
  if (!appShell.classList.contains("hidden")) {
    showPage(location.hash.replace("#", "") || "dashboard");
  }
});

if (location.hash && location.hash !== "#login") {
  openApp();
}