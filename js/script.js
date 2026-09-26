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
document.getElementById("loginForm").addEventListener("submit", async (event) => {
  event.preventDefault();

  const email = document.getElementById("email").value.trim();
  const password = document.getElementById("password").value;

  if (!email || !password) {
    showToast("Please enter email and password.");
    return;
  }

  try {
    const response = await fetch("http://127.0.0.1:5000/api/auth/login", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        email: email,
        password: password
      })
    });

    const data = await response.json();

    if (!response.ok) {
      showToast(data.message || "Login failed.");
      return;
    }

    // Save authentication information
    localStorage.setItem("dharohar_token", data.access_token);
    localStorage.setItem("dharohar_user", JSON.stringify(data.user));

    showToast("Login successful.");

    openApp();

  } catch (error) {
    console.error("Login error:", error);
    showToast("Cannot connect to DHAROHAR server.");
  }
});

// ============================================================
// LOGIN / SIGNUP SWITCHING
// ============================================================

const loginForm = document.getElementById("loginForm");
const signupForm = document.getElementById("signupForm");
const showSignup = document.getElementById("showSignup");
const showLogin = document.getElementById("showLogin");

showSignup.addEventListener("click", () => {
  loginForm.classList.add("hidden");
  signupForm.classList.remove("hidden");
});

showLogin.addEventListener("click", () => {
  signupForm.classList.add("hidden");
  loginForm.classList.remove("hidden");
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

startOcr.addEventListener("click", async () => {
  if (!state.selectedFile) {
    showToast("Please select a land record first.");
    return;
  }

  const token = localStorage.getItem("dharohar_token");

  if (!token) {
    showToast("Please sign in again.");
    logout();
    return;
  }

  try {
    startOcr.disabled = true;
    ocrStatus.textContent = "Processing";
    ocrStatus.classList.add("neutral");
    ocrMessage.textContent = "Uploading document and processing OCR...";
    showToast("Uploading land record...");

    // --------------------------------------------------------
    // STEP 1: Upload document
    // --------------------------------------------------------

    const formData = new FormData();
    formData.append("file", state.selectedFile);

    const uploadResponse = await fetch(
      "http://127.0.0.1:5000/api/documents/upload",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`
        },
        body: formData
      }
    );

    const uploadData = await uploadResponse.json();

    if (!uploadResponse.ok) {
      throw new Error(
        uploadData.message || "Document upload failed."
      );
    }

    const documentId =
      uploadData.document?.id ||
      uploadData.id ||
      uploadData.document_id;

    if (!documentId) {
      throw new Error("Upload succeeded but document ID was not returned.");
    }

    showToast("Document uploaded. Starting OCR...");
    ocrMessage.textContent = "Document uploaded. Running OCR...";

    // --------------------------------------------------------
    // STEP 2: Run OCR
    // --------------------------------------------------------

    const ocrResponse = await fetch(
      `http://127.0.0.1:5000/api/documents/${documentId}/ocr`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${token}`
        }
      }
    );

    const ocrData = await ocrResponse.json();

    if (!ocrResponse.ok) {
      throw new Error(
        ocrData.message ||
        ocrData.error ||
        "OCR processing failed."
      );
    }

    // --------------------------------------------------------
    // STEP 3: Display OCR result
    // --------------------------------------------------------

    ocrStatus.textContent = "Completed";
    ocrStatus.classList.remove("neutral");

    const confidence =
      ocrData.ocr?.confidence ??
      ocrData.confidence ??
      0;

    const extractedText =
      ocrData.ocr?.extracted_text ??
      ocrData.extracted_text ??
      "";

    ocrMessage.textContent =
      `OCR completed successfully. Confidence: ${confidence}%`;

    showToast("OCR completed successfully.");

    // Save result for Result page
    localStorage.setItem(
      "dharohar_ocr_result",
      JSON.stringify({
        document_id: documentId,
        filename: state.selectedFile.name,
        confidence: confidence,
        extracted_text: extractedText
      })
    );

    // Go to result page
    showPage("result");

  } catch (error) {
    console.error("OCR workflow error:", error);

    ocrStatus.textContent = "Failed";
    ocrMessage.textContent = error.message;

    showToast(error.message);

    startOcr.disabled = false;
  }
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

// ============================================================
// SIGNUP / REGISTRATION
// ============================================================

signupForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const name = document.getElementById("signupName").value.trim();
  const email = document.getElementById("signupEmail").value.trim();
  const password = document.getElementById("signupPassword").value;
  const role = document.getElementById("signupRole").value;

  if (!name || !email || !password || !role) {
    showToast("Please fill all fields.");
    return;
  }

  try {
    const response = await fetch(
      "http://127.0.0.1:5000/api/auth/register",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          name: name,
          email: email,
          password: password,
          role: role
        })
      }
    );

    const data = await response.json();

    if (!response.ok) {
      showToast(data.message || "Registration failed.");
      return;
    }

    showToast("Account created successfully.");

    // Clear signup form
    signupForm.reset();

    // Return to login
    signupForm.classList.add("hidden");
    loginForm.classList.remove("hidden");

    // Put registered email into login form
    document.getElementById("email").value = email;

  } catch (error) {
    console.error("Signup error:", error);
    showToast("Cannot connect to DHAROHAR server.");
  }
});