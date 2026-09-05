/* ==========================================================================
   LAND RECORD MANAGEMENT & VALIDATION SYSTEM
   JavaScript Data Module - js/data.js
   Contains realistic Indian land records (Khasra/Khatauni), pre-configured
   duplicate alerts, OCR sample profiles, and audit trail logs.
   ========================================================================== */

// 1. Initial Land Records (Bhoomi / Dharani / RoR Format)
const INITIAL_LAND_RECORDS = [
    {
        id: "LR-2024-001",
        khasraNo: "304/1",
        khatauniNo: "KH-0089",
        deedNo: "DEED-2021-9981",
        ownerName: "Ramesh Kumar Sharma",
        fatherName: "Late Sh. Badri Prasad Sharma",
        district: "Jaipur Rural",
        tehsil: "Rampur",
        village: "Kishorpura",
        areaAcres: 3.45,
        soilType: "Agricultural (Irrigated)",
        marketValue: "₹ 48,50,000",
        regDate: "2021-03-15",
        status: "Verified",
        encumbranceStatus: "Nil / Clear Title",
        boundaries: "North: Canal Road, South: Plot 303, East: Village Road, West: Plot 304/2"
    },
    {
        id: "LR-2024-002",
        khasraNo: "112/A",
        khatauniNo: "KH-0142",
        deedNo: "DEED-2022-3118",
        ownerName: "Sunita Devi",
        fatherName: "W/o Late Om Prakash Verma",
        district: "Jaipur Rural",
        tehsil: "Sadar",
        village: "Rampur",
        areaAcres: 2.50,
        soilType: "Agricultural (Dry Crop)",
        marketValue: "₹ 35,00,000",
        regDate: "2019-11-12",
        status: "Flagged",
        encumbranceStatus: "Dispute Flagged / Overlapping Claim",
        boundaries: "North: Panchayat Pathway, South: River Stream, East: Plot 113, West: Plot 111"
    },
    {
        id: "LR-2024-003",
        khasraNo: "88/1",
        khatauniNo: "KH-0056",
        deedNo: "DEED-2023-4412",
        ownerName: "Rajesh Dattatray Patil",
        fatherName: "Sh. Dattatray Patil",
        district: "Pune North",
        tehsil: "Haveli",
        village: "Shivajinagar",
        areaAcres: 4.10,
        soilType: "Residential Plot",
        marketValue: "₹ 1,20,00,000",
        regDate: "2023-08-20",
        status: "Flagged",
        encumbranceStatus: "Excess Area Encroachment Warning",
        boundaries: "North: State Highway 4, South: Canal, East: Survey 89, West: Survey 87"
    },
    {
        id: "LR-2024-004",
        khasraNo: "204/C",
        khatauniNo: "KH-0771",
        deedNo: "DEED-2022-7721",
        ownerName: "Priya S. Nair",
        fatherName: "Sh. S. K. Nair",
        district: "Indore Central",
        tehsil: "Mhow",
        village: "Kalyanpura",
        areaAcres: 1.80,
        soilType: "Agricultural (Irrigated)",
        marketValue: "₹ 28,00,000",
        regDate: "2022-05-19",
        status: "Verified",
        encumbranceStatus: "Nil / Clear Title",
        boundaries: "North: Well #4, South: Farm Road, East: Plot 205, West: Plot 203"
    },
    {
        id: "LR-2024-005",
        khasraNo: "312",
        khatauniNo: "KH-0290",
        deedNo: "DEED-2020-3341",
        ownerName: "Mohd. Arif Khan",
        fatherName: "Sh. Bashir Khan",
        district: "Lucknow Cantt",
        tehsil: "Bakshi Ka Talab",
        village: "Madhavpur",
        areaAcres: 5.20,
        soilType: "Commercial Land",
        marketValue: "₹ 1,75,00,000",
        regDate: "2020-09-04",
        status: "Verified",
        encumbranceStatus: "Nil / Commercial Bank NOC Verified",
        boundaries: "North: Ring Road, South: Storage Warehouse, East: Petrol Pump, West: Plot 311"
    },
    {
        id: "LR-2024-006",
        khasraNo: "501/2",
        khatauniNo: "KH-0914",
        deedNo: "DEED-2024-0012",
        ownerName: "Vikramaditya Singh",
        fatherName: "Sh. Tejendra Singh",
        district: "Jaipur Rural",
        tehsil: "Amer",
        village: "Kukas",
        areaAcres: 6.75,
        soilType: "Agricultural (Irrigated)",
        marketValue: "₹ 82,00,000",
        regDate: "2024-01-10",
        status: "Pending",
        encumbranceStatus: "Pending Field Patwari Physical Verification",
        boundaries: "North: Aravalli Foothill, South: Highway, East: Survey 502, West: Forest Buffer"
    }
];

// 2. Pre-configured Duplicate Fraud & Discrepancy Alerts
const INITIAL_DUPLICATE_ALERTS = [
    {
        id: "ALT-2024-01",
        khasraNo: "112/A",
        severity: "CRITICAL",
        title: "High-Risk Overlapping Khasra Claim (Double Registration Fraud)",
        description: "Two distinct sale deeds have been registered for the exact same parcel within 6 months without succession mutation.",
        primaryClaimant: {
            name: "Sunita Devi",
            deedNo: "DEED-2019-1082",
            area: "2.50 Acres",
            date: "12-Nov-2019"
        },
        conflictingClaimant: {
            name: "Rajesh Verma",
            deedNo: "DEED-2024-5501",
            area: "2.50 Acres",
            date: "02-Feb-2024"
        },
        location: "Village Rampur, Tehsil Sadar, Jaipur Rural",
        ruleTriggered: "Rule DILRMP-04: Non-Permitted Parallel Conveyance Deed",
        status: "Action Required"
    },
    {
        id: "ALT-2024-02",
        khasraNo: "88/1",
        severity: "HIGH",
        title: "Cadastral Boundary Overlap with Government Grazing Land",
        description: "Submitted deed claims 4.95 Acres, but digitized Cadastral Naksha records only 4.10 Acres. Excess 0.85 Acres encroaches on protected State Gauchar Bhoomi.",
        primaryClaimant: {
            name: "Rajesh Dattatray Patil",
            deedNo: "DEED-2023-4412",
            area: "4.95 Acres (Disputed)",
            date: "20-Aug-2023"
        },
        conflictingClaimant: {
            name: "State Government (Gram Panchayat)",
            deedNo: "GOV-GAUCHAR-1972",
            area: "0.85 Acres Overlap",
            date: "15-Aug-1972"
        },
        location: "Village Shivajinagar, Tehsil Haveli, Pune North",
        ruleTriggered: "Rule SEC-14: Prohibition of Alienation on Common Village Commons",
        status: "Action Required"
    },
    {
        id: "ALT-2024-03",
        khasraNo: "220/4",
        severity: "MEDIUM",
        title: "Rapid Resale Warning (Multiple Sale Attempts in < 30 Days)",
        description: "Plot was conveyed twice in 18 days with 150% valuation surge, characteristic of speculative circular registration or benami transaction.",
        primaryClaimant: {
            name: "Anil Mittal",
            deedNo: "DEED-2024-8110",
            area: "1.20 Acres",
            date: "14-Feb-2024"
        },
        conflictingClaimant: {
            name: "Deepak Soni",
            deedNo: "DEED-2024-8902",
            area: "1.20 Acres",
            date: "01-Mar-2024"
        },
        location: "Village Madhavpur, Lucknow Cantt",
        ruleTriggered: "Rule KYC-09: Suspicious High-Velocity Property Flipping",
        status: "Under Review"
    }
];

// 3. Initial Tamper-Proof Audit Logs
const INITIAL_AUDIT_LOGS = [
    {
        id: "LOG-901",
        timestamp: "2026-09-05 11:20:14",
        userId: "PAT-402",
        role: "Patwari",
        action: "OCR Document Ingestion",
        recordRef: "Khasra 304/1",
        ipAddress: "192.168.1.42",
        status: "Success"
    },
    {
        id: "LOG-902",
        timestamp: "2026-09-05 10:45:00",
        userId: "AI-ENGINE",
        role: "System",
        action: "Automated Duplicate Check Triggered",
        recordRef: "Khasra 112/A",
        ipAddress: "127.0.0.1",
        status: "Warning Alert"
    },
    {
        id: "LOG-903",
        timestamp: "2026-09-04 17:15:32",
        userId: "TEH-101",
        role: "Super Admin",
        action: "Dispute Freeze Applied",
        recordRef: "Khasra 88/1",
        ipAddress: "192.168.1.05",
        status: "Success"
    },
    {
        id: "LOG-904",
        timestamp: "2026-09-04 14:02:18",
        userId: "CITIZEN-PUB",
        role: "Citizen",
        action: "Certified Land Deed RoR View",
        recordRef: "Khasra 204/C",
        ipAddress: "103.22.45.19",
        status: "Success"
    },
    {
        id: "LOG-905",
        timestamp: "2026-09-03 09:30:00",
        userId: "SYS-CRON",
        role: "System",
        action: "Daily Cadastral Integrity Scan",
        recordRef: "All 18 Villages",
        ipAddress: "10.0.0.1",
        status: "Completed"
    }
];

// 4. Automated 5-Point Verification Checklist Definitions
const VERIFICATION_CHECKLIST = [
    {
        id: "chk-1",
        rule: "Aadhaar / KYC Identity Verification",
        description: "Primary and secondary seller biometrics cross-referenced with UIDAI portal."
    },
    {
        id: "chk-2",
        rule: "Treasury e-Challan Stamp Duty Payment",
        description: "Verification of treasury GRN receipt against state revenue collection portal."
    },
    {
        id: "chk-3",
        rule: "12-Year Non-Encumbrance Clearance",
        description: "Searches civil court records and mortgage liens from registered cooperative/commercial banks."
    },
    {
        id: "chk-4",
        rule: "GIS Cadastral Naksha Boundary Overlap",
        description: "Satellite boundary polygon coordinates checked against village settlement map."
    },
    {
        id: "chk-5",
        rule: "Succession & Prior Mutation Clearance",
        description: "Confirms undisputed succession genealogy tree (Virasat) and prior registered cancellations."
    }
];
