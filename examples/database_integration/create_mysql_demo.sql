-- =============================================================
-- MySQL Setup Script — Healthcare Demo Database
-- Team P23-T1 | Kennesaw State University
--
-- Creates the full 6-entity healthcare claims schema matching
-- the project's Domain Specification Document.
--
-- Run with:
--   mysql -u root -p < create_mysql_demo.sql
-- =============================================================

CREATE DATABASE IF NOT EXISTS healthcare_demo;
USE healthcare_demo;

-- Drop tables in reverse dependency order (safe re-run)

DROP TABLE IF EXISTS payment;
DROP TABLE IF EXISTS adjudication;
DROP TABLE IF EXISTS claim;
DROP TABLE IF EXISTS diagnosis;
DROP TABLE IF EXISTS provider;
DROP TABLE IF EXISTS patient;

-- -------------------------------------------------------------
-- Patient
-- -------------------------------------------------------------

CREATE TABLE patient (
    patient_id    INT             NOT NULL AUTO_INCREMENT,
    patient_name  VARCHAR(255)    NOT NULL,
    age           INT,
    gender        VARCHAR(50),
    date_of_birth DATE,
    PRIMARY KEY (patient_id)
);

-- -------------------------------------------------------------
-- Provider
-- -------------------------------------------------------------

CREATE TABLE provider (
    provider_id     INT             NOT NULL AUTO_INCREMENT,
    provider_name   VARCHAR(255)    NOT NULL,
    specialty       VARCHAR(255),
    license_number  VARCHAR(100),
    facility_id     VARCHAR(100),
    PRIMARY KEY (provider_id)
);

-- -------------------------------------------------------------
-- Diagnosis
-- Patient → Diagnosis ← Provider
-- -------------------------------------------------------------

CREATE TABLE diagnosis (
    diagnosis_id    INT             NOT NULL AUTO_INCREMENT,
    patient_id      INT             NOT NULL,
    provider_id     INT             NOT NULL,
    diagnosis_code  VARCHAR(50),
    visit_date      DATE,
    PRIMARY KEY (diagnosis_id),
    FOREIGN KEY (patient_id)  REFERENCES patient(patient_id),
    FOREIGN KEY (provider_id) REFERENCES provider(provider_id)
);

-- -------------------------------------------------------------
-- Claim
-- Diagnosis → Claim ← Patient, Provider
-- -------------------------------------------------------------

CREATE TABLE claim (
    claim_id        INT             NOT NULL AUTO_INCREMENT,
    patient_id      INT             NOT NULL,
    provider_id     INT             NOT NULL,
    diagnosis_id    INT             NOT NULL,
    procedure_code  VARCHAR(50),
    claim_amount    DECIMAL(10, 2),
    submission_date DATE,
    PRIMARY KEY (claim_id),
    FOREIGN KEY (patient_id)   REFERENCES patient(patient_id),
    FOREIGN KEY (provider_id)  REFERENCES provider(provider_id),
    FOREIGN KEY (diagnosis_id) REFERENCES diagnosis(diagnosis_id)
);

-- -------------------------------------------------------------
-- Adjudication
-- Claim → Adjudication
-- -------------------------------------------------------------

CREATE TABLE adjudication (
    adjudication_id INT             NOT NULL AUTO_INCREMENT,
    claim_id        INT             NOT NULL,
    decision        ENUM('Approved', 'Denied', 'Partial'),
    denial_reason   VARCHAR(255),
    approved_amount DECIMAL(10, 2),
    PRIMARY KEY (adjudication_id),
    FOREIGN KEY (claim_id) REFERENCES claim(claim_id)
);

-- -------------------------------------------------------------
-- Payment
-- Claim → Payment
-- -------------------------------------------------------------

CREATE TABLE payment (
    payment_id      INT             NOT NULL AUTO_INCREMENT,
    claim_id        INT             NOT NULL,
    payment_date    DATE,
    payment_amount  DECIMAL(10, 2),
    status          ENUM('Paid', 'Pending', 'NoPay'),
    PRIMARY KEY (payment_id),
    FOREIGN KEY (claim_id) REFERENCES claim(claim_id)
);

-- -------------------------------------------------------------
-- Sample rows (enough to verify FK chains work)
-- -------------------------------------------------------------

INSERT INTO patient (patient_name, age, gender, date_of_birth)
VALUES
    ('Alice Johnson', 34, 'F', '1991-03-15'),
    ('Bob Martinez',  52, 'M', '1972-07-22');

INSERT INTO provider (provider_name, specialty, license_number, facility_id)
VALUES
    ('Dr. Sarah Lee',    'Cardiology',      'LIC-001', 'FAC-A'),
    ('Dr. James Patel',  'General Practice','LIC-002', 'FAC-B');

INSERT INTO diagnosis (patient_id, provider_id, diagnosis_code, visit_date)
VALUES
    (1, 1, 'I10',   '2025-01-10'),
    (2, 2, 'E11.9', '2025-02-05');

INSERT INTO claim (patient_id, provider_id, diagnosis_id, procedure_code, claim_amount, submission_date)
VALUES
    (1, 1, 1, '99213', 250.00, '2025-01-15'),
    (2, 2, 2, '99214', 175.00, '2025-02-10');

INSERT INTO adjudication (claim_id, decision, denial_reason, approved_amount)
VALUES
    (1, 'Approved', NULL,              250.00),
    (2, 'Partial',  'Plan limit reached', 100.00);

INSERT INTO payment (claim_id, payment_date, payment_amount, status)
VALUES
    (1, '2025-01-20', 250.00, 'Paid'),
    (2, '2025-02-15', 100.00, 'Paid');

SELECT 'MySQL healthcare_demo database created successfully.' AS status;